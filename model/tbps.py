import torch
import torch.nn as nn
from loguru import logger

from model import objectives

class TBPS(nn.Module):
    def __init__(self, config, backbone):
        super().__init__()
        self.config = config

        self.backbone = backbone
        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model
        self.embed_dim = config.backbone.embedding_dim

        self.use_sigmoid = config.backbone.use_sigmoid
        if not hasattr(self.backbone, "logit_bias"):
            self.backbone.logit_bias = 0

        self.contain_visual_projection, self.contain_text_projection = (
            self.check_contain_projection()
        )
        if config.loss.get("SS", None):
            self.simclr_mlp = self._build_mlp(
                self.embed_dim, self.embed_dim, self.embed_dim
            )

    def _build_mlp(self, in_dim=512, mlp_dim=128, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
        )

    def check_contain_projection(self):
        """
        Check if the backbone contains visual and text projection.
        """
        contain_visual_projection = hasattr(self.backbone, "visual_projection")
        logger.info(
            f"Contain visual projection: {contain_visual_projection}"
        ) if contain_visual_projection else None
        contain_text_projection = hasattr(self.backbone, "text_projection")
        logger.info(
            f"Contain text projection: {contain_text_projection}"
        ) if contain_text_projection else None
        return contain_visual_projection, contain_text_projection

    def encode_image(self, image, return_last_hidden=False):
        """
        Encodes an image into a tensor.

        Args:
            image (PIL.Image): Image to be encoded.
            return_last_hidden (bool): Whether to return the last hidden state.

        Returns:
            torch.Tensor: The encoded image, use pooler_output as the image embedding."""
        x = self.vision_model(image)
        pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state

        if not hasattr(self.backbone, "visual_projection"):
            # If no projection layer exists, normalize and return
            # normalized_pooler = pooler_output / pooler_output.norm(dim=1, keepdim=True)
            if return_last_hidden:
                return pooler_output, last_hidden
            return pooler_output

        # If projection layer exists, project both outputs
        pooler_output = self.backbone.visual_projection(pooler_output)
        # pooler_output = pooler_output / pooler_output.norm(dim=1, keepdim=True)

        if return_last_hidden:
            last_hidden = self.backbone.visual_projection(last_hidden)
            return pooler_output, last_hidden

        return pooler_output

    def encode_text(self, text, return_last_hidden=False):
        """
        Encodes text into a tensor.

        Args:
            text (dict): Text to be encoded, containing the keys "input_ids" and "attention_mask".
            return_last_hidden (bool): Whether to return the last hidden state.
        Returns:
            torch.Tensor: The encoded text, use pooler_output as the sentence embedding."""
        # text will be a dict that has forms {"input_ids": ..., "attention_mask": ...}
        x = self.text_model(**text)
        pooler_output = x.pooler_output
        last_hidden = x.last_hidden_state

        if not hasattr(self.backbone, "text_projection"):
            # If no projection layer exists, normalize and return
            # normalized_pooler = pooler_output / pooler_output.norm(dim=1, keepdim=True)
            if return_last_hidden:
                return pooler_output, last_hidden
            return pooler_output

        # If projection layer exists, project both outputs
        pooler_output = self.backbone.text_projection(pooler_output)
        # pooler_output = pooler_output / pooler_output.norm(dim=1, keepdim=True)

        if return_last_hidden:
            last_hidden = self.backbone.text_projection(last_hidden)
            return pooler_output, last_hidden

        return pooler_output

    def prepare_sim_targets(self, pids, use_sigmoid=False):
        """
        Prepare similarity targets for constrative learning.

        Args:
            pids (torch.Tensor): Tensor containing the person IDs.
        Returns:
            torch.Tensor: Tensor containing the similarity targets.
        """
        sim_targets = torch.eq(pids.view(-1, 1), pids.view(1, -1)).float()
        if use_sigmoid:
            sim_targets = (
                -torch.ones_like(sim_targets) + 2 * sim_targets
            )  # -1 if different, 1 if same
            return sim_targets

        return sim_targets / sim_targets.sum(
            dim=1, keepdim=True
        )

    def forward(self, batch, current_epoch=None):
        ret = dict()

        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        logit_scale = self.backbone.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        image_pooler_output = self.encode_image(images)
        caption_pooler_output = self.encode_text(caption_input)

        ret.update({"temperature": self.backbone.logit_scale})
        ret.update({"bias": self.backbone.logit_bias})

        # --- A. SimCLR (Self-Supervised) ---
        if self.config.loss.get("SS", None):
            ss_images1_embed = self.simclr_mlp(self.encode_image(batch["ss_images1"]))
            ss_images2_embed = self.simclr_mlp(self.encode_image(batch["ss_images2"]))
            ss_loss = (
                objectives.compute_simclr(
                    ss_images1_embed,
                    ss_images2_embed,
                    self.config.loss.simclr_temperature,
                )
                * self.config.loss.ss_loss_weight
            )
            ret.update({"ss_loss": ss_loss})

        # --- CURRICULUM WEIGHT CALCULATION ---
        if current_epoch is None:
            current_epoch = 0

        T_warmup = 5
        T_ramp_len = 15
        T_stable = 20
        Lambda_target = self.config.loss.get("circle_loss_weight", 0.1)

        if current_epoch <= T_warmup:
            current_circle_weight = 0.0
        elif current_epoch <= T_stable:
            progress = (current_epoch - T_warmup) / T_ramp_len
            current_circle_weight = progress * Lambda_target
        else:
            current_circle_weight = Lambda_target

        ret.update({"circle_loss_weight": torch.tensor(current_circle_weight, device=image_pooler_output.device)})

        # --- B. N-ITC ---
        if self.config.loss.get("NITC", None):
            sim_targets = self.prepare_sim_targets(batch["pids"], self.use_sigmoid)

            nitc_loss = objectives.compute_constrative(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                sim_targets=sim_targets,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                use_sigmoid=self.use_sigmoid,
            )

            if self.config.loss.get("MVS", None):
                aug_images = batch["aug_images"]
                aug_images_features = self.encode_image(aug_images)
                augmented_nitc_loss = objectives.compute_constrative(
                    image_features=image_pooler_output,
                    text_features=caption_pooler_output,
                    sim_targets=sim_targets,
                    logit_scale=logit_scale,
                    use_sigmoid=self.use_sigmoid,
                    logit_bias=self.backbone.logit_bias,
                )
                nitc_loss = (nitc_loss + augmented_nitc_loss) / 2

            ret.update({"nitc_loss": nitc_loss * self.config.loss.nitc_loss_weight})

        # --- C. Cross-Modal Circle Loss (Curriculum) ---
        if self.config.loss.get("CIR", None) and current_circle_weight > 0:
            circle_m = self.config.loss.get("circle_margin", 0.25)
            circle_gamma = self.config.loss.get("circle_gamma", 128)

            cm_circle_loss = objectives.compute_cross_modal_circle(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                pids=batch["pids"],
                m=circle_m,
                gamma=circle_gamma
            )

            final_circle_loss = cm_circle_loss

            if self.config.loss.get("MVS", None):
                if 'aug_images_features' not in locals():
                    aug_images = batch["aug_images"]
                    aug_images_features = self.encode_image(aug_images)

                aug_cm_circle_loss = objectives.compute_cross_modal_circle(
                    image_features=aug_images_features,
                    text_features=caption_pooler_output,
                    pids=batch["pids"],
                    m=circle_m,
                    gamma=circle_gamma
                )
                final_circle_loss = (cm_circle_loss + aug_cm_circle_loss) / 2

            ret.update({"circle_loss": final_circle_loss * current_circle_weight})

        elif self.config.loss.get("CIR", None):
            ret.update({"circle_loss": torch.tensor(0.0, device=image_pooler_output.device, requires_grad=True)})

        # --- D. C-ITC ---
        if self.config.loss.get("CITC", None):
            loss = objectives.compute_citc(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                inmodal_weight=self.config.loss.citc_inmodal_weight,
                intermodal_weight=self.config.loss.citc_intermodal_weight,
            )
            ret.update({"citc_loss": loss * self.config.loss.citc_loss_weight})

        return ret