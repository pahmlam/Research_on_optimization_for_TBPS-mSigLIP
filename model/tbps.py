from collections import OrderedDict

import torch
import torch.nn as nn
from loguru import logger

from model import objectives
from model.layers import Transformer, QuickGELU, LayerNorm
from model.objectives import compute_cir

TASK_LIST = ["ITC", "SDM", "CMPM", "ID", "MLM", "SS", "MVS", "RITC", "CITC", "NITC", "CIR"]


class TBPS(nn.Module):
    def __init__(self, config, backbone, vocab_size, pad_token_id, num_classes=11003):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.backbone = backbone
        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model
        self.embed_dim = config.backbone.embedding_dim

        if config.backbone.freeze.vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            logger.info("Freezing vision backbone")
        if config.backbone.freeze.text:
            for param in self.text_model.parameters():
                param.requires_grad = False
            logger.info("Freezing text backbone")

        self.use_sigmoid = config.backbone.use_sigmoid
        if not hasattr(self.backbone, "logit_bias"):
            self.backbone.logit_bias = 0

        task_lists = []
        for task in TASK_LIST:
            try:
                if config.loss.get(task, None):
                    task_lists.append(task)
            except AttributeError:
                pass
        logger.info(f"Tasks: {task_lists}")

        self.contain_visual_projection, self.contain_text_projection = (
            self.check_contain_projection()
        )
        if config.loss.get("SS", None):
            self.simclr_mlp = self._build_mlp(
                self.embed_dim, self.embed_dim, self.embed_dim
            )

        if config.loss.get("ID", None):
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
            logger.info(f"Classifier initialized for ID Loss.")

        if config.loss.get("MLM", None):
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=config.loss.get("cmt_depth", 4),
                heads=self.embed_dim // 64,
            )
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict(
                    [
                        ("dense", nn.Linear(self.embed_dim, self.embed_dim)),
                        ("gelu", QuickGELU()),
                        ("ln", LayerNorm(self.embed_dim)),
                        (
                            "fc",
                            nn.Linear(self.embed_dim, self.vocab_size),
                        ),
                    ]
                )
            )
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

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

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q), self.ln_pre_i(k), self.ln_pre_i(v), need_weights=False
        )[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

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

    def forward(self, batch, alpha, weights=None, current_epoch=None):
        """
        Forward pass of the model with Curriculum Learning support.

        Args:
            batch (dict): A dictionary containing the input data.
            alpha (float): Soft label ratio.
            weights (torch.Tensor, optional): Boosting weights.
            current_epoch (int, optional): The current training epoch (0-indexed).
        """
        ret = dict()

        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        logit_scale = self.backbone.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        # --- ENCODING FEATURES ---
        if self.config.loss.get("MLM", None) or self.config.loss.get("Ring", None):
            image_pooler_output, image_last_hidden = self.encode_image(images, True)
            if self.config.loss.get("Ring", None):
                caption_pooler_output, caption_last_hidden = self.encode_text(
                    caption_input, True
                )
            else:
                caption_pooler_output = self.encode_text(caption_input)
        else:
            image_pooler_output = self.encode_image(images)
            caption_pooler_output = self.encode_text(caption_input)

        ret.update({"temperature": self.backbone.logit_scale})
        ret.update({"bias": self.backbone.logit_bias})

        strategy = self.config.loss.get("strategy", "auxiliary")

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
        # Mặc định lấy weight từ config (cho static strategies)
        target_circle_weight = self.config.loss.get("circle_loss_weight", 0.0)
        current_circle_weight = target_circle_weight

        # Logic Curriculum Learning [Cite: 10, 45]
        if strategy == "auxiliary_cross_curriculum":
            if current_epoch is None:
                current_epoch = 0 # Fallback an toàn
            
            T_warmup = 5      # Epoch 0-5 (0-indexed)
            T_ramp_len = 15   # 15 Epochs ramp-up
            T_stable = 20     # Epoch > 20
            Lambda_target = 0.1 # Hardcode target theo bài báo hoặc lấy từ config

            if current_epoch <= T_warmup:
                # Giai đoạn 1: Warm-up -> Tắt Circle Loss
                current_circle_weight = 0.0
            elif current_epoch <= T_stable:
                # Giai đoạn 2: Ramp-up -> Tăng tuyến tính
                # Epoch 6 -> (6-5)/15 * 0.1 = 0.006...
                # Epoch 20 -> (20-5)/15 * 0.1 = 0.1
                progress = (current_epoch - T_warmup) / T_ramp_len
                current_circle_weight = progress * Lambda_target
            else:
                # Giai đoạn 3: Stable -> Giữ nguyên 0.1
                current_circle_weight = Lambda_target
            
            # Lưu lại weight thực tế để log
            ret.update({"circle_loss_weight": torch.tensor(current_circle_weight, device=image_pooler_output.device)})

        # --- B. In-modal Circle Loss (Legacy Strategy 2) ---
        if strategy == "auxiliary" and self.config.loss.get("CIR", None):
            circle_m = self.config.loss.get("circle_margin", 0.25)
            circle_gamma = self.config.loss.get("circle_gamma", 64)
            
            img_circle_loss = compute_cir(
                features=image_pooler_output,
                labels=batch["pids"],
                m=circle_m,
                gamma=circle_gamma
            )
            
            txt_circle_loss = compute_cir(
                features=caption_pooler_output,
                labels=batch["pids"],
                m=circle_m,
                gamma=circle_gamma
            )
            
            loss = (img_circle_loss + txt_circle_loss) / 2
            ret.update({"circle_loss": loss * self.config.loss.circle_loss_weight})

        # --- C. Standard N-ITC (Legacy Strategies) ---
        if strategy in ["baseline", "auxiliary"] and self.config.loss.get("NITC", None):
            sim_targets = self.prepare_sim_targets(batch["pids"], self.use_sigmoid)
            image_pooler_output_stopped = (
                image_pooler_output.clone().detach() if alpha != 0 else None
            )
            caption_pooler_output_stopped = (
                caption_pooler_output.clone().detach() if alpha != 0 else None
            )
            
            nitc_loss = objectives.compute_constrative(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                image_features_stopped=image_pooler_output_stopped,
                text_features_stopped=caption_pooler_output_stopped,
                sim_targets=sim_targets,
                alpha=alpha,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                use_sigmoid=self.use_sigmoid,
                weights=weights,
            )
            
            if self.config.loss.get("MVS", None):
                aug_images = batch["aug_images"]
                aug_images_features = self.encode_image(aug_images)
                aug_images_features_stopped = (
                    aug_images_features.clone().detach() if alpha != 0 else None
                )
                augmented_nitc_loss = objectives.compute_constrative(
                    image_features=image_pooler_output,
                    text_features=caption_pooler_output,
                    image_features_stopped=aug_images_features_stopped,
                    text_features_stopped=caption_pooler_output_stopped,
                    sim_targets=sim_targets,
                    alpha=alpha,
                    logit_scale=logit_scale,
                    use_sigmoid=self.use_sigmoid,
                    logit_bias=self.backbone.logit_bias,
                    weights=weights,
                )
                nitc_loss = (nitc_loss + augmented_nitc_loss) / 2

            ret.update({"nitc_loss": nitc_loss * self.config.loss.nitc_loss_weight})

        # --- D. Intrinsic N-ITC (Legacy Strategy 3) ---
        if strategy == "intrinsic":
            circle_m = self.config.loss.get("circle_margin", 0.35)
            circle_gamma = self.config.loss.get("circle_gamma", 80)
            
            intrinsic_loss = objectives.compute_intrinsic_nitc(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                logit_scale=logit_scale, 
                logit_bias=self.backbone.logit_bias,
                pids=batch["pids"],
                m=circle_m,
                gamma=circle_gamma
            )
            
            if self.config.loss.get("MVS", None):
                aug_images = batch["aug_images"]
                aug_images_features = self.encode_image(aug_images)
                
                aug_intrinsic_loss = objectives.compute_intrinsic_nitc(
                    image_features=aug_images_features,
                    text_features=caption_pooler_output,
                    logit_scale=logit_scale,
                    logit_bias=self.backbone.logit_bias,
                    pids=batch["pids"],
                    m=circle_m,
                    gamma=circle_gamma
                )
                intrinsic_loss = (intrinsic_loss + aug_intrinsic_loss) / 2
            
            ret.update({"nitc_loss": intrinsic_loss * self.config.loss.nitc_loss_weight})

        # --- E. Pure Circle Loss (Legacy Strategy 4) ---
        if strategy == "circle_only":
            circle_m = self.config.loss.get("circle_margin", 0.25)
            circle_gamma = self.config.loss.get("circle_gamma", 128)
            
            pure_circle_loss = objectives.compute_cross_modal_circle(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                pids=batch["pids"],
                m=circle_m,
                gamma=circle_gamma
            )

            final_loss = pure_circle_loss

            if self.config.loss.get("MVS", None):
                aug_images = batch["aug_images"]
                aug_images_features = self.encode_image(aug_images)
                
                aug_circle_loss = objectives.compute_cross_modal_circle(
                    image_features=aug_images_features,
                    text_features=caption_pooler_output,
                    pids=batch["pids"],
                    m=circle_m,
                    gamma=circle_gamma
                )
                final_loss = (pure_circle_loss + aug_circle_loss) / 2

            ret.update({"nitc_loss": final_loss * self.config.loss.nitc_loss_weight})
        
        # --- E2. AUXILIARY CROSS-MODAL (Combined Static + Curriculum) ---
        # Xử lý cho cả Mode 5 (Static) và Mode 6 (Curriculum)
        if strategy in ["auxiliary_cross", "auxiliary_cross_curriculum"]:
            
            # 1. Luôn tính N-ITC (Loss nền tảng)
            if self.config.loss.get("NITC", None):
                sim_targets = self.prepare_sim_targets(batch["pids"], self.use_sigmoid)
                image_pooler_output_stopped = (
                    image_pooler_output.clone().detach() if alpha != 0 else None
                )
                caption_pooler_output_stopped = (
                    caption_pooler_output.clone().detach() if alpha != 0 else None
                )
                
                nitc_loss = objectives.compute_constrative(
                    image_features=image_pooler_output,
                    text_features=caption_pooler_output,
                    image_features_stopped=image_pooler_output_stopped,
                    text_features_stopped=caption_pooler_output_stopped,
                    sim_targets=sim_targets,
                    alpha=alpha,
                    logit_scale=logit_scale,
                    logit_bias=self.backbone.logit_bias,
                    use_sigmoid=self.use_sigmoid,
                    weights=weights,
                )
                
                if self.config.loss.get("MVS", None):
                    aug_images = batch["aug_images"]
                    aug_images_features = self.encode_image(aug_images)
                    aug_images_features_stopped = (
                        aug_images_features.clone().detach() if alpha != 0 else None
                    )
                    augmented_nitc_loss = objectives.compute_constrative(
                        image_features=image_pooler_output,
                        text_features=caption_pooler_output,
                        image_features_stopped=aug_images_features_stopped,
                        text_features_stopped=caption_pooler_output_stopped,
                        sim_targets=sim_targets,
                        alpha=alpha,
                        logit_scale=logit_scale,
                        use_sigmoid=self.use_sigmoid,
                        logit_bias=self.backbone.logit_bias,
                        weights=weights,
                    )
                    nitc_loss = (nitc_loss + augmented_nitc_loss) / 2

                ret.update({"nitc_loss": nitc_loss * self.config.loss.nitc_loss_weight})

            # 2. Tính Circle Loss (Chỉ tính khi trọng số > 0 để tiết kiệm compute ở Warmup)
            if self.config.loss.get("CIR", None) and current_circle_weight > 0:
                circle_m = self.config.loss.get("circle_margin", 0.25)
                circle_gamma = self.config.loss.get("circle_gamma", 128)
                
                # Cross-Modal Circle Loss gốc
                cm_circle_loss = objectives.compute_cross_modal_circle(
                    image_features=image_pooler_output,
                    text_features=caption_pooler_output,
                    pids=batch["pids"],
                    m=circle_m,
                    gamma=circle_gamma
                )
                
                final_circle_loss = cm_circle_loss

                # MVS cho Circle Loss
                if self.config.loss.get("MVS", None):
                    # Check nếu aug_features chưa được tính ở block N-ITC
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

                # Nhân với current_circle_weight (Đã xử lý logic động ở trên)
                ret.update({"circle_loss": final_circle_loss * current_circle_weight})
            
            # Nếu đang trong giai đoạn warmup (weight=0), có thể log loss = 0 để tiện theo dõi
            elif self.config.loss.get("CIR", None):
                 ret.update({"circle_loss": torch.tensor(0.0, device=image_pooler_output.device, requires_grad=True)})


        # --- F. Other Losses ---
        # C-ITC
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

        # RITC
        if self.config.loss.get("RITC", None):
            sim_targets = self.prepare_sim_targets(
                batch["pids"], use_sigmoid=self.use_sigmoid
            )
            loss = objectives.compute_ritc(
                image_features=image_pooler_output,
                text_features=caption_pooler_output,
                logit_scale=logit_scale,
                logit_bias=self.backbone.logit_bias,
                sim_targets=sim_targets,
                use_sigmoid=self.use_sigmoid,
                eps=self.config.loss.ritc_eps,
            )
            ret.update({"ritc_loss": loss * self.config.loss.ritc_loss_weight})

        # ITC
        if self.config.loss.get("ITC", None):
            ret.update(
                {
                    "itc_loss": (
                        objectives.compute_itc(
                            image_pooler_output, caption_pooler_output, logit_scale
                        )
                        * self.config.loss.itc_loss_weight
                    )
                }
            )

        # SDM
        if self.config.loss.get("SDM", None):
            ret.update(
                {
                    "sdm_loss": (
                        objectives.compute_sdm(
                            image_pooler_output,
                            caption_pooler_output,
                            batch["pids"],
                            logit_scale,
                            self.backbone.logit_bias,
                            weights=weights,
                        )
                        * self.config.loss.sdm_loss_weight
                    )
                }
            )

        # CMPM
        if self.config.loss.get("CMPM", None):
            ret.update(
                {
                    "cmpm_loss": (
                        objectives.compute_cmpm(
                            image_pooler_output, caption_pooler_output, batch["pids"]
                        )
                        * self.config.loss.cmpm_loss_weight
                    )
                }
            )

        # ID Loss
        if self.config.loss.get("ID", None):
            image_logits = self.classifier(image_pooler_output).float()
            text_logits = self.classifier(caption_pooler_output).float()
            ret.update(
                {
                    "id_loss": (
                        objectives.compute_id(
                            image_logits, text_logits, batch["pids"], weights=weights
                        )
                        * self.config.loss.id_loss_weight
                    )
                }
            )
            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            image_precision = (image_pred == batch["pids"]).float().mean()
            text_precision = (text_pred == batch["pids"]).float().mean()
            ret.update({"img_acc": image_precision})
            ret.update({"txt_acc": text_precision})

        # MLM Loss
        if self.config.loss.get("MLM", None):
            mlm_input = {
                "input_ids": batch["mlm_input_ids"],
                "attention_mask": batch["mlm_attention_mask"],
            }
            mlm_labels = batch["mlm_labels"]

            _, mlm_last_hidden = self.encode_text(mlm_input, return_last_hidden=True)

            x = self.cross_former(mlm_last_hidden, image_last_hidden, image_last_hidden)
            x = self.mlm_head(x)

            scores = x.reshape(-1, self.vocab_size)
            mlm_labels = mlm_labels.reshape(-1)
            ret.update(
                {
                    "mlm_loss": objectives.compute_mlm(
                        scores, mlm_labels, self.pad_token_id
                    )
                    * self.config.loss.mlm_loss_weight
                }
            )

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels != self.pad_token_id)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({"mlm_acc": acc})

        return ret

    def forward2(self, batch):
        caption_input = {
            "input_ids": batch["caption_input_ids"],
            "attention_mask": batch["caption_attention_mask"],
        }

        images = batch["images"]

        with torch.no_grad():
            image_feats = self.encode_image(images)
            text_feats = self.encode_text(caption_input)

        return image_feats, text_feats