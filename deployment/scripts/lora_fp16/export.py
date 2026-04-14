"""
Step 2: Export checkpoint to inference-ready formats.
Merges LoRA into base model, strips optimizer states, converts to FP16 and optionally ONNX.

Usage:
    python deployment/scripts/lora_fp16/export.py \
        --ckpt epoch=53-val_score=51.30.ckpt \
        --output-dir exported_model \
        --format pytorch           # or: onnx, both

Outputs:
    exported_model/
    ├── model_fp16.pt              # PyTorch FP16 state dict (vision + text + logit params)
    ├── model_fp32.pt              # PyTorch FP32 state dict (fallback)
    ├── config.yaml                # Resolved Hydra config from checkpoint
    ├── vision_encoder.onnx        # (if --format onnx/both)
    └── text_encoder.onnx          # (if --format onnx/both)
"""

import argparse
import os
import sys

import torch
import yaml

# Add deployment root to path (lora_fp16/ → scripts/ → deployment/)
_deployment_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _deployment_root)
from utils import TeeLogger

# Add project root to path (deployment/ → project root)
sys.path.insert(0, os.path.dirname(_deployment_root))

from omegaconf import OmegaConf, DictConfig


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_model_from_checkpoint(ckpt_path: str):
    """Load Lightning checkpoint and reconstruct model with merged LoRA."""
    from lightning_models import LitTBPS
    from lightning_data import TBPSDataModule

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt["hyper_parameters"]["config"]
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Build data module just to get train_loader length (needed for LitTBPS init)
    # We use a dummy value since we won't train
    num_iters = 100  # dummy, not used for inference

    # Build model
    lit_model = LitTBPS(config, num_iters_per_epoch=num_iters)

    # Setup LoRA if configured
    if config.get("lora", None):
        lit_model.setup_lora(config.lora)

    # Load state dict
    lit_model.load_state_dict(ckpt["state_dict"], strict=False)
    lit_model.eval()

    # Merge LoRA weights into base model
    try:
        from peft import PeftModel
        if isinstance(lit_model.backbone, PeftModel):
            print("Merging LoRA adapters into base model...")
            lit_model.backbone = lit_model.backbone.merge_and_unload()
            print("LoRA merged successfully.")
    except Exception as e:
        print(f"Warning: Could not merge LoRA: {e}")
        print("Proceeding with unmerged model (LoRA adapters still separate).")

    return lit_model, config


def export_pytorch(lit_model, config, output_dir: str):
    """Export as PyTorch state dicts (FP32 and FP16)."""
    model = lit_model.model

    # Collect inference-relevant state dict
    inference_state = {}
    for k, v in model.state_dict().items():
        inference_state[k] = v

    # FP32
    fp32_path = os.path.join(output_dir, "model_fp32.pt")
    torch.save(inference_state, fp32_path)
    fp32_size = os.path.getsize(fp32_path) / 1024**2
    print(f"Saved FP32: {fp32_path} ({fp32_size:.1f} MB)")

    # FP16 — preserve tensor sharing so torch.save deduplicates properly
    seen = {}  # data_ptr -> converted tensor
    fp16_state = {}
    for k, v in inference_state.items():
        ptr = v.data_ptr()
        if ptr in seen:
            fp16_state[k] = seen[ptr]
        else:
            converted = v.half() if v.is_floating_point() else v
            seen[ptr] = converted
            fp16_state[k] = converted
    fp16_path = os.path.join(output_dir, "model_fp16.pt")
    torch.save(fp16_state, fp16_path)
    fp16_size = os.path.getsize(fp16_path) / 1024**2
    print(f"Saved FP16: {fp16_path} ({fp16_size:.1f} MB)")

    # Save config — use a dumper that represents tuples as plain YAML lists
    config_path = os.path.join(output_dir, "config.yaml")

    class SafeDumper(yaml.SafeDumper):
        pass

    SafeDumper.add_representer(
        tuple,
        lambda dumper, data: dumper.represent_list(data),
    )

    with open(config_path, "w") as f:
        config_dict = OmegaConf.to_container(config, resolve=True)
        yaml.dump(config_dict, f, Dumper=SafeDumper, default_flow_style=False)
    print(f"Saved config: {config_path}")


def export_onnx(lit_model, config, output_dir: str):
    """Export vision and text encoders as separate ONNX models."""
    model = lit_model.model
    model.eval()

    image_size = config.backbone.vision_config.image_size
    if isinstance(image_size, (list, tuple)):
        h, w = image_size
    else:
        h = w = image_size

    max_text_len = config.tokenizer.model_max_length

    # --- Vision encoder ---
    print("\nExporting vision encoder to ONNX...")
    dummy_image = torch.randn(1, 3, h, w)

    class VisionWrapper(torch.nn.Module):
        def __init__(self, tbps_model):
            super().__init__()
            self.model = tbps_model

        def forward(self, image):
            return self.model.encode_image(image)

    vision_path = os.path.join(output_dir, "vision_encoder.onnx")
    torch.onnx.export(
        VisionWrapper(model),
        dummy_image,
        vision_path,
        input_names=["image"],
        output_names=["image_embedding"],
        dynamic_axes={"image": {0: "batch_size"}, "image_embedding": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Saved: {vision_path} ({os.path.getsize(vision_path)/1024**2:.1f} MB)")

    # --- Text encoder ---
    print("Exporting text encoder to ONNX...")
    dummy_input_ids = torch.zeros(1, max_text_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_text_len, dtype=torch.long)

    class TextWrapper(torch.nn.Module):
        def __init__(self, tbps_model):
            super().__init__()
            self.model = tbps_model

        def forward(self, input_ids, attention_mask):
            caption_input = {"input_ids": input_ids, "attention_mask": attention_mask}
            return self.model.encode_text(caption_input)

    text_path = os.path.join(output_dir, "text_encoder.onnx")
    torch.onnx.export(
        TextWrapper(model),
        (dummy_input_ids, dummy_attention_mask),
        text_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embedding": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Saved: {text_path} ({os.path.getsize(text_path)/1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="exported_model")
    parser.add_argument("--format", choices=["pytorch", "onnx", "both"], default="pytorch")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lit_model, config = load_model_from_checkpoint(args.ckpt)

    if args.format in ("pytorch", "both"):
        export_pytorch(lit_model, config, args.output_dir)

    if args.format in ("onnx", "both"):
        export_onnx(lit_model, config, args.output_dir)

    print(f"\nDone! All files saved to: {args.output_dir}/")
    print("\nNext: transfer exported_model/ to RB3 and run deployment/scripts/inference_test.py")


if __name__ == "__main__":
    log_dir = os.path.join(_deployment_root, "logs")
    logger = TeeLogger(log_dir, "export_lora_fp16")
    main()
    logger.close()
