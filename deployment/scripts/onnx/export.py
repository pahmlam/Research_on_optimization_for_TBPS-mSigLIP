"""
Step 2: Convert exported PyTorch model to ONNX.
Loads FP32/FP16 state dict + config from a model directory (output of lora_fp16/export.py),
rebuilds the TBPS model, and exports vision & text encoders as separate ONNX graphs.

Usage:
    python deployment/scripts/onnx/export.py \
        --model-dir exported_model \
        --precision fp32              # or: fp16 (default: fp32, use fp32 for ONNX stability)

Inputs (from lora_fp16/export.py):
    exported_model/
    ├── model_fp32.pt
    ├── model_fp16.pt
    └── config.yaml

Outputs:
    exported_model/
    ├── vision_encoder.onnx
    └── text_encoder.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml

# Add deployment root to path (onnx/ → scripts/ → deployment/)
_deployment_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _deployment_root)
from deploy_utils import TeeLogger

# Add project root to path (deployment/ → project root)
sys.path.insert(0, os.path.dirname(_deployment_root))

from omegaconf import OmegaConf


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_model_from_exported(model_dir: str, precision: str = "fp32"):
    """Rebuild TBPS model from exported state dict + config."""
    from model.build import build_backbone_with_proper_layer_resize
    from model.tbps import TBPS

    config_path = os.path.join(model_dir, "config.yaml")
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = OmegaConf.create(config_dict)

    # Choose state dict file
    if precision == "fp16":
        pt_path = os.path.join(model_dir, "model_fp16.pt")
    else:
        pt_path = os.path.join(model_dir, "model_fp32.pt")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"State dict not found: {pt_path}")

    print(f"Loading state dict: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Rebuild model architecture
    print("Building backbone from config...")
    backbone = build_backbone_with_proper_layer_resize(config.backbone)
    model = TBPS(config=config, backbone=backbone)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    pt_size = os.path.getsize(pt_path) / 1024**2
    print(f"Model loaded ({pt_size:.1f} MB, {precision})")
    return model, config


class VisionWrapper(nn.Module):
    """Wraps TBPS.encode_image for ONNX export."""

    def __init__(self, tbps_model):
        super().__init__()
        self.model = tbps_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)


class TextWrapper(nn.Module):
    """Wraps TBPS.encode_text for ONNX export."""

    def __init__(self, tbps_model):
        super().__init__()
        self.model = tbps_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        caption_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.model.encode_text(caption_input)


def export_onnx(model, config, output_dir: str):
    """Export vision and text encoders as separate ONNX models."""
    image_size = config.backbone.vision_config.image_size
    if isinstance(image_size, (list, tuple)):
        h, w = image_size
    else:
        h = w = image_size

    max_text_len = config.tokenizer.model_max_length

    # --- Vision encoder ---
    print("\nExporting vision encoder to ONNX...")
    dummy_image = torch.randn(1, 3, h, w)

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
    print(f"Saved: {vision_path} ({os.path.getsize(vision_path) / 1024**2:.1f} MB)")

    # --- Text encoder ---
    print("\nExporting text encoder to ONNX...")
    dummy_input_ids = torch.zeros(1, max_text_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_text_len, dtype=torch.long)

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
    print(f"Saved: {text_path} ({os.path.getsize(text_path) / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory from lora_fp16/export.py")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                        help="Which state dict to load (default: fp32 for ONNX stability)")
    args = parser.parse_args()

    model, config = load_model_from_exported(args.model_dir, args.precision)
    export_onnx(model, config, args.model_dir)

    print(f"\nDone! ONNX models saved to: {args.model_dir}/")
    print("Next: run deployment/scripts/inference_test.py or convert to SNPE")


if __name__ == "__main__":
    log_dir = os.path.join(_deployment_root, "logs")
    logger = TeeLogger(log_dir, "export_onnx")
    main()
    logger.close()
