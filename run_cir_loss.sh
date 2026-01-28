#!/bin/bash

echo "========================================================"
echo "TRAINING STRATEGY"
echo "========================================================"
echo "1. Baseline "
echo "2. In-modal Auxiliary"
echo "3. Intrinsic "
echo "4. Pure Circle Loss"
echo "5. Auxiliary Cross-modal (Static Weight)"
echo "6. Auxiliary Cross-modal (Curriculum Learning)"
echo "========================================================"
read -p "Insert your choice (1-6): " choice

COMMON_ARGS="
img_size_str='(256,256)'
dataset=vn3k_vi
dataset.sampler=identity
dataset.num_instance=4
dataset.batch_size=24

trainer.max_epochs=60
trainer.accumulate_grad_batches=3
++trainer.precision=16-mixed

optimizer=cir_test
optimizer.param_groups.default.lr=1e-4
+optimizer._target_=torch.optim.AdamW

backbone.freeze.vision=true
backbone.freeze.text=true


loss.softlabel_ratio=0.0

+lora._target_=peft.LoraConfig
+lora.r=32
+lora.lora_alpha=64
+lora.lora_dropout=0.05
+lora.bias=none
+lora.inference_mode=false
+lora.task_type=FEATURE_EXTRACTION
+lora.target_modules=[\"q_proj\",\"v_proj\",\"k_proj\",\"out_proj\"]
"

case $choice in

1)
    echo ">>> RUNNING MODE 1: BASELINE..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=baseline \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=false \
        loss.circle_loss_weight=0.0 \
        loss.CITC=true
    ;;

2)
    echo ">>> RUNNING MODE 2: AUXILIARY (Kết hợp)..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=auxiliary \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=true \
        loss.circle_loss_weight=0.3 \
        loss.CITC=true
    ;;

3)
    echo ">>> RUNNING MODE 3: INTRINSIC "
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=intrinsic \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=false \
        loss.circle_loss_weight=0.0 \
        loss.CITC=true 
    ;;

4)
    echo ">>> RUNNING MODE 4: PURE CIRCLE "
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=circle_only \
        loss.NITC=false \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=false \
        loss.circle_loss_weight=0.0 \
        loss.CITC=true 
    ;;

5)
    echo ">>> RUNNING MODE 5: AUXILIARY CROSS-MODAL (STATIC)..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=auxiliary_cross \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=true \
        loss.circle_loss_weight=0.1 \
        loss.CITC=true
    ;;

6)
    echo ">>> RUNNING MODE 6: CURRICULUM LEARNING (Cross-modal)..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=auxiliary_cross_curriculum \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=true \
        loss.circle_loss_weight=0.1 \
        loss.CITC=true
    ;;

*)
    echo "Lựa chọn không hợp lệ!"
    exit 1
    ;;
esac