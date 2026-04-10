#!/bin/bash

uv run trainer.py -cn cir_msiglip \
    trainer.max_epochs=60 \
    trainer.accumulate_grad_batches=4 \
    ++trainer.precision=16-mixed \
    \
    dataset.batch_size=24 \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-5 \
    optimizer.param_groups.backbone.lr=1e-5
