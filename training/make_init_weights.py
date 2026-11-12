#!/usr/bin/env python3
"""Convert a published fp16 safetensors checkpoint into fp32 initial weights for fine-tuning.

Run from the root of NVIDIA's Ising-Decoding checkout so that its loader is importable:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=code python make_init_weights.py \
        ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors nvidia_accurate_init_fp32.pt

The output is a bare state dict. finetune.sh copies it into the run's models folder as
PreDecoderModelMemory_v1.0.0.pt; the training code's resume logic then loads the weights, finds no
optimizer checkpoint, and starts from epoch 0 with a fresh optimizer.

For the reported runs this conversion was done interactively; this script is the equivalent.
"""
import sys

import torch
from export.safetensors_utils import load_safetensors  # from NVIDIA's Ising-Decoding

src, dst = sys.argv[1], sys.argv[2]
model, _metadata = load_safetensors(src, device="cpu")
state = {k: v.float() for k, v in model.state_dict().items()}
torch.save(state, dst)
print(f"saved {sum(v.numel() for v in state.values()):,} parameters as fp32 to {dst}")
