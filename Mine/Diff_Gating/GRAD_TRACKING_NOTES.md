# Gate Gradient Tracking Implementation

## Overview
Added comprehensive gradient tracking for the gate parameter `g` in `MetaNet_Gated` to verify whether gradients are flowing and the gate is being updated during training.

## Changes Made

### 1. Enhanced Gate Statistics in `MetaNet_Gated.forward()`
Added tracking of additional gate statistics:
- `last_g_std`: Standard deviation of gate values
- `last_g_min`: Minimum gate value
- `last_g_max`: Maximum gate value

These help understand the gate behavior across the batch.

### 2. Gradient Tracking in `train_one_epoch()`
Added three new tracking variables:
- `gate_grad_sum`: Accumulates gate weight gradient norms
- `gate_grad_norm_sum`: Accumulates squared gradient norms (for std calculation)
- `gate_grad_steps`: Counter for gradient updates

### 3. Per-Batch Gradient Extraction
After each backward pass, the code checks:
```python
if model.prompt_learner.meta_net.gate.weight.grad is not None:
    gate_grad_norm = model.prompt_learner.meta_net.gate.weight.grad.norm().item()
    gate_grad_sum += gate_grad_norm
    gate_grad_norm_sum += gate_grad_norm ** 2
    gate_grad_steps += 1
```

This captures the norm of gradients flowing to the gate weight parameter.

### 4. Real-Time Monitoring
Progress bar now shows `g_grad` alongside loss and gate mean:
```
loss=0.1234 gate=0.5432 g_grad=0.001234
```

### 5. Epoch-Level Statistics
At end of each epoch, returns:
- `epoch_gate_grad_mean`: Average gradient norm for the epoch
- `epoch_gate_grad_std`: Standard deviation of gradient norms

### 6. Console Output
Each epoch prints gate gradient stats:
```
Gate grad  : 0.001234 (std: 0.000456)
```

### 7. Weights & Biases Logging
Logs to wandb:
- `train/gate_grad_mean`: Mean gradient norm
- `train/gate_grad_std`: Std of gradient norms

## Interpretation

### What to Look For

**Gate is being updated properly if:**
- `g_grad` values are consistently non-zero (e.g., > 1e-6)
- `epoch_gate_grad_mean` increases or remains steady across epochs
- Gradients are not NaN or Inf

**Gate is NOT being updated if:**
- `g_grad` is always 0 or NaN
- Gradients are extremely small (< 1e-10)
- Values are stuck in one state despite training

### Expected Ranges
For ViT-B/16 with our architecture:
- Reasonable: `gate_grad_norm` in range `[1e-5, 1e-2]`
- Suspicious: `gate_grad_norm < 1e-8` (might indicate blocked gradients)
- Suspicious: `gate_grad_norm > 1e-1` (might indicate exploding gradients)

## Debugging Steps

1. **Check if gradients exist**: Look for "gate_grad_norm is None"
2. **Check gradient magnitude**: If too small, check learning rate or gradient scaling
3. **Check gate statistics**: If `last_g` is always 0 or 1, gate might not be learning
4. **Check for NaN**: If gradients become NaN, check for numerical instability in loss

## Files Modified
- `/home/Stark/CoOp/Mine/Diff_Gating/cocoop_kan.py`
