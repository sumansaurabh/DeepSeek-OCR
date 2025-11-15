# Attention Diagnostics for DeepSeek-OCR

## Overview

This document describes the attention diagnostics feature added to DeepSeek-OCR to address **GitHub Issue #3243**. This feature provides transparency about which attention implementation is being used during inference (FlashAttention vs SDPA).

## Problem Statement

**Original Issue #3243**: When attention operations get fused in a single op (e.g., with Myelin in TensorRT-LLM), it's not clear from execution traces whether FlashAttention or proper fusion is being used, and whether quantization is applied under the hood.

**Note**: While the original issue references NVIDIA TensorRT-LLM's Myelin compiler and trex-tooltip, this implementation provides similar diagnostic capabilities for the DeepSeek-OCR project which uses vLLM and PyTorch.

## Solution

We've implemented a comprehensive attention diagnostics system that:

1. **Detects available attention backends** (FlashAttention, PyTorch SDPA)
2. **Logs which implementation is used** for each attention layer
3. **Tracks attention call statistics** across different components
4. **Provides detailed summary reports** after inference

## Architecture

### Components

1. **`attention_diagnostics.py`**: Core diagnostics module
   - `AttentionDiagnostics` class: Singleton that tracks attention usage
   - Backend detection (FlashAttention, SDPA, CUDA)
   - Statistics tracking and reporting

2. **Modified Attention Layers**:
   - `deepencoder/clip_sdpa.py`: CLIP vision encoder attention
   - `deepencoder/sam_vary_sdpa.py`: SAM vision encoder attention

3. **Configuration**:
   - `config.py`: Added diagnostic flags
   - `ENABLE_ATTENTION_DIAGNOSTICS`: Enable/disable logging
   - `PRINT_ATTENTION_SUMMARY`: Print summary after inference

4. **Integration**:
   - `run_dpsk_ocr_image.py`: Updated to use diagnostics
   - Other inference scripts can be similarly updated

## Usage

### Basic Usage

1. **Enable diagnostics in `config.py`**:
```python
ENABLE_ATTENTION_DIAGNOSTICS = True
PRINT_ATTENTION_SUMMARY = True
```

2. **Run inference as normal**:
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

3. **View diagnostic output**:
The system will automatically print:
- Available attention backends at startup
- Attention implementation details during inference
- Summary report after completion

### Example Output

```
======================================================================
ATTENTION DIAGNOSTICS ENABLED (GitHub Issue #3243)
======================================================================
[INFO] AttentionDiagnostics: ✓ FlashAttention available (version: 2.7.3)
[INFO] AttentionDiagnostics: ✓ PyTorch SDPA (Scaled Dot Product Attention) available
[INFO] AttentionDiagnostics: ✓ CUDA available (device: NVIDIA A100-SXM4-40GB)
[INFO] AttentionDiagnostics:   CUDA version: 11.8
[INFO] AttentionDiagnostics:   cuDNN version: 8902

... (inference output) ...

======================================================================
ATTENTION IMPLEMENTATION SUMMARY
======================================================================

📊 Backend Availability:
  FlashAttention: ✓ Available
    Version: 2.7.3
    Calls: 48
  PyTorch SDPA: ✓ Available
    Calls: 24
  CUDA: ✓ Available
    Device: NVIDIA A100-SXM4-40GB

🔍 Attention Calls by Component:
  CLIP Encoder: 48
  SAM Encoder: 24
  Total: 72

💡 Recommendations:
  ✓ Attention implementation is optimized!
======================================================================
```

### Programmatic Usage

You can also use the diagnostics API programmatically:

```python
from attention_diagnostics import get_diagnostics, print_attention_summary

# Get diagnostics instance
diag = get_diagnostics()

# Check what's available
summary = diag.get_summary()
print(f"FlashAttention available: {summary['backends']['flash_attention']['available']}")
print(f"FlashAttention version: {summary['backends']['flash_attention']['version']}")

# Print full summary
print_attention_summary()

# Reset statistics
diag.reset_stats()
```

## Understanding the Output

### Backend Availability

- **FlashAttention**: Optimized attention kernel for NVIDIA GPUs
  - ✓ Available: FlashAttention is installed and can be used
  - ✗ Not Available: FlashAttention is not installed
  - Version: Shows the installed FlashAttention version
  - Calls: Number of times FlashAttention was used

- **PyTorch SDPA**: PyTorch's built-in Scaled Dot Product Attention
  - ✓ Available: PyTorch >= 2.0 with SDPA support
  - ✗ Not Available: Older PyTorch version
  - Calls: Number of times SDPA was used

- **CUDA**: NVIDIA GPU support
  - ✓ Available: CUDA-capable GPU detected
  - Device: GPU model name
  - CUDA version: CUDA toolkit version
  - cuDNN version: cuDNN library version

### Attention Calls by Component

- **CLIP Encoder**: Vision encoder using CLIP architecture
  - Uses FlashAttention when `use_flash_attn=True` in config
  - Falls back to SDPA when FlashAttention unavailable

- **SAM Encoder**: Vision encoder using SAM (Segment Anything Model) architecture
  - Currently uses SDPA (not FlashAttention)
  - Supports relative positional embeddings

### Recommendations

The system provides actionable recommendations:

1. **FlashAttention not installed**:
   ```
   ⚠ Install FlashAttention for optimal performance:
     pip install flash-attn==2.7.3 --no-build-isolation
   ```

2. **FlashAttention available but not used**:
   ```
   ⚠ FlashAttention is available but not being used.
     Check that use_flash_attn=True in model config.
   ```

3. **Optimized configuration**:
   ```
   ✓ Attention implementation is optimized!
   ```

## Configuration Options

### In `config.py`

```python
# Enable/disable attention diagnostics
ENABLE_ATTENTION_DIAGNOSTICS = True  # Set to False to disable

# Print summary after inference
PRINT_ATTENTION_SUMMARY = True  # Set to False to skip summary
```

### In Model Config

The CLIP encoder attention implementation is controlled by:

```python
vit_model_cfg = adict(
    # ... other config ...
    use_flash_attn=True,  # Use FlashAttention if available
    # ... other config ...
)
```

## Performance Implications

### FlashAttention vs SDPA

| Implementation | Speed | Memory | Notes |
|---------------|-------|--------|-------|
| FlashAttention | ⚡⚡⚡ Fastest | 💾 Most efficient | Requires CUDA, optimal for long sequences |
| PyTorch SDPA | ⚡⚡ Fast | 💾 Efficient | Built-in, good fallback |
| Naive Attention | ⚡ Slow | 💾💾 High memory | Not used in this project |

### When to Use Each

- **FlashAttention**: 
  - Best for production deployments
  - Optimal for batch processing
  - Requires CUDA-capable GPU

- **SDPA**:
  - Good fallback when FlashAttention unavailable
  - Works on CPU (slower)
  - Easier to debug

## Troubleshooting

### FlashAttention Not Detected

If FlashAttention is installed but not detected:

1. Check installation:
   ```bash
   pip list | grep flash-attn
   ```

2. Verify CUDA compatibility:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

3. Reinstall if needed:
   ```bash
   pip uninstall flash-attn
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

### Diagnostics Not Working

If diagnostics don't appear:

1. Check `config.py`:
   ```python
   ENABLE_ATTENTION_DIAGNOSTICS = True
   ```

2. Verify module import:
   ```python
   from attention_diagnostics import get_diagnostics
   diag = get_diagnostics()
   diag.print_summary()
   ```

3. Check for import errors in console output

### Performance Issues

If inference is slower than expected:

1. Check if FlashAttention is being used:
   - Look for "FlashAttention: ✓ Available" in output
   - Check "Calls" count is > 0

2. Verify GPU utilization:
   ```bash
   nvidia-smi
   ```

3. Check CUDA version compatibility:
   - FlashAttention requires CUDA 11.6+
   - Optimal with CUDA 11.8 or 12.x

## Technical Details

### Attention Implementation Detection

The diagnostics system detects attention implementations at runtime:

1. **FlashAttention**: Checks for `flash_attn` module import
2. **SDPA**: Checks for `torch.nn.functional.scaled_dot_product_attention`
3. **CUDA**: Checks `torch.cuda.is_available()`

### Logging Strategy

- **First forward pass**: Logs attention configuration
- **Subsequent passes**: Silent (no repeated logging)
- **Summary**: Aggregated statistics at end

### Thread Safety

The `AttentionDiagnostics` class is implemented as a singleton with thread-safe initialization, suitable for multi-threaded inference scenarios.

## Extending the Diagnostics

### Adding New Attention Types

To add diagnostics for additional attention layers:

1. Import the diagnostics module:
   ```python
   from attention_diagnostics import log_attention_info
   ```

2. Add logging in the attention forward method:
   ```python
   def forward(self, x):
       if self._first_forward:
           log_attention_info(
               attention_type='your_attention_type',
               use_flash=self.use_flash_attention,
               batch_size=x.shape[0],
               seq_len=x.shape[1],
               num_heads=self.num_heads
           )
           self._first_forward = False
       # ... rest of forward pass
   ```

### Custom Metrics

To track custom metrics:

```python
from attention_diagnostics import get_diagnostics

diag = get_diagnostics()
# Add custom tracking logic
diag.attention_stats['custom_metric'] = value
```

## Related Issues

- **GitHub Issue #3243**: Original issue requesting attention fusion transparency
- Related to TensorRT-LLM Myelin compiler (different project)
- This implementation provides similar diagnostics for vLLM/PyTorch

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Repository](https://github.com/deepseek-ai/DeepSeek-OCR)

## Contributing

To improve the diagnostics system:

1. Add more detailed metrics
2. Support additional attention implementations
3. Improve visualization of statistics
4. Add export functionality (JSON, CSV)

## License

This diagnostic feature follows the same license as the DeepSeek-OCR project.
