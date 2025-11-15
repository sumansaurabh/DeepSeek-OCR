# Quick Start Guide: Attention Diagnostics

## What This Solves

**GitHub Issue #3243**: Provides visibility into attention fusion, FlashAttention usage, and quantization details that were previously hidden.

## 5-Minute Setup

### Step 1: Enable Diagnostics

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Change these lines:
ENABLE_ATTENTION_DIAGNOSTICS = True   # Was: False
ATTENTION_DIAGNOSTICS_VERBOSE = True  # Was: False
ATTENTION_DIAGNOSTICS_OUTPUT = './attention_report'  # Was: ''
```

### Step 2: Run Your Model

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Option A: Use the example script
python run_with_diagnostics.py

# Option B: Use your existing script (diagnostics auto-enabled)
python run_dpsk_ocr_image.py
```

### Step 3: View Results

The diagnostics will be printed to console and saved to files:
- `attention_report.json` - Machine-readable format
- `attention_report.txt` - Human-readable format

## What You'll See

```
============================================================
ATTENTION DIAGNOSTICS SUMMARY
============================================================
Total Attention Layers: 24
FlashAttention Enabled Layers: 24

Backend Distribution:
  FlashAttention-2: 24

Fusion Type Distribution:
  QKV-Fused: 24

Quantization Distribution:
  BF16: 24
============================================================
```

## Key Information Provided

✓ **FlashAttention Status**: Confirms if FlashAttention is actually being used
✓ **Fusion Type**: Shows QKV-Fused, Full-Fused, Myelin-Optimized, or None
✓ **Quantization**: Displays INT8, FP8, FP16, BF16, Implicit, or None
✓ **Per-Layer Details**: Batch size, sequence length, device, dtype

## Common Use Cases

### 1. Verify FlashAttention is Active

```python
from attention_diagnostics import get_collector

collector = get_collector()
summary = collector.get_summary()

if summary['flash_attention_layers'] > 0:
    print("✓ FlashAttention is working!")
else:
    print("✗ FlashAttention is NOT active")
```

### 2. Check Quantization

```python
summary = collector.get_summary()
quant_types = summary['quantization_distribution']

if 'INT8' in quant_types or 'FP8' in quant_types:
    print("✓ Quantization is active")
```

### 3. Export for Analysis

```python
collector.export_to_file('my_report.json', format='json')
```

## Troubleshooting

### Diagnostics Not Showing?

1. Check `ENABLE_ATTENTION_DIAGNOSTICS = True` in config.py
2. Ensure model is actually running (not just loading)
3. Enable verbose mode: `ATTENTION_DIAGNOSTICS_VERBOSE = True`

### Want Less Output?

```python
# In config.py
ATTENTION_DIAGNOSTICS_VERBOSE = False  # Disable detailed logging
```

### Need Programmatic Access?

```python
from attention_diagnostics import enable_diagnostics, get_collector

enable_diagnostics(verbose=False)
# ... run model ...
collector = get_collector()

# Access data
for diag in collector.diagnostics:
    print(f"Layer {diag.layer_id}: {diag.backend.value}")
```

## Performance Impact

- **Disabled** (default): 0% overhead
- **Enabled**: < 1% overhead
- **Collection**: Only on first forward pass per layer

## Next Steps

- Read full documentation: `ATTENTION_DIAGNOSTICS_README.md`
- View example script: `run_with_diagnostics.py`
- Check implementation: `attention_diagnostics.py`

## Questions?

Refer to the main documentation or check the inline code comments in `attention_diagnostics.py`.
