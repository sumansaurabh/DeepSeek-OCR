# Fix for GitHub Issue #7: LlamaFlashAttention2 ImportError

## Problem Description

When using DeepSeek-OCR with transformers version 4.52.0 or newer (including 4.57.1), you may encounter this error:

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

## Root Cause

The `LlamaFlashAttention2` class was removed from the transformers library starting with version 4.52.0. However, DeepSeek-OCR's model code (which is loaded from Hugging Face Hub using `trust_remote_code=True`) still imports this class, causing the ImportError.

## Solution

### Quick Fix

Install the compatible transformers version:

```bash
pip uninstall transformers -y
pip install transformers==4.46.3
```

### Complete Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd DeepSeek-OCR
   ```

2. **Install all requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify compatibility**:
   ```bash
   python check_compatibility.py
   ```

4. **Run the model safely**:
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-hf
   python run_dpsk_ocr_safe.py
   ```

## For Google Colab Users

If you're running DeepSeek-OCR in Google Colab, follow these steps:

```python
# Uninstall the pre-installed transformers version
!pip uninstall transformers -y

# Install the compatible version
!pip install transformers==4.46.3

# Restart the runtime (important!)
# Click: Runtime -> Restart runtime

# After restart, verify the version
import transformers
print(f"transformers version: {transformers.__version__}")

# Now you can load the model
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
```

## Version Compatibility

| transformers Version | Status | Notes |
|---------------------|--------|-------|
| 4.46.0 - 4.51.9 | ✅ Compatible | Recommended: 4.46.3 |
| 4.52.0+ | ❌ Incompatible | LlamaFlashAttention2 removed |
| 4.57.1 | ❌ Incompatible | Reported in Issue #7 |

## Tools Provided

This repository now includes helpful tools to diagnose and prevent this issue:

### 1. Compatibility Checker (`check_compatibility.py`)

Validates your environment before running DeepSeek-OCR:

```bash
python check_compatibility.py
```

**Output example:**
```
======================================================================
DeepSeek-OCR Compatibility Checker
======================================================================

1. Checking Python version...
   Installed: Python 3.9.24
   ✓ OK

2. Checking transformers version...
   Installed: transformers 4.57.1
   ❌ ERROR: Version too new (maximum: 4.51.9)
   Recommended: transformers==4.46.3
   ...
```

### 2. Safe Runner (`run_dpsk_ocr_safe.py`)

A wrapper script that checks compatibility before loading the model:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr_safe.py
```

This script will:
- Check your transformers version
- Verify LlamaFlashAttention2 can be imported
- Provide clear error messages if issues are detected
- Only load the model if the environment is compatible

## Troubleshooting

### Q: I installed transformers==4.46.3 but still get the error

**A:** Try these steps:

1. Clear pip cache:
   ```bash
   pip cache purge
   ```

2. Reinstall transformers:
   ```bash
   pip uninstall transformers -y
   pip install --no-cache-dir transformers==4.46.3
   ```

3. Restart your Python kernel/runtime

4. Verify the version:
   ```python
   import transformers
   print(transformers.__version__)
   ```

### Q: Can I use a newer transformers version?

**A:** Not currently. The model code on Hugging Face Hub needs to be updated to support newer transformers versions. Until then, you must use transformers 4.46.0-4.51.9.

### Q: Will this be fixed in the future?

**A:** This issue requires updating the model code on Hugging Face Hub to either:
1. Use conditional imports for LlamaFlashAttention2
2. Update to use the new attention mechanism in transformers 4.52+

The DeepSeek team would need to update the model repository on Hugging Face.

## Related Issues

- [DeepSeek-OCR Issue #7](https://github.com/deepseek-ai/DeepSeek-OCR/issues/7)
- [DeepSeek-VL2 Issue #87](https://github.com/deepseek-ai/DeepSeek-VL2/issues/87)

## Testing Environment

This fix has been tested with:
- Python: 3.8, 3.9, 3.10, 3.11, 3.12
- transformers: 4.46.3
- torch: 2.0.0+
- CUDA: 11.8+

## Additional Notes

- The `requirements.txt` file has been updated with comments explaining the version constraint
- The README.md now includes a comprehensive troubleshooting section
- Both diagnostic tools are designed to be user-friendly and provide actionable error messages

## Support

If you continue to experience issues after following this guide:

1. Run `python check_compatibility.py` and share the output
2. Share your Python version: `python --version`
3. Share your transformers version: `pip show transformers`
4. Open an issue on the [DeepSeek-OCR GitHub repository](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
