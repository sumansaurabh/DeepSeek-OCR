# Quick Start: Fix LlamaFlashAttention2 Import Error

Got the error `ImportError: cannot import name 'LlamaFlashAttention2'`? Here's how to fix it in 30 seconds.

## The Error

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

## Quick Fix (Copy & Paste)

### Option 1: Python Script

Add this at the **TOP** of your script, before any transformers imports:

```python
from transformers.models.llama import modeling_llama

if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
    class LlamaFlashAttention2:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("LlamaFlashAttention2 not available")
    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2

# Now import and use normally
from transformers import AutoModel, AutoTokenizer
```

### Option 2: Google Colab

**Cell 1:**
```python
from transformers.models.llama import modeling_llama

if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
    class LlamaFlashAttention2:
        def __init__(self, *args, **kwargs):
            pass
    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2

print("✓ Fix applied!")
```

**Cell 2:**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    use_safetensors=True
)

if torch.cuda.is_available():
    model = model.cuda().to(torch.bfloat16)

print("✓ Model loaded!")
```

### Option 3: Use Provided Script

```bash
# Instead of run_dpsk_ocr.py, use:
python run_dpsk_ocr_fixed.py
```

## That's It!

Your model will now load without errors. The fix is safe and doesn't affect model performance.

## Why Does This Work?

- The error occurs because transformers removed `LlamaFlashAttention2`
- DeepSeek-OCR's custom code imports it but never uses it
- We add a dummy class to satisfy the import
- The dummy class is never called, so everything works normally

## Need More Details?

See [FLASH_ATTENTION_FIX.md](FLASH_ATTENTION_FIX.md) for:
- Detailed explanation
- Alternative solutions
- Troubleshooting guide
- Technical details

## Test the Fix

```bash
python test_fix.py
```

Should output: `🎉 All tests passed!`

---

**Problem solved? Star the repo!** ⭐
