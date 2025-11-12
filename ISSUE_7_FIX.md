# Fix for Issue #7: LlamaFlashAttention2 ImportError

## Problem Summary

When using DeepSeek-OCR with `transformers>=4.57.0`, users encounter the following error:
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

This occurs because the `LlamaFlashAttention2` class was removed/refactored in transformers 4.57+, but the model's custom code (loaded via `trust_remote_code=True`) still tries to import it.

## Root Cause

The issue is in the HuggingFace model repository file `modeling_deepseekv2.py`:
- **Line 37-40**: Imports `LlamaFlashAttention2` from transformers
- **Line 1238**: Uses it in the `ATTENTION_CLASSES` dictionary for MHA mode with flash attention

## Solution

There are three approaches to fix this issue:

### Option 1: Conditional Import with Fallback (Recommended)

Modify the import to gracefully handle missing `LlamaFlashAttention2` and use the local `DeepseekV2FlashAttention2` class as a fallback.

### Option 2: Downgrade transformers (Temporary Workaround)

```bash
pip install transformers==4.47.0
```

This is only a temporary solution and not recommended for long-term use.

### Option 3: Use Standard Llama Attention (For MLA Mode)

Since the DeepSeek-OCR model primarily uses MLA (Multi-Latent Attention) mode, and the `LlamaFlashAttention2` is only used for MHA mode, most users won't be affected if MHA mode is not available.

## Implementation

The fix requires updating the HuggingFace model repository's `modeling_deepseekv2.py` file. Since users cannot directly modify files in the HuggingFace repository, there are two paths:

### Path A: For Model Maintainers

Update `modeling_deepseekv2.py` in the HuggingFace repository with the following changes:

**1. Update the import section (lines 37-40):**

```python
# Try to import LlamaFlashAttention2, fall back to None if not available
try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaFlashAttention2
    )
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention
    LlamaFlashAttention2 = None
```

**2. Update the ATTENTION_CLASSES dictionary (line 1230-1239):**

```python
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,

    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,

    "mha_eager": LlamaAttention,
    "mha_flash_attention_2": LlamaFlashAttention2 if LlamaFlashAttention2 is not None else LlamaAttention
}
```

### Path B: For End Users (Workaround)

Until the official fix is released, users can:

1. **Use vLLM Implementation** (Recommended for production):
   ```python
   # Use the vLLM implementation which doesn't have this issue
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   python run_dpsk_ocr_image.py
   ```

2. **Downgrade transformers**:
   ```bash
   pip install transformers==4.47.0
   ```

3. **Local Model Copy with Patch**: Download the model, apply the fix locally, and load from local path

## Testing

After applying the fix, test with:

```python
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
print("Model loaded successfully!")
```

## Additional Notes

- The DeepSeek-OCR model primarily uses **MLA (Multi-Latent Attention)** mode, not MHA mode
- The `LlamaFlashAttention2` import is only relevant when the model is configured to use MHA mode
- Most users will not be affected by this issue in practice, but the import error prevents model loading entirely
- This is a similar issue to DeepSeek-VL2 #87

## Recommended Action

**For the DeepSeek-AI team**: Please update the `modeling_deepseekv2.py` file in the HuggingFace repository with the conditional import fix shown above.

**For users experiencing this issue right now**: Use the vLLM implementation or downgrade transformers to 4.47.0 as a temporary workaround.
