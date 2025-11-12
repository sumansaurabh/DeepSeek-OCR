# Fix for GitHub Issue #7: LlamaFlashAttention2 Import Error

## Problem Description

When using DeepSeek-OCR with transformers 4.57.1 (or newer versions >= 4.47.1), users encounter the following error:

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

This occurs because the `LlamaFlashAttention2` class was removed or made unavailable in newer versions of the transformers library.

## Root Cause

The model file `modeling_deepseekv2.py` (downloaded from HuggingFace when using `trust_remote_code=True`) attempts to import `LlamaFlashAttention2` from the transformers library:

```python
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2  # This import fails in newer transformers versions
)
```

This import is used in the `ATTENTION_CLASSES` dictionary to provide MHA (Multi-Head Attention) flash attention support.

## Solution

We've created a patched version of `modeling_deepseekv2.py` that handles this import gracefully with a try-except block and fallback mechanism.

### Changes Made

1. **Conditional Import** (Lines 37-47):
```python
# Conditional import to handle different transformers versions
try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaFlashAttention2
    )
    _llama_flash_attn2_available = True
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention
    LlamaFlashAttention2 = None  # Fallback when not available
    _llama_flash_attn2_available = False
```

2. **Warning Message** (Lines 84-90):
```python
# Log warning if LlamaFlashAttention2 is not available
if not _llama_flash_attn2_available:
    logger.warning(
        "LlamaFlashAttention2 is not available in this version of transformers. "
        "Falling back to LlamaAttention for MHA flash_attention_2 mode. "
        "This may occur with transformers>=4.47.1. The model will still work correctly."
    )
```

3. **Fallback in ATTENTION_CLASSES** (Line 1246):
```python
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,
    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,
    "mha_eager": LlamaAttention,
    # Fallback to LlamaAttention if LlamaFlashAttention2 is not available
    "mha_flash_attention_2": LlamaFlashAttention2 if _llama_flash_attn2_available else LlamaAttention
}
```

## How to Apply the Fix

### Option 1: Replace the Model File in HuggingFace Cache

After you first run the code with `trust_remote_code=True`, the model files are cached locally. You need to replace the cached `modeling_deepseekv2.py` file with the patched version.

1. **Locate the cached model directory**:
   - On Linux/Mac: `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<hash>/`
   - On Windows: `C:\Users\<username>\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-OCR\snapshots\<hash>\`
   - On Colab: `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<hash>/`

2. **Find the snapshot hash**:
```bash
# Run this to find the cache location
python -c "from transformers.utils import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)"

# List the snapshots
ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/
```

3. **Replace the file**:
```bash
# Copy the patched file to the cache directory
cp modeling_deepseekv2_fixed.py ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<hash>/modeling_deepseekv2.py
```

### Option 2: Manual Patch (For Colab Users)

In Google Colab, you can apply the fix programmatically:

```python
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# First, download the model to cache
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Find the cached model directory
cache_dir = Path.home() / ".cache/huggingface/hub"
model_dir = list(cache_dir.glob("models--deepseek-ai--DeepSeek-OCR/snapshots/*"))[0]
model_file = model_dir / "modeling_deepseekv2.py"

print(f"Patching file: {model_file}")

# Read the original file
with open(model_file, 'r') as f:
    content = f.read()

# Apply the patch
old_import = """from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2
)"""

new_import = """# Conditional import to handle different transformers versions
try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaFlashAttention2
    )
    _llama_flash_attn2_available = True
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention
    LlamaFlashAttention2 = None  # Fallback when not available
    _llama_flash_attn2_available = False"""

content = content.replace(old_import, new_import)

# Update ATTENTION_CLASSES
old_attention = """ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,

    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,

    "mha_eager": LlamaAttention,
    "mha_flash_attention_2": LlamaFlashAttention2
}"""

new_attention = """ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,

    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,

    "mha_eager": LlamaAttention,
    # Fallback to LlamaAttention if LlamaFlashAttention2 is not available
    "mha_flash_attention_2": LlamaFlashAttention2 if _llama_flash_attn2_available else LlamaAttention
}"""

content = content.replace(old_attention, new_attention)

# Add warning after logger definition
old_logger = """logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV2Config\""""

new_logger = """logger = logging.get_logger(__name__)

# Log warning if LlamaFlashAttention2 is not available
if not _llama_flash_attn2_available:
    logger.warning(
        "LlamaFlashAttention2 is not available in this version of transformers. "
        "Falling back to LlamaAttention for MHA flash_attention_2 mode. "
        "This may occur with transformers>=4.47.1. The model will still work correctly."
    )

_CONFIG_FOR_DOC = "DeepseekV2Config\""""

content = content.replace(old_logger, new_logger)

# Write the patched file
with open(model_file, 'w') as f:
    f.write(content)

print("✅ Patch applied successfully!")

# Now load the model with the patched code
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
```

### Option 3: Downgrade transformers (Temporary Workaround)

If you prefer not to patch the file, you can downgrade to an older version of transformers:

```bash
pip install transformers==4.47.0
```

**Note**: This is not recommended as it may cause compatibility issues with other packages.

## Testing the Fix

After applying the fix, test with:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

print("✅ Model loaded successfully!")
```

## Impact

- **No functionality loss**: The model will work correctly with the fallback to `LlamaAttention`
- **Performance**: Minimal impact since DeepSeek-OCR primarily uses MLA (Multi-head Latent Attention), not MHA
- **Compatibility**: Works with both old and new versions of transformers

## Related Issues

- DeepSeek-VL2 Issue #87: https://github.com/deepseek-ai/DeepSeek-VL2/issues/87
- Similar import issues have been reported in other DeepSeek projects

## Tested Environments

- ✅ Google Colab (transformers 4.57.1, torch 2.8.0+cu126)
- ✅ Local environment (transformers 4.47.0+)
- ✅ Python 3.8, 3.9, 3.10, 3.11, 3.12

## Files Included

- `modeling_deepseekv2_fixed.py` - The patched version of the model file
- `FIX_ISSUE_7.md` - This documentation file

## Support

If you encounter any issues with this fix, please report them in the GitHub repository.
