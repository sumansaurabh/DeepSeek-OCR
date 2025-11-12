# Fix for Issue #7: LlamaFlashAttention2 Import Error

## Problem Description

When using DeepSeek-OCR with transformers 4.57.1+ (and potentially earlier versions like 4.51.1+), users encounter the following error:

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

This occurs when loading the model with `trust_remote_code=True`:

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
```

## Root Cause

The issue stems from:
1. The DeepSeek-OCR model hosted on HuggingFace includes custom modeling code that imports `LlamaFlashAttention2`
2. This class has been removed or never existed in certain versions of the `transformers` library
3. **Importantly**: DeepSeek-OCR does not actually USE `LlamaFlashAttention2` - it's an unused import in the custom modeling code

## Solutions

### Solution 1: Use the Provided Fix Script (Recommended)

We've created a patch script that adds a fallback `LlamaFlashAttention2` class to the transformers module.

**Step 1**: Import and apply the fix before loading the model:

```python
from fix_flash_attention_import import apply_fix
apply_fix()

# Now load the model normally
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
```

**Step 2**: Or use the pre-patched script:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr_fixed.py
```

### Solution 2: Inline Patch

Add this code BEFORE importing transformers:

```python
# Apply patch before importing transformers
from transformers.models.llama import modeling_llama

if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
    class LlamaFlashAttention2:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "LlamaFlashAttention2 is not available in this transformers version. "
                "However, DeepSeek-OCR does not require this class."
            )

    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2

# Now import and use the model
from transformers import AutoModel, AutoTokenizer
# ... rest of your code
```

### Solution 3: Downgrade Transformers (Not Recommended)

Some users have reported success with older transformers versions:

```bash
pip install transformers==4.41.1
# or
pip install transformers==4.46.3  # The version in requirements.txt
```

**Note**: This is not recommended as it prevents you from using newer transformers features and bug fixes.

### Solution 4: Use vLLM Instead

The vLLM implementation (`DeepSeek-OCR-vllm/`) does not have this issue and is recommended for production use:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

## For Colab Users

In Google Colab, apply the fix at the beginning of your notebook:

```python
# Cell 1: Install dependencies
!pip install torch torchvision transformers einops pillow

# Cell 2: Apply the fix
from transformers.models.llama import modeling_llama

if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
    class LlamaFlashAttention2:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("LlamaFlashAttention2 not available")
    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
    print("✓ Patch applied successfully")

# Cell 3: Now load and use the model
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
model = model.eval()

# If CUDA is available
if torch.cuda.is_available():
    model = model.cuda().to(torch.bfloat16)

print("✓ Model loaded successfully!")
```

## Testing the Fix

To verify the fix works:

```python
import sys
sys.path.insert(0, '/path/to/DeepSeek-OCR')

from fix_flash_attention_import import apply_fix
apply_fix()

from transformers import AutoModel, AutoTokenizer
import torch

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'deepseek-ai/DeepSeek-OCR',
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded")

    print("Loading model...")
    model = AutoModel.from_pretrained(
        'deepseek-ai/DeepSeek-OCR',
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    print("✓ Model loaded successfully!")

    print("\nFix verified! The model can now be used normally.")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("The fix may need adjustment for your environment.")
except Exception as e:
    print(f"✗ Error: {e}")
```

## Technical Details

### Why This Works

1. The custom modeling code in the HuggingFace model repository imports `LlamaFlashAttention2` but never actually uses it
2. By adding a dummy class to the `transformers.models.llama.modeling_llama` module, we satisfy the import requirement
3. The dummy class raises an error if instantiated, but this never happens in DeepSeek-OCR's execution path
4. This approach is safe because:
   - It doesn't modify transformers source code
   - It only affects the current Python session
   - It doesn't interfere with actual model functionality

### Environment Tested

- ✓ transformers==4.57.1
- ✓ transformers==4.46.3
- ✓ torch==2.8.0+cu126
- ✓ torch==2.6.0
- ✓ Python 3.8+
- ✓ Google Colab

## Related Issues

- [DeepSeek-VL2 Issue #87](https://github.com/deepseek-ai/DeepSeek-VL2/issues/87)
- [DeepSeek-OCR Issue #7](https://github.com/deepseek-ai/DeepSeek-OCR/issues/7)

## Contributing

If you encounter issues with this fix or have improvements, please open an issue or submit a pull request.
