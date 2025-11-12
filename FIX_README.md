# DeepSeek-OCR Issue #7 Fix: LlamaFlashAttention2 ImportError

This directory contains the fix for [Issue #7](https://github.com/deepseek-ai/DeepSeek-OCR/issues/7) which causes an `ImportError` when using DeepSeek-OCR with transformers 4.57.1+.

## Problem

When running DeepSeek-OCR with recent versions of transformers (4.57.1+), users encounter:

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

This happens because:
1. The `LlamaFlashAttention2` class was removed/refactored in transformers 4.57+
2. DeepSeek-OCR's custom model code (in HuggingFace repo) still imports this class
3. The model fails to load even though most users don't need this class

## Files in This Fix

- **`ISSUE_7_FIX.md`**: Detailed analysis and solution documentation
- **`modeling_deepseekv2.patch`**: Patch file for the modeling code
- **`apply_fix.py`**: Python script to automatically download and patch the model
- **`FIX_README.md`**: This file

## Quick Solutions

### Option 1: Use vLLM (Recommended for Production)

The vLLM implementation doesn't have this issue:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

### Option 2: Downgrade Transformers (Quick Workaround)

```bash
pip install transformers==4.47.0
```

### Option 3: Apply the Fix (For Transformers Usage)

Use the provided script to download and patch the model:

```bash
# Install required dependencies
pip install huggingface-hub transformers

# Download and patch the model
python apply_fix.py --model-path ./deepseek-ocr-patched

# Or patch an existing model directory
python apply_fix.py --patch-existing /path/to/existing/model
```

Then use the patched model:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = './deepseek-ocr-patched'  # Use your patched model path

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Your inference code here...
```

## Technical Details

### What the Fix Does

The fix modifies `modeling_deepseekv2.py` in two places:

1. **Import section**: Wraps the import in a try-except block
   ```python
   try:
       from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2
   except ImportError:
       from transformers.models.llama.modeling_llama import LlamaAttention
       LlamaFlashAttention2 = None
   ```

2. **ATTENTION_CLASSES dictionary**: Uses conditional assignment
   ```python
   "mha_flash_attention_2": LlamaFlashAttention2 if LlamaFlashAttention2 is not None else LlamaAttention
   ```

### Why This Works

- DeepSeek-OCR primarily uses **MLA (Multi-Latent Attention)** mode, not MHA
- The `LlamaFlashAttention2` is only needed for MHA mode with Flash Attention
- By falling back to `LlamaAttention`, the model can still function for most use cases
- The fix is backward compatible with older transformers versions

## For Model Maintainers

If you maintain the DeepSeek-OCR model on HuggingFace, you can apply this fix by:

1. Cloning your model repository:
   ```bash
   git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
   cd DeepSeek-OCR
   ```

2. Applying the patch:
   ```bash
   patch -p1 < modeling_deepseekv2.patch
   ```

3. Committing and pushing:
   ```bash
   git add modeling_deepseekv2.py
   git commit -m "Fix: Handle missing LlamaFlashAttention2 in transformers>=4.57.0"
   git push
   ```

## Related Issues

- [DeepSeek-VL2 #87](https://github.com/deepseek-ai/DeepSeek-VL2/issues/87) - Same issue in DeepSeek-VL2
- [Axolotl #2266](https://github.com/axolotl-ai-cloud/axolotl/issues/2266) - Similar issue in Axolotl

## Testing

After applying the fix, verify it works:

```python
# test_fix.py
from transformers import AutoModel, AutoTokenizer
import torch

print("Testing DeepSeek-OCR with patched model...")

model_name = './deepseek-ocr-patched'  # Your patched model path

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer loaded successfully")

    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    print("✓ Model loaded successfully")

    print(f"✓ Model device: {model.device}")
    print(f"✓ Model dtype: {model.dtype}")
    print("\n✓✓✓ All tests passed! The fix is working correctly.")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
```

## Environment

Tested with:
- Python 3.8+
- transformers 4.57.1
- torch 2.8.0+cu126
- CUDA 11.8/12.6

## License

This fix is provided under the same license as the DeepSeek-OCR project (MIT License).

## Contributing

If you find issues with this fix or have suggestions for improvement, please:
1. Open an issue on the main DeepSeek-OCR repository
2. Reference Issue #7 in your report
3. Provide details about your environment and error messages
