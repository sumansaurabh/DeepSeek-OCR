# Troubleshooting Guide

## Issue #7: ImportError: cannot import name 'LlamaFlashAttention2'

### Error Message
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

### Description
This error occurs when using transformers version 4.50.0 or higher. The `LlamaFlashAttention2` class was removed or restructured in newer versions of the transformers library, but the DeepSeek-OCR model code still references it.

### Root Cause
- DeepSeek-OCR uses `trust_remote_code=True` to load custom model code from Hugging Face
- The remote `modeling_deepseekv2.py` file imports `LlamaFlashAttention2` from transformers
- This class is not available in transformers >= 4.50.0
- The import is used for multi-head attention (MHA) with flash attention 2

### Solution 1: Use Compatible Transformers Version (Recommended)

**For new installations:**
```bash
pip install "transformers>=4.46.3,<4.50.0"
```

**For existing installations (Colab, etc.):**
```bash
pip uninstall transformers -y
pip install "transformers>=4.46.3,<4.50.0"
```

**Complete setup for Colab:**
```bash
# Uninstall existing transformers
pip uninstall transformers -y

# Install compatible version
pip install "transformers>=4.46.3,<4.50.0"
pip install tokenizers==0.20.3
pip install PyMuPDF img2pdf einops easydict addict Pillow numpy

# Install flash-attn if using GPU
pip install flash-attn==2.7.3 --no-build-isolation
```

### Solution 2: Use Eager Attention (Workaround)

If you must use transformers >= 4.50.0, you can disable flash attention:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Use 'eager' instead of 'flash_attention_2'
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='eager',  # Changed from 'flash_attention_2'
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
```

**Note:** This workaround will be slower than flash attention but will work with newer transformers versions.

### Solution 3: Use vLLM (Alternative)

For production use cases, consider using the vLLM implementation which has better version compatibility:

```bash
# Install vLLM (official upstream support as of 2025/10/23)
uv venv
source .venv/bin/activate
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

See the [vLLM Inference section](README.md#vllm-inference) in the README for usage examples.

### Verification

After applying the fix, verify your installation:

```python
import transformers
print(f"Transformers version: {transformers.__version__}")

# Should be between 4.46.3 and 4.50.0
version = tuple(map(int, transformers.__version__.split('.')[:2]))
if version >= (4, 50):
    print("⚠️  Warning: You may encounter LlamaFlashAttention2 import errors")
else:
    print("✅ Compatible transformers version")
```

### Related Issues
- [DeepSeek-VL2 Issue #87](https://github.com/deepseek-ai/DeepSeek-VL2/issues/87) - Similar issue in DeepSeek-VL2
- [DeepSeek-OCR Issue #7](https://github.com/deepseek-ai/DeepSeek-OCR/issues/7) - Original issue report

### Environment Information

**Tested Working Configuration:**
- transformers: 4.46.3 - 4.49.x
- torch: 2.6.0 (recommended) or 2.8.0+cu126
- flash-attn: 2.7.3
- Python: 3.12.9

**Known Incompatible:**
- transformers: >= 4.50.0 (with flash_attention_2)
- transformers: >= 4.57.1 (with flash_attention_2)

### Additional Help

If you continue to experience issues:
1. Check your transformers version: `pip show transformers`
2. Ensure you've restarted your Python kernel/runtime after reinstalling
3. Try the eager attention workaround (Solution 2)
4. Report the issue with your environment details on [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
