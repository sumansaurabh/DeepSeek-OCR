# Quick Fix Guide for Issue #7

## 🚨 Error: `ImportError: cannot import name 'LlamaFlashAttention2'`

### ⚡ Quick Fix (Recommended)

**In Colab or Jupyter:**
```bash
!pip uninstall transformers -y
!pip install "transformers>=4.46.3,<4.50.0"
```

**In Terminal:**
```bash
pip uninstall transformers -y
pip install "transformers>=4.46.3,<4.50.0"
```

Then restart your Python kernel/runtime.

### ✅ Verify Fix

```python
import transformers
print(f"Transformers version: {transformers.__version__}")
# Should show version between 4.46.3 and 4.49.x
```

Or run:
```bash
python verify_transformers_version.py
```

### 🔄 Alternative: Use Eager Attention

If you **must** use transformers >= 4.50.0:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Change this line:
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='eager',  # ← Changed from 'flash_attention_2'
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
```

**Note:** This will be slower but will work with newer transformers versions.

### 📚 More Details

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive solutions and explanations.

### 🆘 Still Having Issues?

1. Make sure you restarted your Python kernel after reinstalling transformers
2. Check your version: `pip show transformers`
3. Try the eager attention workaround above
4. Report the issue on [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR/issues) with your environment details
