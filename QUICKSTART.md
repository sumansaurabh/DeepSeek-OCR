# Quick Start Guide - Fix for Issue #7

Having trouble loading DeepSeek-OCR with this error?

```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

**You're in the right place!** Here's how to fix it in 5 minutes or less.

---

## ⚡ Fastest Solution (1 minute)

Just downgrade transformers:

```bash
pip install transformers==4.47.0
```

Then run your code normally. ✅ Done!

---

## 🔧 Better Solution: Use vLLM (2 minutes)

The vLLM implementation doesn't have this issue:

```bash
# Navigate to vLLM directory
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Run inference
python run_dpsk_ocr_image.py
```

✅ Production-ready and faster!

---

## 🛠️ Best Solution: Apply the Fix (5 minutes)

This keeps your transformers up-to-date:

### Step 1: Run the fix script

```bash
python apply_fix.py --model-path ./deepseek-ocr-patched
```

### Step 2: Use the patched model

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Point to your patched model
model_name = './deepseek-ocr-patched'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Now run your inference
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'your_image.jpg'
res = model.infer(tokenizer, prompt=prompt, image_file=image_file)
```

✅ All set!

---

## 🔍 Need More Info?

- **Full technical details**: See `ISSUE_7_FIX.md`
- **Implementation guide**: See `FIX_README.md`
- **Executive summary**: See `SOLUTION_SUMMARY.md`

---

## 💡 Which Solution Should I Use?

| Solution | Use When | Pros | Cons |
|----------|----------|------|------|
| **Downgrade** | Quick test/debug | Fastest (30 seconds) | Outdated library |
| **vLLM** | Production deployment | Fast, stable, no fix needed | Different API |
| **Apply Fix** | Development with transformers | Up-to-date libraries | Takes 5 minutes |

---

## ❓ Still Having Issues?

1. Check your transformers version: `pip show transformers`
2. Make sure you have the required dependencies: `pip install -r requirements.txt`
3. Read the detailed docs: `ISSUE_7_FIX.md`
4. Run tests: `python test_fix.py --skip-model-load`

---

## 🎯 What This Fix Does

The fix makes the model code work with both:
- ✅ Old transformers (< 4.57.0) - where `LlamaFlashAttention2` exists
- ✅ New transformers (≥ 4.57.0) - where it was removed

**Technical**: It uses a try-except block to import the class if available, or fall back to standard attention if not.

**Impact**: Zero for most users (model uses MLA mode by default, not MHA).

---

## 📦 What You Get

After running the fix, you'll have:
- A patched version of the model that works with any transformers version
- All the original functionality intact
- No performance degradation
- Future-proof code

---

## 🚀 Ready to Go!

Pick your solution above and get back to building awesome OCR applications with DeepSeek-OCR!

Questions? Check out the other documentation files or open an issue on GitHub.
