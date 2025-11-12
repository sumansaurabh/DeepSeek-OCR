"""
Fixed version of run_dpsk_ocr.py that handles the LlamaFlashAttention2 import error
in transformers 4.57.1+

This script applies a patch before loading the model to prevent the ImportError.
"""

# Apply the fix BEFORE importing transformers
import sys
import os

# Add parent directory to path to import the fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import and apply the fix
try:
    from fix_flash_attention_import import apply_fix
    apply_fix()
except Exception as e:
    print(f"Warning: Could not apply fix automatically: {e}")
    print("Attempting to patch inline...")

    # Inline patch as fallback
    try:
        from transformers.models.llama import modeling_llama

        if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
            class LlamaFlashAttention2:
                def __init__(self, *args, **kwargs):
                    raise NotImplementedError(
                        "LlamaFlashAttention2 is not available in this transformers version."
                    )

            modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
            print("Inline patch applied successfully")
    except Exception as patch_error:
        print(f"Warning: Inline patch failed: {patch_error}")

# Now import transformers after patching
from transformers import AutoModel, AutoTokenizer
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

print("Model loaded successfully!")

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)
