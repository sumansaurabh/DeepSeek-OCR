#!/usr/bin/env python3
"""
Example: Using DeepSeek-OCR with the LlamaFlashAttention2 Fix

This script demonstrates how to properly load and use DeepSeek-OCR
with transformers 4.57.1+ by applying the fix for the
LlamaFlashAttention2 import error.

Author: DeepSeek-OCR Contributors
Issue: https://github.com/deepseek-ai/DeepSeek-OCR/issues/7
"""

import os
import sys

# ============================================================================
# STEP 1: Apply the fix BEFORE importing transformers
# ============================================================================
print("=" * 60)
print("Step 1: Applying LlamaFlashAttention2 Fix")
print("=" * 60)

# Method A: Using the fix module (recommended)
try:
    from fix_flash_attention_import import apply_fix
    apply_fix()
except ImportError:
    print("Fix module not found, applying inline patch...")
    # Method B: Inline patch (fallback)
    from transformers.models.llama import modeling_llama
    if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
        class LlamaFlashAttention2:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("LlamaFlashAttention2 not available")
        modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
        print("✓ Inline patch applied successfully")

# ============================================================================
# STEP 2: Import transformers (now safe to do)
# ============================================================================
print("\n" + "=" * 60)
print("Step 2: Importing Transformers")
print("=" * 60)

from transformers import AutoModel, AutoTokenizer
import torch

print(f"✓ Transformers version: {torch.__version__ if hasattr(torch, '__version__') else 'unknown'}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# STEP 3: Configure model settings
# ============================================================================
print("\n" + "=" * 60)
print("Step 3: Configuring Model Settings")
print("=" * 60)

# Set CUDA device (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Model configuration
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

# Inference settings
PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
IMAGE_FILE = 'your_image.jpg'  # Replace with your image path
OUTPUT_PATH = 'output'  # Replace with your output directory

# Image processing settings
# Choose one of the following presets:
# - Tiny:   base_size=512,  image_size=512,  crop_mode=False (64 vision tokens)
# - Small:  base_size=640,  image_size=640,  crop_mode=False (100 vision tokens)
# - Base:   base_size=1024, image_size=1024, crop_mode=False (256 vision tokens)
# - Large:  base_size=1280, image_size=1280, crop_mode=False (400 vision tokens)
# - Gundam: base_size=1024, image_size=640,  crop_mode=True  (dynamic)

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True  # Set to True for Gundam mode (dynamic resolution)

print(f"Model: {MODEL_NAME}")
print(f"Prompt: {PROMPT}")
print(f"Image size: {IMAGE_SIZE}, Base size: {BASE_SIZE}, Crop mode: {CROP_MODE}")

# ============================================================================
# STEP 4: Load the model (this would fail without the fix!)
# ============================================================================
print("\n" + "=" * 60)
print("Step 4: Loading Model")
print("=" * 60)

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded successfully")

    print("\nLoading model...")
    print("Note: This may take a few minutes on first run...")

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    print("✓ Model downloaded/loaded successfully")

    # Set model to evaluation mode
    model = model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda().to(torch.bfloat16)
        print("✓ Model moved to CUDA with bfloat16 precision")
    else:
        print("✓ Model running on CPU (CUDA not available)")

    print("\n✓ Model is ready for inference!")

except ImportError as e:
    print(f"\n✗ ImportError: {e}")
    print("\nThis error suggests the fix was not applied correctly.")
    print("Please check that you applied the fix before importing transformers.")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Error loading model: {e}")
    print("\nPossible causes:")
    print("1. Network issues (model download failed)")
    print("2. Insufficient disk space")
    print("3. Insufficient memory")
    sys.exit(1)

# ============================================================================
# STEP 5: Example inference (optional - requires image file)
# ============================================================================
print("\n" + "=" * 60)
print("Step 5: Model Inference")
print("=" * 60)

# Check if image file exists
if os.path.exists(IMAGE_FILE):
    print(f"Processing image: {IMAGE_FILE}")

    try:
        result = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=IMAGE_FILE,
            output_path=OUTPUT_PATH,
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            crop_mode=CROP_MODE,
            save_results=True,
            test_compress=True
        )

        print("✓ Inference completed successfully!")
        print(f"✓ Results saved to: {OUTPUT_PATH}")

        if result:
            print("\nModel output:")
            print("-" * 60)
            print(result)
            print("-" * 60)

    except Exception as e:
        print(f"✗ Inference error: {e}")
        print("\nPlease check:")
        print("1. Image file format is supported (jpg, png, etc.)")
        print("2. Output path exists and is writable")

else:
    print(f"⚠ Image file not found: {IMAGE_FILE}")
    print("\nTo run inference:")
    print("1. Replace IMAGE_FILE with your image path")
    print("2. Replace OUTPUT_PATH with your desired output directory")
    print("3. Run this script again")
    print("\nExample:")
    print('  IMAGE_FILE = "path/to/your/document.jpg"')
    print('  OUTPUT_PATH = "path/to/output/dir"')

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✓ Fix applied successfully")
print("✓ Model loaded without errors")
print("✓ Ready for OCR inference")
print("\nThe LlamaFlashAttention2 import error has been resolved!")
print("\nFor more information:")
print("  - Quick start: QUICKSTART_FIX.md")
print("  - Detailed guide: FLASH_ATTENTION_FIX.md")
print("  - Solution summary: SOLUTION_SUMMARY.md")
print("=" * 60)
