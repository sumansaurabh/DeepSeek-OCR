#!/usr/bin/env python3
"""
Automatic patch script for DeepSeek-OCR Issue #7
Fixes the LlamaFlashAttention2 import error in transformers>=4.47.1
"""

import os
import sys
from pathlib import Path

def find_model_cache():
    """Find the cached DeepSeek-OCR model directory"""
    cache_dir = Path.home() / ".cache/huggingface/hub"

    if not cache_dir.exists():
        print("❌ HuggingFace cache directory not found.")
        print("   Please run the model at least once to download it first.")
        return None

    model_dirs = list(cache_dir.glob("models--deepseek-ai--DeepSeek-OCR/snapshots/*"))

    if not model_dirs:
        print("❌ DeepSeek-OCR model not found in cache.")
        print("   Please run the model at least once to download it first.")
        return None

    # Use the most recent snapshot
    model_dir = max(model_dirs, key=os.path.getmtime)
    return model_dir

def backup_file(file_path):
    """Create a backup of the original file"""
    backup_path = file_path.with_suffix('.py.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy(file_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    else:
        print(f"ℹ️  Backup already exists: {backup_path}")

def apply_patch(model_file):
    """Apply the patch to fix the import error"""
    print(f"\n🔧 Patching file: {model_file}")

    # Read the original file
    with open(model_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if '_llama_flash_attn2_available' in content:
        print("ℹ️  File is already patched!")
        return True

    # Apply the patches
    patches_applied = 0

    # Patch 1: Import statement
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

    if old_import in content:
        content = content.replace(old_import, new_import)
        patches_applied += 1
        print("  ✅ Patch 1/3: Import statement fixed")
    else:
        print("  ⚠️  Patch 1/3: Import pattern not found (may be already modified)")

    # Patch 2: ATTENTION_CLASSES dictionary
    old_attention = """    "mha_eager": LlamaAttention,
    "mha_flash_attention_2": LlamaFlashAttention2
}"""

    new_attention = """    "mha_eager": LlamaAttention,
    # Fallback to LlamaAttention if LlamaFlashAttention2 is not available
    "mha_flash_attention_2": LlamaFlashAttention2 if _llama_flash_attn2_available else LlamaAttention
}"""

    if old_attention in content:
        content = content.replace(old_attention, new_attention)
        patches_applied += 1
        print("  ✅ Patch 2/3: ATTENTION_CLASSES dictionary fixed")
    else:
        print("  ⚠️  Patch 2/3: ATTENTION_CLASSES pattern not found")

    # Patch 3: Add warning after logger definition
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

    if old_logger in content:
        content = content.replace(old_logger, new_logger)
        patches_applied += 1
        print("  ✅ Patch 3/3: Warning message added")
    else:
        print("  ⚠️  Patch 3/3: Logger pattern not found")

    # Write the patched content
    if patches_applied > 0:
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Successfully applied {patches_applied} patches!")
        return True
    else:
        print("\n❌ No patches were applied. The file may have a different structure.")
        return False

def main():
    print("=" * 70)
    print("  DeepSeek-OCR Issue #7 Fix - LlamaFlashAttention2 Import Error")
    print("=" * 70)
    print()

    # Find the model directory
    model_dir = find_model_cache()
    if model_dir is None:
        sys.exit(1)

    print(f"📁 Found model directory: {model_dir}")

    # Find the modeling file
    model_file = model_dir / "modeling_deepseekv2.py"
    if not model_file.exists():
        print(f"❌ File not found: {model_file}")
        sys.exit(1)

    print(f"📄 Found model file: {model_file}")

    # Create backup
    backup_file(model_file)

    # Apply patch
    success = apply_patch(model_file)

    if success:
        print("\n" + "=" * 70)
        print("✨ Patch completed successfully!")
        print("=" * 70)
        print("\nYou can now use DeepSeek-OCR without import errors.")
        print("\nTest with:")
        print("  from transformers import AutoModel, AutoTokenizer")
        print("  model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)")
        print()
    else:
        print("\n" + "=" * 70)
        print("⚠️  Patch failed - please apply manually")
        print("=" * 70)
        print("\nSee FIX_ISSUE_7.md for manual patching instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
