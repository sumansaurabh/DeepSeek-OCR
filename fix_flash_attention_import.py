"""
Fix for LlamaFlashAttention2 import error in DeepSeek-OCR with transformers 4.57.1+

This script patches the transformers library to handle the missing LlamaFlashAttention2 import
that occurs when using trust_remote_code=True with DeepSeek-OCR models.

Issue: https://github.com/deepseek-ai/DeepSeek-OCR/issues/7
Related: https://github.com/deepseek-ai/DeepSeek-VL2/issues/87
"""

import sys
import importlib.util
from types import ModuleType


def create_flash_attention_fallback():
    """
    Create a fallback module for LlamaFlashAttention2.
    This allows the model to load without errors even though this class is not used.
    """
    # Create a dummy LlamaFlashAttention2 class
    class LlamaFlashAttention2:
        """Fallback class for LlamaFlashAttention2 that is not used in DeepSeek-OCR"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "LlamaFlashAttention2 is not available in this transformers version. "
                "However, DeepSeek-OCR does not require this class to function properly."
            )

    return LlamaFlashAttention2


def patch_transformers_llama():
    """
    Patch the transformers.models.llama.modeling_llama module to add LlamaFlashAttention2
    """
    try:
        # Import the modeling_llama module
        from transformers.models.llama import modeling_llama

        # Check if LlamaFlashAttention2 is already available
        if hasattr(modeling_llama, 'LlamaFlashAttention2'):
            print("LlamaFlashAttention2 already exists in transformers.models.llama.modeling_llama")
            return True

        # Add the fallback class to the module
        modeling_llama.LlamaFlashAttention2 = create_flash_attention_fallback()
        print("Successfully patched transformers.models.llama.modeling_llama with LlamaFlashAttention2 fallback")
        return True

    except ImportError as e:
        print(f"Warning: Could not import transformers.models.llama.modeling_llama: {e}")
        return False
    except Exception as e:
        print(f"Error patching transformers: {e}")
        return False


def apply_fix():
    """
    Apply the fix for the LlamaFlashAttention2 import error.
    Call this function before loading DeepSeek-OCR models with trust_remote_code=True.
    """
    print("Applying fix for LlamaFlashAttention2 import error...")
    success = patch_transformers_llama()

    if success:
        print("Fix applied successfully! You can now load DeepSeek-OCR models.")
    else:
        print("Warning: Fix could not be fully applied. The model may still encounter import errors.")

    return success


if __name__ == "__main__":
    # Apply the fix when run as a script
    apply_fix()
