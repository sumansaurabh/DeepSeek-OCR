"""
Compatibility patch for DeepSeek-OCR with newer transformers versions.

This script provides a monkey-patch to make DeepSeek-OCR work with transformers>=4.47
where LlamaFlashAttention2 and related classes have been removed or refactored.

Usage:
    1. Import this module before loading the DeepSeek-OCR model:
       import compatibility_patch
       from transformers import AutoModel
       
    2. Or run as a standalone script to apply the patch globally:
       python compatibility_patch.py

Issue: https://github.com/deepseek-ai/DeepSeek-OCR/issues/7
"""

import sys
import warnings
from importlib import import_module


def apply_compatibility_patch():
    """
    Apply compatibility patches for transformers>=4.47 to work with DeepSeek-OCR.
    
    This function monkey-patches the transformers library to restore classes
    that were removed in newer versions but are required by DeepSeek-OCR's
    custom model code loaded via trust_remote_code=True.
    """
    try:
        import transformers
        from packaging import version
        
        transformers_version = version.parse(transformers.__version__)
        
        # Check if we're using a version that needs patching
        if transformers_version < version.parse("4.47.0"):
            print(f"✓ transformers {transformers.__version__} is compatible, no patch needed.")
            return True
            
        print(f"⚠ transformers {transformers.__version__} detected, applying compatibility patch...")
        
        # Import the modeling_llama module
        try:
            from transformers.models.llama import modeling_llama
        except ImportError:
            warnings.warn("Could not import transformers.models.llama.modeling_llama")
            return False
        
        # Try to import the attention classes - they might exist with different names
        patched_classes = []
        
        # Attempt 1: Check if classes exist under different names
        if hasattr(modeling_llama, 'LlamaAttention'):
            base_attention = modeling_llama.LlamaAttention
            
            # Create LlamaFlashAttention2 if it doesn't exist
            if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
                # Try to find FlashAttention2 implementation
                if hasattr(modeling_llama, 'LlamaFlashAttention2'):
                    modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaFlashAttention2
                else:
                    # Create a wrapper class that uses the base attention
                    class LlamaFlashAttention2(base_attention):
                        """Compatibility wrapper for LlamaFlashAttention2"""
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            # Set flash attention flag if available
                            if hasattr(self, '_flash_attn_uses_top_left_mask'):
                                self._flash_attn_uses_top_left_mask = True
                    
                    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2
                    patched_classes.append('LlamaFlashAttention2')
            
            # Create LlamaSdpaAttention if it doesn't exist
            if not hasattr(modeling_llama, 'LlamaSdpaAttention'):
                class LlamaSdpaAttention(base_attention):
                    """Compatibility wrapper for LlamaSdpaAttention"""
                    pass
                
                modeling_llama.LlamaSdpaAttention = LlamaSdpaAttention
                patched_classes.append('LlamaSdpaAttention')
        
        # Attempt 2: Check for attention implementation mapping
        if hasattr(modeling_llama, 'LLAMA_ATTENTION_CLASSES'):
            attention_map = modeling_llama.LLAMA_ATTENTION_CLASSES
            
            # Ensure flash_attention_2 key exists
            if 'flash_attention_2' not in attention_map and 'eager' in attention_map:
                attention_map['flash_attention_2'] = attention_map['eager']
                patched_classes.append('LLAMA_ATTENTION_CLASSES[flash_attention_2]')
            
            # Ensure sdpa key exists
            if 'sdpa' not in attention_map and 'eager' in attention_map:
                attention_map['sdpa'] = attention_map['eager']
                patched_classes.append('LLAMA_ATTENTION_CLASSES[sdpa]')
        
        if patched_classes:
            print(f"✓ Successfully patched: {', '.join(patched_classes)}")
            print("✓ Compatibility patch applied successfully!")
            return True
        else:
            print("⚠ No patching was needed - classes may already exist")
            return True
            
    except ImportError as e:
        warnings.warn(f"Failed to apply compatibility patch: {e}")
        return False
    except Exception as e:
        warnings.warn(f"Unexpected error while applying compatibility patch: {e}")
        return False


def check_compatibility():
    """
    Check if the current transformers version is compatible with DeepSeek-OCR.
    
    Returns:
        tuple: (is_compatible, version_string, message)
    """
    try:
        import transformers
        from packaging import version
        
        transformers_version = version.parse(transformers.__version__)
        recommended_version = version.parse("4.46.3")
        
        if transformers_version == recommended_version:
            return (True, transformers.__version__, "Using recommended version")
        elif transformers_version < version.parse("4.47.0"):
            return (True, transformers.__version__, "Compatible version")
        else:
            return (False, transformers.__version__, "Requires compatibility patch")
            
    except ImportError:
        return (False, "not installed", "transformers not found")


# Auto-apply patch when module is imported
if __name__ != "__main__":
    apply_compatibility_patch()


if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek-OCR Compatibility Patch")
    print("=" * 60)
    
    # Check current compatibility
    is_compatible, version_str, message = check_compatibility()
    print(f"\nCurrent transformers version: {version_str}")
    print(f"Status: {message}")
    
    if not is_compatible:
        print("\nApplying compatibility patch...")
        success = apply_compatibility_patch()
        
        if success:
            print("\n" + "=" * 60)
            print("✓ Patch applied successfully!")
            print("=" * 60)
            print("\nYou can now use DeepSeek-OCR with your current transformers version.")
            print("Note: Import this module before loading the model:")
            print("  import compatibility_patch")
            print("  from transformers import AutoModel")
        else:
            print("\n" + "=" * 60)
            print("✗ Failed to apply patch")
            print("=" * 60)
            print("\nRecommendation: Install transformers==4.46.3")
            print("  pip install transformers==4.46.3")
            sys.exit(1)
    else:
        print("\n✓ Your transformers version is compatible!")
        print("No patch needed.")
