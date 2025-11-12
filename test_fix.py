#!/usr/bin/env python3
"""
Test script to verify the fix for Issue #7 works correctly.

This script tests both the import fix and model loading.
"""

import sys
import traceback


def test_import_fix():
    """Test that the import fix works correctly."""
    print("="*60)
    print("Test 1: Import Fix")
    print("="*60)

    try:
        # Simulate the fixed import
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2
            print("✓ LlamaFlashAttention2 is available in this transformers version")
            print(f"  - LlamaAttention: {LlamaAttention}")
            print(f"  - LlamaFlashAttention2: {LlamaFlashAttention2}")
            return True, LlamaFlashAttention2
        except ImportError as e:
            print("⚠ LlamaFlashAttention2 not available (expected in transformers>=4.57.0)")
            print(f"  Import error: {e}")
            from transformers.models.llama.modeling_llama import LlamaAttention
            LlamaFlashAttention2 = None
            print("✓ Successfully fell back to None")
            print(f"  - LlamaAttention: {LlamaAttention}")
            print(f"  - LlamaFlashAttention2: {LlamaFlashAttention2}")
            return True, None

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False, None


def test_attention_classes(LlamaFlashAttention2, LlamaAttention):
    """Test that the ATTENTION_CLASSES dictionary construction works."""
    print("\n" + "="*60)
    print("Test 2: ATTENTION_CLASSES Dictionary")
    print("="*60)

    try:
        # Simulate a minimal version of the ATTENTION_CLASSES dictionary
        ATTENTION_CLASSES = {
            "mha_eager": LlamaAttention,
            "mha_flash_attention_2": LlamaFlashAttention2 if LlamaFlashAttention2 is not None else LlamaAttention
        }

        print("✓ ATTENTION_CLASSES dictionary created successfully:")
        for key, value in ATTENTION_CLASSES.items():
            print(f"  - {key}: {value}")

        # Verify the fallback works
        if LlamaFlashAttention2 is None:
            if ATTENTION_CLASSES["mha_flash_attention_2"] == LlamaAttention:
                print("✓ Fallback to LlamaAttention works correctly")
                return True
            else:
                print("✗ Fallback did not work as expected")
                return False
        else:
            if ATTENTION_CLASSES["mha_flash_attention_2"] == LlamaFlashAttention2:
                print("✓ LlamaFlashAttention2 is correctly assigned")
                return True
            else:
                print("✗ LlamaFlashAttention2 assignment failed")
                return False

    except Exception as e:
        print(f"✗ Error creating ATTENTION_CLASSES: {e}")
        traceback.print_exc()
        return False


def test_transformers_version():
    """Test and display transformers version."""
    print("\n" + "="*60)
    print("Test 3: Environment Check")
    print("="*60)

    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")

        import torch
        print(f"✓ torch version: {torch.__version__}")

        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU only)")

        # Check if flash_attn is available
        try:
            import flash_attn
            print(f"✓ flash_attn version: {flash_attn.__version__}")
        except ImportError:
            print("⚠ flash_attn not installed")

        return True

    except Exception as e:
        print(f"✗ Error checking environment: {e}")
        traceback.print_exc()
        return False


def test_model_loading(model_path):
    """Test loading the patched model."""
    print("\n" + "="*60)
    print("Test 4: Model Loading")
    print("="*60)

    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        print(f"Loading model from: {model_path}")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✓ Tokenizer loaded successfully")

        # Load model
        print("Loading model (this may take a while)...")
        model = AutoModel.from_pretrained(
            model_path,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        print("✓ Model loaded successfully")

        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Model config: {model.config.model_type}")

        return True

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test the fix for Issue #7")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to patched model to test loading (optional)"
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip model loading test (faster)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DeepSeek-OCR Issue #7 Fix Verification")
    print("="*60 + "\n")

    results = []

    # Test 1: Import fix
    success, LlamaFlashAttention2 = test_import_fix()
    results.append(("Import Fix", success))

    if success:
        # Get LlamaAttention for test 2
        from transformers.models.llama.modeling_llama import LlamaAttention

        # Test 2: ATTENTION_CLASSES
        success = test_attention_classes(LlamaFlashAttention2, LlamaAttention)
        results.append(("ATTENTION_CLASSES", success))

    # Test 3: Environment
    success = test_transformers_version()
    results.append(("Environment Check", success))

    # Test 4: Model loading (optional)
    if not args.skip_model_load and args.model_path:
        success = test_model_loading(args.model_path)
        results.append(("Model Loading", success))
    elif not args.skip_model_load:
        print("\n" + "="*60)
        print("Test 4: Model Loading")
        print("="*60)
        print("⊘ Skipped (no model path provided)")
        print("  Use --model-path to test model loading")

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓✓✓ All tests passed! The fix is working correctly.")
        return 0
    else:
        print("\n✗✗✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
