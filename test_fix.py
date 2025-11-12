#!/usr/bin/env python3
"""
Test script to verify the LlamaFlashAttention2 import fix works correctly.

Usage:
    python test_fix.py

This script will:
1. Apply the fix
2. Attempt to import transformers modules
3. Verify the patch is working
4. Simulate the model loading process (without actually downloading the model)
"""

import sys
import os


def test_patch_application():
    """Test if the patch can be applied successfully"""
    print("=" * 60)
    print("Test 1: Applying the patch")
    print("=" * 60)

    try:
        from fix_flash_attention_import import apply_fix
        result = apply_fix()

        if result:
            print("✓ Patch applied successfully")
            return True
        else:
            print("✗ Patch application returned False")
            return False

    except Exception as e:
        print(f"✗ Error applying patch: {e}")
        return False


def test_import_llama_module():
    """Test if the transformers llama module can be imported"""
    print("\n" + "=" * 60)
    print("Test 2: Importing transformers.models.llama.modeling_llama")
    print("=" * 60)

    try:
        from transformers.models.llama import modeling_llama
        print("✓ Successfully imported modeling_llama")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_flash_attention_class_exists():
    """Test if LlamaFlashAttention2 exists in the module"""
    print("\n" + "=" * 60)
    print("Test 3: Checking if LlamaFlashAttention2 is available")
    print("=" * 60)

    try:
        from transformers.models.llama import modeling_llama

        if hasattr(modeling_llama, 'LlamaFlashAttention2'):
            print("✓ LlamaFlashAttention2 class is available")
            print(f"  Class: {modeling_llama.LlamaFlashAttention2}")
            return True
        else:
            print("✗ LlamaFlashAttention2 class not found")
            return False

    except Exception as e:
        print(f"✗ Error checking class: {e}")
        return False


def test_transformers_import():
    """Test if transformers AutoModel can be imported"""
    print("\n" + "=" * 60)
    print("Test 4: Importing transformers AutoModel and AutoTokenizer")
    print("=" * 60)

    try:
        from transformers import AutoModel, AutoTokenizer
        print("✓ Successfully imported AutoModel and AutoTokenizer")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_simulated_import_from_custom_code():
    """Simulate what the custom modeling code does"""
    print("\n" + "=" * 60)
    print("Test 5: Simulating custom model code import")
    print("=" * 60)

    try:
        # This simulates what happens in the custom modeling code
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        print("✓ Successfully imported LlamaFlashAttention2 (this would have failed before the fix)")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  This is the error users were experiencing!")
        return False


def print_environment_info():
    """Print information about the environment"""
    print("\n" + "=" * 60)
    print("Environment Information")
    print("=" * 60)

    try:
        import transformers
        print(f"transformers version: {transformers.__version__}")
    except:
        print("transformers: Not installed")

    try:
        import torch
        print(f"torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except:
        print("torch: Not installed")

    print(f"Python version: {sys.version}")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 60)
    print("LlamaFlashAttention2 Fix - Test Suite")
    print("=" * 60)

    print_environment_info()

    tests = [
        ("Patch Application", test_patch_application),
        ("Import Llama Module", test_import_llama_module),
        ("Flash Attention Class", test_flash_attention_class_exists),
        ("Transformers Import", test_transformers_import),
        ("Custom Code Simulation", test_simulated_import_from_custom_code),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised an exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The fix is working correctly.")
        print("\nYou can now use DeepSeek-OCR with transformers 4.57.1+ by:")
        print("1. Importing fix_flash_attention_import and calling apply_fix()")
        print("2. Or using the run_dpsk_ocr_fixed.py script")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure transformers is installed: pip install transformers")
        print("2. Ensure fix_flash_attention_import.py is in the same directory or PYTHONPATH")
        print("3. Check that you have the required Python version (3.8+)")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
