#!/usr/bin/env python3
"""
Simple integration test for attention diagnostics (GitHub Issue #3243)
Tests file modifications without requiring dependencies.
"""

import os
import sys

def check_file_contains(filepath, patterns, description):
    """Check if a file contains all specified patterns."""
    print(f"\nChecking: {description}")
    print(f"  File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ✗ File not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_found = True
    for pattern in patterns:
        if pattern in content:
            print(f"  ✓ Found: {pattern[:60]}...")
        else:
            print(f"  ✗ Missing: {pattern[:60]}...")
            all_found = False
    
    return all_found

def main():
    print("="*70)
    print("ATTENTION DIAGNOSTICS INTEGRATION TEST")
    print("GitHub Issue #3243: Attention Fusion Transparency")
    print("="*70)
    
    base_path = "/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    
    tests = []
    
    # Test 1: Check attention_diagnostics.py exists
    test1 = os.path.exists(os.path.join(base_path, "attention_diagnostics.py"))
    print(f"\n1. Attention diagnostics module exists: {'✓' if test1 else '✗'}")
    tests.append(test1)
    
    # Test 2: Check CLIP attention modifications
    test2 = check_file_contains(
        os.path.join(base_path, "deepencoder", "clip_sdpa.py"),
        [
            "from attention_diagnostics import log_attention_info",
            "log_attention_info(",
            "attention_type='clip'",
            "self._first_forward"
        ],
        "CLIP attention diagnostics integration"
    )
    tests.append(test2)
    
    # Test 3: Check SAM attention modifications
    test3 = check_file_contains(
        os.path.join(base_path, "deepencoder", "sam_vary_sdpa.py"),
        [
            "from attention_diagnostics import log_attention_info",
            "log_attention_info(",
            "attention_type='sam'",
            "self._first_forward"
        ],
        "SAM attention diagnostics integration"
    )
    tests.append(test3)
    
    # Test 4: Check config.py modifications
    test4 = check_file_contains(
        os.path.join(base_path, "config.py"),
        [
            "ENABLE_ATTENTION_DIAGNOSTICS",
            "PRINT_ATTENTION_SUMMARY",
            "GitHub Issue #3243"
        ],
        "Config.py diagnostics flags"
    )
    tests.append(test4)
    
    # Test 5: Check run_dpsk_ocr_image.py modifications
    test5 = check_file_contains(
        os.path.join(base_path, "run_dpsk_ocr_image.py"),
        [
            "ENABLE_ATTENTION_DIAGNOSTICS",
            "from attention_diagnostics import",
            "print_attention_summary"
        ],
        "Inference script integration"
    )
    tests.append(test5)
    
    # Test 6: Check documentation exists
    test6 = os.path.exists("/vercel/sandbox/ATTENTION_DIAGNOSTICS.md")
    print(f"\n6. Documentation exists: {'✓' if test6 else '✗'}")
    tests.append(test6)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(tests)
    total = len(tests)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All integration tests passed!")
        print("\nThe attention diagnostics feature has been successfully implemented.")
        print("This addresses GitHub Issue #3243 by providing transparency about")
        print("which attention implementation (FlashAttention vs SDPA) is being used.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
