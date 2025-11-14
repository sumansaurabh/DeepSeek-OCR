#!/usr/bin/env python3
"""
Test script to validate the CUDA device-side assert fix.
This script tests the get_rel_pos function with various edge cases.
"""

import torch
import sys
import os

# Add the DeepSeek-OCR-vllm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

try:
    from deepencoder.sam_vary_sdpa import get_rel_pos
    print("✓ Successfully imported get_rel_pos function")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def test_get_rel_pos():
    """Test the get_rel_pos function with various input sizes."""
    
    test_cases = [
        # (q_size, k_size, rel_pos_size, description)
        (16, 16, 31, "Equal sizes - standard case"),
        (32, 16, 63, "Different sizes - q > k"),
        (16, 32, 63, "Different sizes - k > q"),
        (64, 64, 127, "Large equal sizes"),
        (100, 50, 199, "Large different sizes"),
        (1, 1, 1, "Minimal sizes"),
        (256, 256, 511, "Very large sizes"),
    ]
    
    print("\n" + "="*70)
    print("Testing get_rel_pos function with various input sizes")
    print("="*70)
    
    all_passed = True
    
    for q_size, k_size, rel_pos_size, description in test_cases:
        try:
            # Create a dummy relative position tensor
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rel_pos = torch.randn(rel_pos_size, 64, device=device)
            
            # Call the function
            result = get_rel_pos(q_size, k_size, rel_pos)
            
            # Verify the output shape
            expected_shape = (q_size, k_size, 64)
            if result.shape == expected_shape:
                print(f"✓ PASS: {description}")
                print(f"  Input: q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
                print(f"  Output shape: {result.shape}")
            else:
                print(f"✗ FAIL: {description}")
                print(f"  Expected shape: {expected_shape}, Got: {result.shape}")
                all_passed = False
                
        except RuntimeError as e:
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                print(f"✗ FAIL: {description}")
                print(f"  CUDA Error: {e}")
                all_passed = False
            else:
                raise
        except Exception as e:
            print(f"✗ FAIL: {description}")
            print(f"  Unexpected error: {e}")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests PASSED! The fix is working correctly.")
    else:
        print("✗ Some tests FAILED. Please review the errors above.")
    print("="*70 + "\n")
    
    return all_passed


def test_edge_cases():
    """Test edge cases that might trigger out-of-bounds indexing."""
    
    print("\n" + "="*70)
    print("Testing edge cases for out-of-bounds prevention")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test case that might have caused the original issue
    # When q_size and k_size are very different, relative_coords can go out of bounds
    edge_cases = [
        (100, 10, 199),  # Large ratio difference
        (10, 100, 199),  # Inverse ratio
        (200, 50, 399),  # Very large ratio
    ]
    
    all_passed = True
    
    for q_size, k_size, rel_pos_size in edge_cases:
        try:
            rel_pos = torch.randn(rel_pos_size, 64, device=device)
            result = get_rel_pos(q_size, k_size, rel_pos)
            print(f"✓ PASS: q_size={q_size}, k_size={k_size} - No CUDA errors")
        except RuntimeError as e:
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                print(f"✗ FAIL: q_size={q_size}, k_size={k_size} - CUDA Error: {e}")
                all_passed = False
            else:
                raise
    
    print("="*70 + "\n")
    return all_passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CUDA Device-Side Assert Fix Validation")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    # Run tests
    test1_passed = test_get_rel_pos()
    test2_passed = test_edge_cases()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED!")
        print("The CUDA device-side assert fix is working correctly.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the errors above.")
        sys.exit(1)
