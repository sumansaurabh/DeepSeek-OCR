#!/usr/bin/env python3
"""
Test script to verify the CUDA device-side assert fix.
This script tests the get_rel_pos function with various input sizes
that could potentially trigger out-of-bounds indexing.
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
    """Test get_rel_pos with various input sizes."""
    print("\n" + "="*60)
    print("Testing get_rel_pos function with various input sizes")
    print("="*60)
    
    # Test cases with different q_size and k_size combinations
    test_cases = [
        (16, 16, 64),   # Equal sizes
        (32, 16, 128),  # Different sizes
        (64, 32, 256),  # Larger sizes
        (8, 16, 32),    # Smaller q_size
        (16, 8, 32),    # Smaller k_size
        (100, 50, 200), # Edge case sizes
        (1, 1, 16),     # Minimal sizes
        (256, 256, 512),# Large sizes
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cpu':
        print("⚠ Warning: CUDA not available, testing on CPU")
        print("  The CUDA device-side assert error can only be fully tested on GPU")
    
    passed = 0
    failed = 0
    
    for q_size, k_size, rel_pos_size in test_cases:
        try:
            # Create a random relative position tensor
            rel_pos = torch.randn(rel_pos_size, 64, device=device)
            
            # Call get_rel_pos
            result = get_rel_pos(q_size, k_size, rel_pos)
            
            # Verify output shape
            expected_shape = (q_size, k_size, 64)
            if result.shape == expected_shape:
                print(f"✓ Test passed: q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
                print(f"  Output shape: {result.shape}")
                passed += 1
            else:
                print(f"✗ Test failed: q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
                print(f"  Expected shape: {expected_shape}, Got: {result.shape}")
                failed += 1
                
        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e):
                print(f"✗ CUDA Error: q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
                print(f"  Error: {e}")
                failed += 1
            else:
                raise
        except Exception as e:
            print(f"✗ Unexpected error: q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


def test_bounds_checking():
    """Test that the bounds checking prevents out-of-bounds access."""
    print("\n" + "="*60)
    print("Testing bounds checking for edge cases")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test case that would previously cause out-of-bounds error
    # This simulates the scenario where relative_coords could exceed max_rel_dist
    try:
        q_size, k_size = 40, 20
        rel_pos_size = 79  # 2 * max(40, 20) - 1 = 79
        
        rel_pos = torch.randn(rel_pos_size, 32, device=device)
        result = get_rel_pos(q_size, k_size, rel_pos)
        
        print(f"✓ Bounds checking test passed")
        print(f"  q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
        print(f"  Output shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Bounds checking test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUDA Device-Side Assert Fix Validation")
    print("="*60)
    
    # Run tests
    test1_passed = test_get_rel_pos()
    test2_passed = test_bounds_checking()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("✓ All tests passed! The fix is working correctly.")
        print("\nThe CUDA device-side assert error should now be resolved.")
        print("Images that previously caused errors should now process successfully.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please review the errors above.")
        sys.exit(1)
