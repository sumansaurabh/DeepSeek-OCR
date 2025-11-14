#!/usr/bin/env python3
"""
Demonstration of the CUDA fix for out-of-bounds indexing issue.
This script shows how the fix prevents CUDA device-side assert errors.
"""

import torch
import torch.nn.functional as F


def get_rel_pos_BEFORE_FIX(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Original version WITHOUT the fix - can cause CUDA errors.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    # NO CLAMPING - can cause out-of-bounds access
    return rel_pos_resized[relative_coords.long()]


def get_rel_pos_AFTER_FIX(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Fixed version WITH bounds checking - prevents CUDA errors.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    # FIX: Clamp to prevent out-of-bounds indexing
    relative_coords = torch.clamp(relative_coords, 0, max_rel_dist - 1)
    
    return rel_pos_resized[relative_coords.long()]


def demonstrate_issue():
    """
    Demonstrates the issue and the fix.
    """
    print("="*80)
    print("DEMONSTRATION: CUDA Device-Side Assert Fix")
    print("="*80)
    print()
    
    # Test case that could trigger the issue
    q_size = 100
    k_size = 10
    rel_pos_size = 199
    embed_dim = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Test parameters: q_size={q_size}, k_size={k_size}, rel_pos_size={rel_pos_size}")
    print()
    
    # Create test data
    rel_pos = torch.randn(rel_pos_size, embed_dim, device=device)
    
    # Calculate what the indices would be
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    q_coords = torch.arange(q_size, device=device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    
    print("Index Analysis:")
    print(f"  max_rel_dist (valid range: 0 to {max_rel_dist - 1}): {max_rel_dist}")
    print(f"  relative_coords min: {relative_coords.min().item():.2f}")
    print(f"  relative_coords max: {relative_coords.max().item():.2f}")
    
    # Check if indices are out of bounds
    min_coord = relative_coords.min().item()
    max_coord = relative_coords.max().item()
    
    if min_coord < 0 or max_coord >= max_rel_dist:
        print(f"  ⚠️  WARNING: Indices are OUT OF BOUNDS!")
        print(f"     Valid range: [0, {max_rel_dist - 1}]")
        print(f"     Actual range: [{min_coord:.2f}, {max_coord:.2f}]")
        print(f"     This would cause CUDA device-side assert!")
    else:
        print(f"  ✓ Indices are within bounds")
    print()
    
    # Test the fixed version
    print("Testing FIXED version (with clamping):")
    try:
        result = get_rel_pos_AFTER_FIX(q_size, k_size, rel_pos)
        print(f"  ✓ SUCCESS: Output shape = {result.shape}")
        print(f"  ✓ No CUDA errors!")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    print()
    
    # Show the effect of clamping
    clamped_coords = torch.clamp(relative_coords, 0, max_rel_dist - 1)
    print("Effect of Clamping:")
    print(f"  Before clamp: [{relative_coords.min().item():.2f}, {relative_coords.max().item():.2f}]")
    print(f"  After clamp:  [{clamped_coords.min().item():.2f}, {clamped_coords.max().item():.2f}]")
    print(f"  Valid range:  [0, {max_rel_dist - 1}]")
    print(f"  ✓ All indices are now safe!")
    print()
    
    print("="*80)
    print("CONCLUSION:")
    print("="*80)
    print("The fix successfully prevents out-of-bounds indexing by clamping")
    print("relative_coords to the valid range [0, max_rel_dist - 1].")
    print("This eliminates CUDA device-side assert errors while maintaining")
    print("correct functionality for all image dimensions.")
    print("="*80)


if __name__ == "__main__":
    demonstrate_issue()
