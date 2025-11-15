# Fix for GitHub Issue #93: CUDA device-side assert triggered

## Problem Description

Some images cause a `RuntimeError: CUDA error: device-side assert triggered` during inference, while others process successfully. The error message (in Chinese) asks: "有的图片可以正常处理，有的会出现这个错误，这是什么原因呢" (Some images can be processed normally, but some get this error, what's the reason?)

## Root Cause Analysis

The CUDA device-side assert error occurs due to **invalid tensor reshape operations** when feature map dimensions are not perfect squares. Specifically:

### Architecture Flow

1. **Image Input**: Images are padded to `BASE_SIZE` (default: 1024x1024) for global view and cropped to `IMAGE_SIZE` (default: 640x640) patches
2. **Patch Embedding**: SAM ViT-B model uses 16x16 patch embedding (stride=16)
3. **Convolution Layers**: Two stride-2 convolutions reduce spatial dimensions:
   - Input: `(H/16, W/16)` patches
   - After conv2 (stride=2): `(H/32, W/32)`
   - After conv3 (stride=2): `(H/64, W/64)`
4. **Feature Flattening**: Features are flattened to shape `[B, H*W/64/64, C]`
5. **Critical Assumption**: Code at `deepseek_ocr.py:415-419` assumes `H*W/64/64` is a **perfect square**

### The Bug

```python
# Line 415-416 (and 418-419, 452-453)
_, hw, n_dim = global_features.shape
h = w = int(hw ** 0.5)  # ❌ Assumes hw is a perfect square!
```

If `hw` is NOT a perfect square (e.g., 96), then:
- `int(96 ** 0.5) = 9`
- But `9 * 9 = 81 ≠ 96`
- Subsequent `.view(h, w, n_dim)` operation fails with CUDA assert

### When Does This Happen?

This occurs when:
1. **Improper image dimensions**: Images not properly padded to required sizes
2. **Invalid configuration**: `BASE_SIZE` or `IMAGE_SIZE` not divisible by 64
3. **Preprocessing failures**: Edge cases in dynamic cropping or padding

## Solution Implemented

### 1. Validation in Model Forward Pass (`deepseek_ocr.py`)

Added three layers of validation at the critical reshape points:

**Location 1: Lines 415-425 (global features with crops)**
```python
_, hw, n_dim = global_features.shape
# Validate that hw is a perfect square to avoid CUDA device-side assert
sqrt_hw = hw ** 0.5
if sqrt_hw != int(sqrt_hw):
    raise ValueError(
        f"Expected global features spatial dimensions to be a perfect square, "
        f"but got hw={hw} (sqrt={sqrt_hw:.2f}). "
        f"This may be caused by incompatible image dimensions. "
        f"Image shape: {image_ori.shape}, Features shape: {global_features.shape}"
    )
h = w = int(sqrt_hw)
```

**Location 2: Lines 427-437 (local features validation)**
```python
_2, hw2, n_dim2 = local_features.shape
sqrt_hw2 = hw2 ** 0.5
if sqrt_hw2 != int(sqrt_hw2):
    raise ValueError(
        f"Expected local features spatial dimensions to be a perfect square, "
        f"but got hw2={hw2} (sqrt={sqrt_hw2:.2f}). "
        f"This may be caused by incompatible patch dimensions. "
        f"Patches shape: {patches.shape}, Features shape: {local_features.shape}"
    )
h2 = w2 = int(sqrt_hw2)
```

**Location 3: Lines 441-451 (reshape validation)**
```python
# Validate reshape dimensions before performing the operation
expected_local_size = height_crop_num * width_crop_num * h2 * w2 * n_dim2
actual_local_size = local_features.numel()
if expected_local_size != actual_local_size:
    raise ValueError(
        f"Cannot reshape local_features: expected total size {expected_local_size} "
        f"(height_crop_num={height_crop_num} * width_crop_num={width_crop_num} * "
        f"h2={h2} * w2={w2} * n_dim2={n_dim2}) "
        f"but got {actual_local_size} elements. "
        f"Features shape: {local_features.shape}"
    )
```

**Location 4: Lines 482-492 (no patches branch)**
```python
_, hw, n_dim = global_features.shape
sqrt_hw = hw ** 0.5
if sqrt_hw != int(sqrt_hw):
    raise ValueError(
        f"Expected global features spatial dimensions to be a perfect square, "
        f"but got hw={hw} (sqrt={sqrt_hw:.2f}). "
        f"This may be caused by incompatible image dimensions. "
        f"Image shape: {image_ori.shape}, Features shape: {global_features.shape}"
    )
h = w = int(sqrt_hw)
```

### 2. Preprocessing Validation (`process/image_process.py`)

Added early validation to catch configuration errors:

**Global View Validation (Lines 389-404)**
```python
# Validate that the global view has the expected dimensions
if global_view.size != (self.base_size, self.base_size):
    raise ValueError(
        f"Global view padding failed: expected size ({self.base_size}, {self.base_size}), "
        f"but got {global_view.size}. Original image size: {image.size}"
    )

# Additional validation: ensure base_size produces square feature maps
# For ViT with patch_size=16 and 2 stride-2 convolutions, final size is (base_size/16/2/2)^2
# This means base_size must be divisible by 64 (16 * 2 * 2)
if self.base_size % 64 != 0:
    raise ValueError(
        f"base_size={self.base_size} will produce non-integer feature map dimensions. "
        f"base_size must be divisible by 64 to ensure proper feature extraction. "
        f"Recommended values: 512, 640, 1024, 1280, 1536, etc."
    )
```

**Crop Validation (Lines 418-440)**
```python
if num_width_tiles > 1 or num_height_tiles > 1:
    # Validate image_size for crops as well
    if self.image_size % 64 != 0:
        raise ValueError(
            f"image_size={self.image_size} will produce non-integer feature map dimensions in crops. "
            f"image_size must be divisible by 64 to ensure proper feature extraction. "
            f"Recommended values: 512, 640, 768, 1024, etc."
        )

    for i in range(len(images_crop_raw)):
        crop_img = images_crop_raw[i]
        # Validate crop size
        if crop_img.size != (self.image_size, self.image_size):
            raise ValueError(
                f"Crop {i} has incorrect size: expected ({self.image_size}, {self.image_size}), "
                f"but got {crop_img.size}"
            )
        images_crop_list.append(self.image_transform(crop_img))
```

## Benefits of This Fix

1. **Prevents CUDA Errors**: Catches invalid dimensions BEFORE they cause CUDA device-side asserts
2. **Clear Error Messages**: Provides detailed diagnostic information about:
   - What went wrong (non-square dimensions)
   - Where it went wrong (image shapes, feature shapes)
   - How to fix it (recommended values)
3. **Early Detection**: Validates in preprocessing to fail fast with clear Python exceptions instead of cryptic CUDA errors
4. **Debugging Aid**: Error messages include actual tensor shapes for troubleshooting

## Recommended Configuration Values

For `BASE_SIZE` and `IMAGE_SIZE` in `config.py`, use values divisible by 64:

| Mode | BASE_SIZE | IMAGE_SIZE | Description |
|------|-----------|------------|-------------|
| Tiny | 512 | 512 | Lowest resolution |
| Small | 640 | 640 | Small documents |
| Base | 1024 | 1024 | Standard resolution |
| Large | 1280 | 1280 | High resolution |
| **Gundam** | **1024** | **640** | **Default: Multi-scale** |
| Ultra | 1536 | 768 | Very high resolution |

**✓ Valid values**: 512, 640, 768, 1024, 1280, 1536, 1792, 2048, etc.
**✗ Invalid values**: 500, 600, 800, 900, 1100, 1200, etc.

## Testing Recommendations

To verify the fix works:

1. **Test with valid configurations**:
   ```python
   # config.py
   BASE_SIZE = 1024  # ✓ Valid
   IMAGE_SIZE = 640  # ✓ Valid
   ```

2. **Test with invalid configurations** (should raise clear errors):
   ```python
   # config.py
   BASE_SIZE = 1000  # ✗ Not divisible by 64
   IMAGE_SIZE = 600  # ✗ Not divisible by 64
   ```

3. **Test with various image sizes**:
   - Square images (512x512, 1024x1024)
   - Rectangular images (800x600, 1920x1080)
   - Very small images (100x100)
   - Very large images (4096x4096)

4. **Test with problematic images** from issue #93:
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   python run_dpsk_ocr_image.py
   ```

## Expected Behavior

### Before Fix
- ❌ Silent CUDA device-side assert error
- ❌ Cryptic error message with no useful information
- ❌ Requires `CUDA_LAUNCH_BLOCKING=1` for debugging

### After Fix
- ✅ Clear Python ValueError with diagnostic information
- ✅ Error message explains the problem and solution
- ✅ Fails fast in preprocessing (for config issues) or model forward pass (for edge cases)
- ✅ Includes tensor shapes and recommended fixes

## Files Modified

1. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`**
   - Lines 415-451: Added validation for global and local features (with crops)
   - Lines 482-492: Added validation for global features (without crops)

2. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py`**
   - Lines 389-404: Added global view size validation
   - Lines 418-440: Added crop size validation

## Technical Details

### Why 64?
The requirement for divisibility by 64 comes from:
- **Patch size**: 16 (from ViT architecture)
- **Stride-2 conv #1**: Divides by 2
- **Stride-2 conv #2**: Divides by 2
- **Total**: 16 × 2 × 2 = 64

For a square output: `(size / 64)²` must yield an integer dimension.

### Perfect Square Requirement
The code assumes flattened features can be reshaped to `(h, w, C)` where `h = w`. This requires:
```
hw = h × w = h² (since h = w)
∴ hw must be a perfect square
```

Valid `hw` values: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, ...

For common image sizes:
- 512×512 → 512/64 = 8 → 8² = 64 ✓
- 640×640 → 640/64 = 10 → 10² = 100 ✓
- 1024×1024 → 1024/64 = 16 → 16² = 256 ✓
- 1280×1280 → 1280/64 = 20 → 20² = 400 ✓

## Additional Notes

This fix is **defensive and informative** rather than attempting to automatically correct invalid inputs. This design choice ensures:

1. Users are aware of configuration constraints
2. Invalid images are explicitly rejected rather than silently producing incorrect results
3. Debugging is straightforward with clear error messages
4. The model behavior is predictable and documented

## Related Files

- Issue: GitHub Issue #93
- Model: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`
- Configuration: `config.py`
- Main entry point: `run_dpsk_ocr_image.py`
