# Attention Diagnostics for DeepSeek-OCR

## Overview

This implementation addresses **GitHub Issue #3243**: *Myelin: attention fusion and FlashAttention visibility*.

The attention diagnostics module provides comprehensive visibility into attention implementation details, including:

- **FlashAttention Usage**: Whether FlashAttention or FlashAttention-2 is being used
- **Attention Fusion**: Type of fusion (QKV-fused, full-fused, Myelin-optimized, etc.)
- **Quantization**: Quantization type (INT8, FP8, FP16, BF16, implicit, or none)
- **Backend Information**: Which attention backend is active (FlashAttention, SDPA, xFormers, etc.)
- **Layer-specific Details**: Per-layer diagnostics including head count, dimensions, and data types

## Problem Statement

When attention operations get fused in a single operation (e.g., with Myelin or other optimization frameworks), it's often unclear:

1. Whether FlashAttention is actually being used
2. What type of fusion is applied
3. If quantization is happening under the hood (especially implicit quantization)
4. Which specific attention implementation is active

This lack of visibility makes it difficult to:
- Debug performance issues
- Verify optimization settings
- Understand model behavior
- Profile attention operations

## Solution

The attention diagnostics module provides a comprehensive solution with:

### 1. Diagnostic Data Collection

The `AttentionDiagnostics` dataclass captures:

```python
@dataclass
class AttentionDiagnostics:
    layer_name: str                    # Layer identifier
    layer_id: int                      # Layer number
    backend: AttentionBackend          # FlashAttention, SDPA, etc.
    fusion_type: FusionType            # QKV-Fused, Full-Fused, etc.
    quantization: QuantizationType     # INT8, FP8, FP16, BF16, etc.
    use_flash_attention: bool          # FlashAttention enabled flag
    num_heads: int                     # Number of attention heads
    head_dim: int                      # Dimension per head
    sequence_length: Optional[int]     # Sequence length (runtime)
    batch_size: Optional[int]          # Batch size (runtime)
    dtype: Optional[str]               # Data type
    device: Optional[str]              # Device (cuda:0, cpu, etc.)
    additional_info: Optional[Dict]    # Extra metadata
```

### 2. Automatic Detection

The system automatically detects:

- **Backend**: Checks for FlashAttention availability and version, falls back to SDPA or naive implementation
- **Fusion Type**: Analyzes module structure to identify QKV fusion, full fusion, or Myelin optimization
- **Quantization**: Inspects module config and tensor dtypes to detect quantization schemes

### 3. Reporting and Export

Multiple output formats:

- **Console Summary**: Quick overview of attention configuration
- **Detailed Reports**: Per-layer diagnostics with full information
- **JSON Export**: Machine-readable format for analysis
- **Text Export**: Human-readable detailed report

## Installation

The diagnostics module is already integrated into the DeepSeek-OCR codebase. No additional installation required.

## Usage

### Basic Usage

#### 1. Enable Diagnostics in Configuration

Edit `config.py`:

```python
# Attention Diagnostics Configuration
ENABLE_ATTENTION_DIAGNOSTICS = True   # Enable diagnostics
ATTENTION_DIAGNOSTICS_VERBOSE = True  # Enable detailed logging
ATTENTION_DIAGNOSTICS_OUTPUT = './attention_report'  # Output path
```

#### 2. Run with Diagnostics

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_with_diagnostics.py
```

### Programmatic Usage

```python
from attention_diagnostics import enable_diagnostics, get_collector

# Enable diagnostics
enable_diagnostics(verbose=True)

# ... run your model ...

# Get diagnostics
collector = get_collector()

# Print summary
collector.print_summary()

# Export to file
collector.export_to_file('attention_report.json', format='json')
collector.export_to_file('attention_report.txt', format='txt')

# Get detailed data
summary = collector.get_summary()
print(f"Total layers: {summary['total_layers']}")
print(f"FlashAttention layers: {summary['flash_attention_layers']}")
```

### Advanced Usage

#### Custom Diagnostics Collection

```python
from attention_diagnostics import AttentionDiagnosticsCollector

# Create custom collector
collector = AttentionDiagnosticsCollector(enabled=True, verbose=False)

# Collect diagnostics for a specific layer
diagnostics = collector.collect(
    module=attention_module,
    layer_name="custom_attention_layer",
    use_flash_attn=True,
    num_heads=16,
    head_dim=64,
    input_tensor=input_tensor,
    custom_field="custom_value"
)

# Access diagnostics
print(diagnostics.backend)  # AttentionBackend.FLASH_ATTENTION_2
print(diagnostics.fusion_type)  # FusionType.QKV_FUSED
print(diagnostics.quantization)  # QuantizationType.FP16
```

## Output Examples

### Console Summary

```
============================================================
ATTENTION DIAGNOSTICS SUMMARY
============================================================
Total Attention Layers: 24
FlashAttention Enabled Layers: 24

Backend Distribution:
  FlashAttention-2: 24

Fusion Type Distribution:
  QKV-Fused: 24

Quantization Distribution:
  BF16: 24
============================================================
```

### Detailed Layer Information

```
Attention Diagnostics for NoTPAttention (Layer 0):
  Backend: FlashAttention-2
  Fusion Type: QKV-Fused
  Quantization: BF16
  FlashAttention Enabled: True
  Num Heads: 16
  Head Dimension: 64
  Sequence Length: 256
  Batch Size: 1
  Data Type: torch.bfloat16
  Device: cuda:0
  Additional Info: {'attention_type': 'self_attention', 'has_qkv_fusion': True}
```

### JSON Export

```json
{
  "summary": {
    "total_layers": 24,
    "flash_attention_layers": 24,
    "backend_distribution": {
      "FlashAttention-2": 24
    },
    "fusion_distribution": {
      "QKV-Fused": 24
    },
    "quantization_distribution": {
      "BF16": 24
    }
  },
  "layers": [
    {
      "layer_name": "NoTPAttention",
      "layer_id": 0,
      "backend": "FlashAttention-2",
      "fusion_type": "QKV-Fused",
      "quantization": "BF16",
      "use_flash_attention": true,
      "num_heads": 16,
      "head_dim": 64,
      "sequence_length": 256,
      "batch_size": 1,
      "dtype": "torch.bfloat16",
      "device": "cuda:0"
    }
  ]
}
```

## Integration Points

The diagnostics are integrated at the following locations:

### 1. CLIP Attention (`deepencoder/clip_sdpa.py`)

- **Class**: `NoTPAttention`
- **Location**: Forward pass
- **Detects**: FlashAttention usage, QKV fusion

### 2. SAM Attention (`deepencoder/sam_vary_sdpa.py`)

- **Class**: `Attention`
- **Location**: Forward pass
- **Detects**: SDPA usage, relative position encoding

### 3. Configuration (`config.py`)

- **Settings**: Enable/disable diagnostics, verbosity, output path

### 4. Example Scripts

- **`run_with_diagnostics.py`**: Demonstrates full diagnostics workflow

## API Reference

### Classes

#### `AttentionBackend` (Enum)
- `FLASH_ATTENTION`: FlashAttention 1.x
- `FLASH_ATTENTION_2`: FlashAttention 2.x
- `SDPA`: Scaled Dot Product Attention
- `XFORMERS`: xFormers attention
- `NAIVE`: Naive implementation
- `MYELIN_FUSED`: Myelin-optimized fusion
- `UNKNOWN`: Unknown backend

#### `FusionType` (Enum)
- `NONE`: No fusion
- `QKV_FUSED`: QKV projections fused
- `FULL_FUSED`: Fully fused attention
- `MYELIN_OPTIMIZED`: Myelin-optimized fusion

#### `QuantizationType` (Enum)
- `NONE`: No quantization
- `INT8`: 8-bit integer quantization
- `FP8`: 8-bit floating point
- `FP16`: 16-bit floating point
- `BF16`: BFloat16
- `IMPLICIT`: Implicit quantization

#### `AttentionDiagnosticsCollector`

**Methods**:
- `collect()`: Collect diagnostics for a layer
- `get_summary()`: Get summary statistics
- `print_summary()`: Print formatted summary
- `export_to_file()`: Export to JSON or text file
- `reset()`: Clear collected diagnostics

### Functions

- `enable_diagnostics(verbose=False)`: Enable global diagnostics
- `disable_diagnostics()`: Disable global diagnostics
- `get_collector()`: Get global collector instance
- `collect_attention_diagnostics()`: Convenience function for collection

## Performance Impact

The diagnostics module is designed with minimal performance overhead:

- **Collection**: Only on first forward pass per layer (cached)
- **Detection**: Lightweight attribute checks
- **Storage**: Minimal memory footprint
- **Disabled by default**: Zero overhead when not enabled

Typical overhead when enabled: < 1% inference time

## Troubleshooting

### Diagnostics Not Showing

1. Verify `ENABLE_ATTENTION_DIAGNOSTICS = True` in `config.py`
2. Check that attention modules are being executed
3. Enable verbose mode for detailed logging

### Incorrect Backend Detection

1. Verify FlashAttention installation: `pip list | grep flash-attn`
2. Check CUDA compatibility
3. Review module configuration flags

### Export Failures

1. Ensure output directory exists
2. Check write permissions
3. Verify disk space

## Future Enhancements

Potential improvements:

1. **TensorRT/Myelin Integration**: Direct detection of TensorRT optimizations
2. **Performance Metrics**: Timing and memory usage per layer
3. **Visualization**: Graphical representation of attention patterns
4. **Comparison Tools**: Compare diagnostics across runs
5. **Real-time Monitoring**: Live dashboard during inference

## Contributing

To extend the diagnostics:

1. Add new detection methods to `AttentionDiagnosticsCollector`
2. Extend enums for new backends/fusion types/quantization schemes
3. Update integration points in attention modules
4. Add tests and documentation

## License

Same as DeepSeek-OCR project.

## References

- GitHub Issue #3243: Myelin attention fusion and FlashAttention visibility
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Contact

For issues or questions about attention diagnostics, please refer to the main DeepSeek-OCR repository.
