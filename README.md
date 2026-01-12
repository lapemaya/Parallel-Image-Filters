# Parallel Image Filters

A high-performance image convolution library implementing various parallelization strategies for 2D convolution operations on RGB images. The project demonstrates the performance differences between sequential, CPU-parallel (Python/joblib), and GPU-parallel (CUDA) implementations.

## Overview

This project implements optimized 2D image convolution with multiple parallelization approaches:

- **Sequential Implementation** (Python): Vectorized convolution using NumPy with sliding window views
- **CPU Parallel Implementation** (Python): Block-level and channel-level parallelization using joblib with shared memory
- **GPU Implementation** (CUDA): Highly optimized CUDA kernels with shared memory, constant memory, and template-based unrolling

## Features

- Multiple convolution kernel presets (Gaussian blur, sharpen, edge detection, emboss)
- Support for custom kernel sizes (up to 7x7 for CUDA)
- Zero-padding boundary handling
- Optimized memory access patterns
- Performance profiling capabilities

## Project Structure

```
.
├── ConvSeq.py                  # Sequential vectorized implementation
├── ConvParallelAdvanced.py     # Advanced parallel CPU implementation
├── CudaConv.cu                 # CUDA GPU-accelerated implementation
├── CMakeLists.txt              # Build configuration for CUDA
├── place.png, bear.png, banana.png  # Test images
└── README.md
```

## Implementations

### 1. Sequential Implementation (ConvSeq.py)

Uses NumPy's `sliding_window_view` and `einsum` for vectorized convolution operations.

**Key optimizations:**
- Pre-flipped kernel to avoid repeated operations
- Float32 precision for memory efficiency
- C-contiguous memory layout
- Single clipping operation at the end

**Usage:**
```python
from ConvSeq import apply_convolution, KERNEL_GAUSSIAN
from PIL import Image
import numpy as np

img = Image.open("place.png").convert("RGB")
arr = np.array(img)
result = apply_convolution(arr, KERNEL_GAUSSIAN, normalize=True)
```

### 2. CPU Parallel Implementation (ConvParallelAdvanced.py)

Parallelizes both image blocks and channels using joblib with multiprocessing.

**Key optimizations:**
- Shared memory to avoid pickling overhead
- Block-based decomposition for cache efficiency
- Parallel processing across channels and spatial regions
- Automatic block size calculation based on available cores

**Usage:**
```python
from ConvParallelAdvanced import apply_convolution, KERNEL_GAUSSIAN_7x7
from PIL import Image
import numpy as np

img = Image.open("place.png").convert("RGB")
arr = np.array(img)
result = apply_convolution(arr, KERNEL_GAUSSIAN_7x7, normalize=True, n_jobs=-1)
```

### 3. CUDA GPU Implementation (CudaConv.cu)

Highly optimized CUDA implementation with multiple performance features.

**Key optimizations:**
- **Pinned memory**: ~2x faster host-device transfers
- **Constant memory**: Fast kernel coefficient access with dedicated cache
- **Shared memory**: Cooperative tile loading with halo to reduce global memory accesses
- **Template specialization**: Compile-time kernel sizes (K=5, K=7) for loop unrolling
- **Buffer reuse**: Single GPU allocation for all 3 RGB channels
- **Optimal block configuration**: Configurable tile size for performance tuning

**Architecture:**
- `conv2d_kernel_optimized_K<K>`: Template version with compile-time unrolling
- `conv2d_kernel_optimized`: Runtime kernel size version
- `convolveCUDA_RGB_Optimized`: Host function managing memory and kernel launches

**Usage:**
```bash
# Build
mkdir build && cd build
cmake ..
make

# Run
./CudaConv [image_path]
./CudaConv place.png
```

**Configuration parameters (in CudaConv.cu:443-473):**
- `K`: Kernel size (5 or 7 for optimized template path)
- `tileSize`: Output tile size per block (default: 16)
- `MAX_KERNEL_SIZE`: Maximum supported kernel dimension

## Requirements

### Python Implementations
```bash
pip install numpy pillow joblib
```

### CUDA Implementation
- CUDA Toolkit (compute capability 6.0+)
- CMake 3.20+
- OpenCV 4.x
- C++20 compatible compiler

## Performance Comparison

Expected performance characteristics for large images (e.g., 2000x2000 pixels):

| Implementation | Relative Speed | Best Use Case |
|----------------|---------------|---------------|
| Sequential     | 1x (baseline) | Small images, prototyping |
| CPU Parallel   | 2-4x          | Medium images, multi-core CPUs |
| CUDA GPU       | 10-50x        | Large images, batch processing |

Note: Actual speedup depends on hardware, image size, and kernel dimensions.

## Building and Running

### Python Scripts
```bash
# Sequential
python ConvSeq.py

# Parallel CPU
python ConvParallelAdvanced.py
```

### CUDA Program
```bash
mkdir build
cd build
cmake ..
make
./CudaConv ../place.png
```

## Kernel Presets

Available in all implementations:
- `KERNEL_BLUR`: 3x3 box blur
- `KERNEL_SHARPEN`: 3x3 sharpening filter
- `KERNEL_EDGE`: 3x3 edge detection (Laplacian)
- `KERNEL_EMBOSS`: 3x3 embossing effect
- `KERNEL_GAUSSIAN`: 3x3 Gaussian blur
- `KERNEL_GAUSSIAN_5x5`: 5x5 Gaussian blur
- `KERNEL_GAUSSIAN_7x7`: 7x7 Gaussian blur

## Implementation Details

### Memory Layout
All implementations use **planar RGB format** internally:
- Sequential/Parallel: NumPy arrays with shape `(H, W, 3)`
- CUDA: Separate buffers for R, G, B channels processed sequentially

### Boundary Handling
All implementations use **zero-padding** for pixels outside image boundaries.

### Convolution Operation
Standard 2D discrete convolution with kernel flipping:
```
out[i,j] = Σ Σ img[i+m, j+n] * kernel[M-m, N-n]
         m n
```

## Code References

Key functions by file:
- ConvSeq.py:34 - `convolve_channel_vectorized`: Vectorized sliding window convolution
- ConvParallelAdvanced.py:39 - `process_block_channel_shm`: Shared memory block processing
- CudaConv.cu:91 - `conv2d_kernel_optimized_K`: Template CUDA kernel
- CudaConv.cu:298 - `convolveCUDA_RGB_Optimized`: Main GPU convolution function

## License

This project is provided as-is for educational and research purposes.
