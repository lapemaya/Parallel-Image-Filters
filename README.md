# Parallel Image Filters

An educational image convolution library demonstrating naive and optimized implementations for 2D convolution operations on RGB images. The project compares sequential and parallel approaches in Python alongside a highly-optimized GPU implementation in CUDA.

This work presents a comparative analysis showing speedup from ~5× for naive CPU parallelization to over 370× for GPU acceleration on large images.

## Overview

This project implements 2D image convolution with the following approaches:

- **Sequential Naive Implementation** (Python): Simple triple-loop baseline for educational purposes
- **CPU Parallel Naive Implementation** (Python): Block-based parallelization using joblib/multiprocessing with naive convolution
- **GPU Implementation** (CUDA): Massively parallel SIMT execution with shared memory tiling, constant memory, and template specialization

## Features

- Naive convolution with explicit loops (educational)
- Simple block-based parallelization for CPU
- Multiple convolution kernel presets (Gaussian blur, sharpen, edge detection, emboss)
- Support for custom kernel sizes (up to 7x7 for CUDA)
- Zero-padding boundary handling
- Highly optimized GPU implementation with shared memory and constant memory
- Performance profiling capabilities

## Project Structure

```
.
├── ConvolutionScripts/
│   ├── ConvSeqDumb.py              # Sequential naive implementation (loops)
│   ├── ConvParallelAdvancedDumb.py # Parallel CPU with naive convolution
│   └── CudaConv.cu                 # CUDA GPU-accelerated implementation
├── AnalyseResults/
│   ├── benchmark_results.json      # Raw benchmark data
│   ├── speedup_results.json        # Computed speedup metrics
│   ├── compute_speedup.py          # Speedup computation script
│   ├── report.tex                  # Detailed technical report
│   └── *.py                        # Table generation scripts
├── CMakeLists.txt                  # Build configuration for CUDA
├── benchmark.sh                    # Automated benchmark script
├── place.png, bear.png, banana.png # Test images
└── README.md
```

## Implementations

### 1. Sequential Naive Implementation (ConvSeqDumb.py)

Simple triple-loop implementation over image rows, columns and kernel entries. Good for understanding convolution fundamentals; not optimized for performance.

**Key characteristics:**
- Explicit Python loops for clarity
- Direct pixel-by-pixel computation
- Minimal memory optimization
- Educational reference implementation

**Usage:**
```bash
python ConvolutionScripts/ConvSeqDumb.py --input place.png --kernel gaussian
```

### 2. CPU Parallel Naive Implementation (ConvParallelAdvancedDumb.py)

Divides the image into blocks and parallelizes across processes using multiprocessing/joblib. Each worker runs a naive convolution on its assigned block.

**Key characteristics:**
- Block-based spatial decomposition
- Shared memory to reduce data duplication
- Parallel processing across channels and spatial regions
- Naive convolution within each block (explicit loops)
- Automatic block size calculation based on available cores

**Parallelization strategy:**
- Simultaneous parallelism on both color channels (R, G, B) and spatial blocks
- Each process handles one block of one channel independently
- Load balancing across available cores

**Usage:**
```bash
python ConvolutionScripts/ConvParallelAdvancedDumb.py --input place.png --kernel gaussian --n_jobs 4
```

### 3. CUDA GPU Implementation (CudaConv.cu)

Highly optimized CUDA implementation with multiple performance features.

**Key optimizations:**
- **Pinned memory**: ~2x faster host-device transfers
- **Constant memory**: Fast kernel coefficient access with dedicated cache
- **Shared memory**: Cooperative tile loading with halo to reduce global memory accesses
- **Template specialization**: Compile-time kernel sizes (K=3, K=5, K=7) for loop unrolling
- **Buffer reuse**: Single GPU allocation for all 3 RGB channels
- **Optimal block configuration**: Configurable tile size for performance tuning (8×8 to 24×24)

**Architecture:**
- `conv2d_kernel_optimized_K<K>`: Template version with compile-time unrolling
- `conv2d_kernel_optimized`: Runtime kernel size version
- `convolveCUDA_RGB_Optimized`: Host function managing memory and kernel launches
- Spatial tiling with halo regions for independent block computation

**Processing workflow:**
- RGB channels processed separately but sequentially on GPU
- Each thread computes one output pixel
- Cooperative loading of input tiles into shared memory
- Synchronization before computation to ensure data availability

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

**Configuration parameters (in CudaConv.cu):**
- `K`: Kernel size (3, 5, or 7 for optimized template path)
- `tileSize`: Output tile size per block (default: 16, recommended: 8-24)
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

### Hardware Setup
- **CPU**: 13th Gen Intel Core i9-13900H (14 cores / 20 threads, 4.7 GHz base, 5.3 GHz boost)
- **RAM**: 32 GB DDR5-6000
- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM, CUDA 13.0)
- **OS**: Linux
- **Python**: 3.13 with NumPy 1.24.3 (Intel MKL backend)

### Benchmark Results

Performance measured on various image sizes (800×800 to 8000×8000 pixels) with different kernel sizes (3×3, 5×5, 7×7):

#### Sequential Implementation - Absolute Times
| Image Size | K=3×3 | K=5×5 | K=7×7 |
|------------|-------|-------|-------|
| 800×800 | 3.82s | 3.94s | 3.83s |
| 1600×1600 | 15.26s | 15.26s | 15.86s |
| 3200×3200 | 59.48s | 60.80s | 60.00s |
| 6400×6400 | 234.06s | 238.84s | 244.38s |
| 8000×8000 | 369.10s | 370.77s | 373.99s |

#### CPU Parallel - Best Speedup (vs Sequential)
| Image Size | K=3×3 (20 jobs) | K=5×5 (20 jobs) | K=7×7 (20 jobs) |
|------------|-----------------|-----------------|-----------------|
| 800×800 | **6.09×** | **6.05×** | **5.69×** |
| 1600×1600 | **4.99×** | **4.89×** | **5.00×** |
| 3200×3200 | **4.84×** | **4.89×** | **4.80×** |
| 6400×6400 | **4.78×** | **4.74×** | **4.83×** |
| 8000×8000 | **4.83×** | **4.81×** | **4.77×** |

**Key findings for CPU Parallel:**
- Maximum speedup: ~6× with 20 processes
- Efficiency: 24-72% (decreases with more processes due to memory bandwidth saturation)
- Best for images 800×800 to 3200×3200
- Speedup plateaus beyond 12 processes

#### CUDA GPU - Best Speedup (vs Sequential)
| Image Size | K=3×3 (tile=8/16) | K=5×5 (tile=8) | K=7×7 (tile=24) |
|------------|-------------------|----------------|-----------------|
| 800×800 | **424.9×** (9ms) | **438.4×** (9ms) | **425.7×** (9ms) |
| 1600×1600 | **390.9×** (39ms) | **390.9×** (39ms) | **396.4×** (40ms) |
| 3200×3200 | **373.1×** (160ms) | **377.7×** (161ms) | **370.3×** (162ms) |
| 6400×6400 | **363.4×** (644ms) | **369.6×** (646ms) | **376.6×** (649ms) |
| 8000×8000 | **367.8×** (1004ms) | **369.1×** (1005ms) | **370.0×** (1011ms) |

**Key findings for CUDA:**
- Consistent speedup: **350-440× across all configurations**
- Time per MAC: ~0.1-0.6 ns (vs 7-44 ns for CPU parallel)
- GPU is **70-80× more efficient per operation** than CPU
- Tile sizes 8×8 to 24×24 perform similarly; 32×32 shows slight degradation
- Optimal for all image sizes, especially 3200×3200 and larger

### Computational Complexity Analysis

For an RGB image of size H×W with kernel K×K:
- **Complexity**: O(3 × H × W × K²) operations
- **Per pixel**: ~2K² FLOPS (K² multiplications + K² additions)
- **Total FLOPS**: ~3 × H × W × K² × 2
- **Example** (8000×8000, K=7): ~18.6 billion FLOPS

### Performance Summary

| Implementation | Speedup Range | Time per MAC | Best Use Case |
|----------------|---------------|--------------|---------------|
| Sequential     | 1× (baseline) | 36-44 ns (K=3) | Prototyping, small images |
| CPU Parallel   | 2.5-6.1× | 7-44 ns | Medium images without GPU |
| CUDA GPU       | 336-439× | 0.1-0.6 ns | All production workloads |

**Memory-bound regime**: All implementations are limited by memory bandwidth rather than compute capacity for K≤7. Larger kernels would shift toward compute-bound behavior where GPU advantage increases further.

### Parallel Efficiency (CPU)
Efficiency = Speedup / Number of Processes

| Image Size | 4 jobs | 8 jobs | 12 jobs | 16 jobs | 20 jobs |
|------------|--------|--------|---------|---------|---------|
| 800×800 | 71.6% | 56.5% | 47.5% | 36.0% | 30.4% |
| 8000×8000 | 63.8% | 46.1% | 37.5% | 28.9% | 24.2% |

The sub-linear scaling indicates overhead from process management, memory contention, and GIL limitations despite using multiprocessing.

### Global Implementation Comparison

Direct comparison of absolute execution times for key configurations:

| Configuration | Sequential | CPU Parallel (best) | CUDA (best) | CPU vs Seq | GPU vs Seq |
|--------------|-----------|---------------------|-------------|------------|------------|
| 800×800, K=3 | 3.82s | 0.63s | 0.009s | 6.09× | 424.9× |
| 3200×3200, K=5 | 60.80s | 12.43s | 0.161s | 4.89× | 377.7× |
| 8000×8000, K=7 | 373.99s | 78.41s | 1.011s | 4.77× | 370.0× |

### Time per MAC (Multiply-Accumulate) Analysis

Normalized performance metric showing efficiency per operation:

**CPU Parallel (best n_jobs):**
| Image Size | K=3 | K=5 | K=7 |
|------------|-----|-----|-----|
| 800×800 | 36.35 ns | 13.58 ns | 7.16 ns |
| 8000×8000 | 44.24 ns | 16.06 ns | 8.33 ns |

**CUDA GPU (best tile size):**
| Image Size | K=3 | K=5 | K=7 |
|------------|-----|-----|-----|
| 800×800 | 0.521 ns | 0.188 ns | 0.096 ns |
| 8000×8000 | 0.581 ns | 0.209 ns | 0.108 ns |

**Key insights:**
- GPU is **70-80× more efficient per operation** than CPU parallel
- Time per MAC decreases with kernel size (overhead amortization)
- Performance stabilizes for larger images (compute-bound regime)
- All implementations show memory-bound behavior for K≤7

## Building and Running

### Python Scripts
```bash
# Sequential (naive)
python ConvolutionScripts/ConvSeqDumb.py --input place.png --kernel gaussian

# Parallel CPU (naive)
python ConvolutionScripts/ConvParallelAdvancedDumb.py --input place.png --kernel gaussian --n_jobs 4
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
out[i,j] = Σ Σ img[i+m, j+n] * kernel[k_h-m-1, k_w-n-1]
         m n
```
where m ∈ [0, k_h-1], n ∈ [0, k_w-1]

## Code References

Key functions by file:
- `ConvolutionScripts/ConvSeqDumb.py` - `convolve_channel`: Naive triple-loop convolution
- `ConvolutionScripts/ConvParallelAdvancedDumb.py` - `process_block_channel_shm`: Shared memory block processing with naive convolution
- `ConvolutionScripts/CudaConv.cu` - `conv2d_kernel_optimized_K`: Template CUDA kernel
- `ConvolutionScripts/CudaConv.cu` - `convolveCUDA_RGB_Optimized`: Main GPU convolution function

## License

This project is provided as-is for educational and research purposes.

## Documentation

For a detailed technical analysis including:
- Complete mathematical formulation of the convolution operation
- In-depth discussion of optimization strategies
- Comprehensive performance analysis with statistical metrics
- Architectural considerations for CPU and GPU implementations

Please refer to `AnalyseResults/report.tex` (LaTeX source) which contains the full academic report of this work.

## Notes

- The benchmark results in this README are extracted from the comprehensive study documented in the report
- All speedup measurements use the sequential naive implementation as baseline
- CPU parallel efficiency accounts for the sub-linear scaling typical of memory-bound workloads
- CUDA measurements include host-device transfer times using pinned memory
- The naive ("Dumb") Python implementations serve as educational references showing the fundamental algorithms

