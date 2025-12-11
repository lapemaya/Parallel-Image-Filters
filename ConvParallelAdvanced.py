#!/usr/bin/env python3
"""
Advanced parallel image convolution script using joblib.
Parallelizes both image blocks AND channels simultaneously.
Uses shared memory to avoid pickling overhead.
"""
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import shared_memory
import cProfile
import pstats

# Kernel presets
KERNEL_BLUR = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float)
KERNEL_SHARPEN = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float)
KERNEL_EDGE = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
KERNEL_EMBOSS = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float)
KERNEL_GAUSSIAN = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float)

def process_block_channel_shm(shm_name, img_shape, dtype_str, kernel_flipped, start_i, end_i, start_j, end_j, channel_idx):
    """Process a single block for a single channel using shared memory."""
    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    # Reconstruct the numpy array view from shared memory
    img_float = np.ndarray(img_shape, dtype=dtype_str, buffer=shm.buf)

    # Get the channel
    channel = img_float[:, :, channel_idx]

    # Calculate kernel dimensions from the kernel itself
    kh, kw = kernel_flipped.shape

    # Extract region with kernel overlap
    region_start_i = start_i
    region_end_i = end_i + kh - 1
    region_start_j = start_j
    region_end_j = end_j + kw - 1

    region = channel[region_start_i:region_end_i, region_start_j:region_end_j]

    # Calculate output dimensions for this block
    out_h = end_i - start_i
    out_w = end_j - start_j

    out = np.zeros((out_h, out_w), dtype=np.float32)

    # Convolve this block
    for i in range(out_h):
        for j in range(out_w):
            window = region[i:i+kh, j:j+kw]
            out[i,j] = np.sum(window * kernel_flipped)

    # Clean up shared memory reference (don't unlink, just close)
    shm.close()

    return channel_idx, start_i, end_i, start_j, end_j, out

def apply_convolution(img_arr, kernel, normalize=False, n_jobs=-1, block_size=None):
    # Normalize kernel if requested
    if normalize:
        s = kernel.sum()
        if s != 0:
            kernel = kernel / s

    # Pre-flip kernel once
    kernel_flipped = np.flipud(np.fliplr(kernel)).astype(np.float32)

    # Convert to float32 (half memory vs float64) and make C-contiguous
    img_float = np.ascontiguousarray(img_arr.astype(np.float32))

    # Calculate output dimensions
    kh, kw = kernel.shape

    h, w = img_arr.shape[0], img_arr.shape[1]
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Determine block size (if None, use automatic calculation)
    if block_size is None:
        # Default: divide image into blocks based on available cores
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        # Aim for ~2 blocks per core (since we also parallelize over channels)
        total_blocks_per_channel = max(1, n_cores // 3)  # 3 channels
        total_blocks = total_blocks_per_channel * 3
        block_size = max(64, int(np.sqrt(out_h * out_w / total_blocks_per_channel)))

    # Create blocks coordinates
    blocks = []
    for i in range(0, out_h, block_size):
        for j in range(0, out_w, block_size):
            end_i = min(i + block_size, out_h)
            end_j = min(j + block_size, out_w)
            blocks.append((i, end_i, j, end_j))

    # Create shared memory for the input image
    shm = shared_memory.SharedMemory(create=True, size=img_float.nbytes)

    # Copy image data to shared memory
    shm_array = np.ndarray(img_float.shape, dtype=img_float.dtype, buffer=shm.buf)
    np.copyto(shm_array, img_float)

    try:
        # Pre-allocate output array
        out = np.zeros((out_h, out_w, 3), dtype=np.float32)

        # Create all tasks (blocks × channels)
        tasks = []
        for c in range(3):
            for start_i, end_i, start_j, end_j in blocks:
                tasks.append((shm.name, img_float.shape, img_float.dtype.str, kernel_flipped,
                             start_i, end_i, start_j, end_j, c))

        # Process all blocks for all channels in parallel
        # Use prefer="processes" to get true parallelism with shared memory (no GIL)
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(process_block_channel_shm)(
                shm_name, shape, dtype_str, kernel_flipped, start_i, end_i, start_j, end_j, c
            ) for shm_name, shape, dtype_str, kernel_flipped, start_i, end_i, start_j, end_j, c in tasks
        )

        # Assemble results into output array
        for channel_idx, start_i, end_i, start_j, end_j, block_result in results:
            out[start_i:end_i, start_j:end_j, channel_idx] = block_result

        # Clip and convert to uint8 at the end
        out_uint8 = np.clip(out, 0, 255).astype(np.uint8)

    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()

    return out_uint8

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        # Configuration
        input_path = "place.png"
        output_path = "output_parallel_advanced.png"
        kernel = KERNEL_BLUR
        normalize = True
        n_jobs = -1  # -1 uses all available cores
        block_size = None  # None = automatic, or set manually (e.g., 128, 256)

        # Load RGB image
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img)

        print(f"Processing image: {arr.shape[0]}x{arr.shape[1]} pixels")
        print(f"Parallelization: blocks × channels (fully parallel)")


        # Apply convolution in parallel (blocks × channels simultaneously)
        result = apply_convolution(arr, kernel, normalize=normalize, n_jobs=n_jobs, block_size=block_size)

        # Save result
        out_img = Image.fromarray(result)
        out_img.save(output_path)
        print(f"Saved: {output_path}")
    # Print profiling results
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)


