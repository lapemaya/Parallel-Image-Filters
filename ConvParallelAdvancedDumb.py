# ==== IMPORTS ====
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import shared_memory
import cProfile
import pstats
import time

# ===== KERNEL PRESETS =====
KERNEL_BLUR = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float)
KERNEL_SHARPEN = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float)
KERNEL_EDGE = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
KERNEL_EMBOSS = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float)
KERNEL_GAUSSIAN = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float)
KERNEL_GAUSSIAN_5x5 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=float)

KERNEL_GAUSSIAN_7x7 = np.array([
    [1,  6,  15,  20,  15,  6, 1],
    [6,  36,  90, 120,  90, 36, 6],
    [15, 90, 225, 300, 225, 90, 15],
    [20,120, 300, 400, 300,120, 20],
    [15, 90, 225, 300, 225, 90, 15],
    [6,  36,  90, 120,  90, 36, 6],
    [1,  6,  15,  20,  15,  6, 1]
], dtype=float)


# ============================================
# ======== CONVOLUTION FUNCTION ==============
# ============================================
def process_block_channel_shm(shm_input_name, shm_output_name, img_shape, out_shape, dtype_str,
                               kernel_flipped, channel_idx, start_i, end_i, start_j, end_j, kh, kw):
    
    # ==== ACCESS SHARED MEMORY ====
    shm_input = shared_memory.SharedMemory(name=shm_input_name)
    shm_output = shared_memory.SharedMemory(name=shm_output_name)
    
    # ==== CREATE NUMPY ARRAYS FROM SHARED MEMORY ====
    img_arr = np.ndarray(img_shape, dtype=dtype_str, buffer=shm_input.buf)
    out_arr = np.ndarray(out_shape, dtype=dtype_str, buffer=shm_output.buf)

    # ==== EXTRACT CHANNEL AND REGION PART ====
    channel = img_arr[:, :, channel_idx]
    region_start_i = start_i
    region_end_i = end_i + kh - 1
    region_start_j = start_j
    region_end_j = end_j + kw - 1

    # ==== EXTRACT REGION WITH KERNEL OVERLAP ====
    region = channel[region_start_i:region_end_i, region_start_j:region_end_j]
    out_h = end_i - start_i
    out_w = end_j - start_j
    block_result = np.zeros((out_h, out_w), dtype=np.float32)
    
    # ==== NESTED LOOPS CONVOLUTION FOR BLOCK ====
    for i in range(out_h):
        for j in range(out_w):
            window = region[i:i+kh, j:j+kw]
            block_result[i, j] = np.sum(window * kernel_flipped)

    # ==== WRITE BLOCK RESULT TO OUTPUT SHARED MEMORY ====
    out_arr[start_i:end_i, start_j:end_j, channel_idx] = block_result
    shm_input.close()
    shm_output.close()

    return True



# ============================================
# ========= SETUP CONVOLUTION ================
# ============================================
def apply_convolution(img_arr, kernel, normalize=False, n_jobs=-1, block_size=None):
    # ==== SETUP =====
    if normalize:
        s = kernel.sum()
        if s != 0:
            kernel = kernel / s
    kernel_flipped = np.flipud(np.fliplr(kernel)).astype(np.float32)
    img_float = np.ascontiguousarray(img_arr.astype(np.float32))


    # ==== PREPARE OUTPUT ====
    kh, kw = kernel.shape
    h, w = img_arr.shape[0], img_arr.shape[1]
    out_h = h - kh + 1
    out_w = w - kw + 1

    # ==== DETERMINE BLOCK SIZE ====
    if block_size is None:
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        total_blocks_per_channel = max(1, n_cores*1 // 3)  
        block_size = max(9, int(np.sqrt(out_h * out_w / total_blocks_per_channel)))

    # ==== CREATE BLOCKS ====
    blocks = []
    for i in range(0, out_h, block_size):
        for j in range(0, out_w, block_size):
            end_i = min(i + block_size, out_h)
            end_j = min(j + block_size, out_w)
            blocks.append((i, end_i, j, end_j))

    # ==== SETUP SHARED MEMORY ====
    shm_input = shared_memory.SharedMemory(create=True, size=img_float.nbytes)
    input_arr = np.ndarray(img_float.shape, dtype=img_float.dtype, buffer=shm_input.buf)

    # ==== COPY INPUT DATA TO SHM ====
    np.copyto(input_arr, img_float)

    # ==== CREATE SHM FOR OUTPUT ====
    out_shape = (out_h, out_w, 3)
    out_nbytes = out_h * out_w * 3 * np.dtype(np.float32).itemsize
    shm_output = shared_memory.SharedMemory(create=True, size=out_nbytes)
    output_arr = np.ndarray(out_shape, dtype=np.float32, buffer=shm_output.buf)
    output_arr[:] = 0

    # ==== PROCESS BLOCKS IN PARALLEL ====
    try:
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(process_block_channel_shm)(
                shm_input.name, shm_output.name, img_float.shape, out_shape,
                img_float.dtype.str, kernel_flipped, c, start_i, end_i, start_j, end_j, kh, kw
            )
            # ==== LOOP OVER CHANNELS AND BLOCKS ====
            for c in range(3)
            for start_i, end_i, start_j, end_j in blocks
        )
        # ==== CLIP AND CONVERT TO UINT8 ====
        out_uint8 = np.clip(output_arr, 0, 255).astype(np.uint8)

    # ==== CLEANUP SHARED MEMORY ====
    finally:
        shm_input.close()
        shm_input.unlink()
        shm_output.close()
        shm_output.unlink()

    return out_uint8


def apply_convolution_timed(img_arr, kernel, normalize=False, n_jobs=-1, block_size=None):
    t0 = time.perf_counter()
    out = apply_convolution(img_arr, kernel, normalize=normalize, n_jobs=n_jobs, block_size=block_size)
    t1 = time.perf_counter()
    return out, (t1 - t0)


# ============================================
# ================= MAIN =====================
# ============================================
if __name__ == "__main__":
    with cProfile.Profile() as pr:

        # ==== PARAMETERS ====
        input_path = "/home/lapemaya/PycharmProjects/Parallel-Image-Filters/place.png"
        output_path = "output_parallel_advanced.png"
        kernel = KERNEL_GAUSSIAN_7x7
        normalize = True
        n_jobs = -1 
        block_size = None  

        # ===== LOAD IMAGE =====
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img)

        # ==== APPLY CONVOLUTION =====
        result = apply_convolution(arr, kernel, normalize=normalize, n_jobs=n_jobs, block_size=block_size)

    # ===== PROFILE STATS =====
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime').print_stats(10)
