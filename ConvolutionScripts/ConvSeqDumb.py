# ==== IMPORTS ====
import numpy as np
from PIL import Image
import cProfile
import pstats
import io
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
def convolve_channel_vectorized(channel, kernel_flipped):

    # ==== SETUP PARAMETERS ====
    kh, kw = kernel_flipped.shape
    ch, cw = channel.shape
    out_h = ch - kh + 1
    out_w = cw - kw + 1
    result = np.zeros((out_h, out_w), dtype=np.float32)
    
    # ==== NESTED LOOPS CONVOLUTION ====
    for i in range(out_h):
        for j in range(out_w):
            window = channel[i:i+kh, j:j+kw]
            result[i, j] = np.sum(window * kernel_flipped)
    
    return result

# ============================================
# ========= SETUP CONVOLUTION ================
# ============================================
def apply_convolution(img_arr, kernel, normalize=False):

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
    out = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # ==== CONVOLVE EACH CHANNEL ====
    for c in range(3):
        out[:, :, c] = convolve_channel_vectorized(img_float[:, :, c], kernel_flipped)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def apply_convolution_timed(img_arr, kernel, normalize=False, n_jobs=1):
    # ==== TIME THE CONVOLUTION ====
    # n_jobs parameter is ignored for sequential implementation
    t0 = time.perf_counter()
    out = apply_convolution(img_arr, kernel, normalize=normalize)
    t1 = time.perf_counter()
    return out, (t1 - t0)


# ============================================
# ================ MAIN ======================
# ============================================
if __name__ == "__main__":

    # ==== PARAMETERS ====
    input_path = "place.png"
    output_path = "output_sequential.jpg"
    kernel = KERNEL_GAUSSIAN_7x7
    normalize = True

    # ===== LOAD IMAGE =====
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)

    # ===== PROFILE SETUP =====
    profiler = cProfile.Profile()
    profiler.enable()

    # ===== RUN CONVOLUTION =====
    result = apply_convolution(arr, kernel, normalize=normalize)
    profiler.disable()

    # ===== PRINT RESULTS PROFILER =====
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(20)
    print("\n=== Profiling Results ===")
    print(s.getvalue())

