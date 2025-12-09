#!/usr/bin/env python3
"""
Simple image convolution script.
"""
import numpy as np
from PIL import Image

# Kernel presets
KERNEL_BLUR = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float)
KERNEL_SHARPEN = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float)
KERNEL_EDGE = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
KERNEL_EMBOSS = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float)
KERNEL_GAUSSIAN = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float)

def convolve_channel(channel, kernel_flipped,kh,kw,out_h,out_w):
    out = np.zeros((out_h, out_w), dtype=float)
    # convolution without padding, stride=1
    for i in range(out_h):
        for j in range(out_w):
            region = channel[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * kernel_flipped)  #conv effetiva
    return out

def apply_convolution(img_arr, kernel, normalize=False):
    # Normalize kernel if requested
    if normalize:
        s = kernel.sum()
        if s != 0:
            kernel = kernel / s

    # Pre-flip kernel once
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Convert to float once for all channels
    img_float = img_arr.astype(float)

    # Calculate output dimensions
    kh, kw = kernel.shape
    h, w = img_arr.shape[0], img_arr.shape[1]
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Pre-allocate output array
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Process each channel
    for c in range(3):
        convc = convolve_channel(img_float[:,:,c], kernel_flipped,kh,kw,out_h,out_w)
        out[:,:,c] = np.clip(convc, 0, 255).astype(np.uint8)

    return out

if __name__ == "__main__":
    # Configuration
    input_path = "place.png"
    output_path = "output.jpg"
    kernel = KERNEL_EDGE
    normalize = True

    # Load RGB image
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)

    # Apply convolution
    result = apply_convolution(arr, kernel, normalize=normalize)

    # Save result
    out_img = Image.fromarray(result)
    out_img.save(output_path)
    print(f"Saved: {output_path}")

