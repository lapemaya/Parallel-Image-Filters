#!/usr/bin/env python3
"""
Benchmark script to compare sequential vs parallel convolution versions.
"""
import numpy as np
from PIL import Image
import time
import multiprocessing

# Import all versions
import ConvSeq
import ConvParallel
import ConvParallelAdvanced

def benchmark_convolution(img_path, kernel, normalize=True, n_runs=3):
    """Run benchmark comparing all versions."""

    # Load image
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    print(f"Image size: {arr.shape[0]}x{arr.shape[1]} pixels")
    print(f"Kernel size: {kernel.shape[0]}x{kernel.shape[1]}")
    print(f"Number of runs: {n_runs}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print("=" * 70)

    results = {}

    # Benchmark sequential version
    print("\n1. SEQUENTIAL VERSION")
    print("-" * 70)
    times_seq = []
    for i in range(n_runs):
        start = time.time()
        result_seq = ConvSeq.apply_convolution(arr, kernel, normalize=normalize)
        elapsed = time.time() - start
        times_seq.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} seconds")

    avg_seq = np.mean(times_seq)
    std_seq = np.std(times_seq)
    print(f"Average: {avg_seq:.4f} ± {std_seq:.4f} seconds")
    results['sequential'] = (result_seq, avg_seq)

    # Benchmark parallel version (blocks per channel)
    print("\n2. PARALLEL VERSION (blocks per channel)")
    print("-" * 70)
    times_par = []
    for i in range(n_runs):
        start = time.time()
        result_par = ConvParallel.apply_convolution(arr, kernel, normalize=normalize, n_jobs=-1)
        elapsed = time.time() - start
        times_par.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} seconds")

    avg_par = np.mean(times_par)
    std_par = np.std(times_par)
    print(f"Average: {avg_par:.4f} ± {std_par:.4f} seconds")
    speedup_par = avg_seq / avg_par
    print(f"Speedup: {speedup_par:.2f}x")
    results['parallel'] = (result_par, avg_par)

    # Benchmark advanced parallel version (blocks × channels)
    print("\n3. ADVANCED PARALLEL VERSION (blocks × channels)")
    print("-" * 70)
    times_adv = []
    for i in range(n_runs):
        start = time.time()
        result_adv = ConvParallelAdvanced.apply_convolution(arr, kernel, normalize=normalize, n_jobs=-1)
        elapsed = time.time() - start
        times_adv.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} seconds")

    avg_adv = np.mean(times_adv)
    std_adv = np.std(times_adv)
    print(f"Average: {avg_adv:.4f} ± {std_adv:.4f} seconds")
    speedup_adv = avg_seq / avg_adv
    print(f"Speedup: {speedup_adv:.2f}x")
    results['advanced'] = (result_adv, avg_adv)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Sequential:              {avg_seq:.4f}s  (1.00x)")
    print(f"Parallel (blocks):       {avg_par:.4f}s  ({speedup_par:.2f}x speedup)")
    print(f"Parallel (blocks×ch):    {avg_adv:.4f}s  ({speedup_adv:.2f}x speedup)")

    # Find best version
    best = min([(avg_seq, 'Sequential'), (avg_par, 'Parallel'), (avg_adv, 'Advanced')], key=lambda x: x[0])
    print(f"\n🏆 Best: {best[1]} ({best[0]:.4f}s)")

    # Verify results are identical
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    diff_par = np.abs(result_seq.astype(float) - result_par.astype(float)).max()
    diff_adv = np.abs(result_seq.astype(float) - result_adv.astype(float)).max()
    print(f"Max difference (Sequential vs Parallel):  {diff_par}")
    print(f"Max difference (Sequential vs Advanced):  {diff_adv}")

    if diff_par == 0 and diff_adv == 0:
        print("✓ All results are identical!")
    else:
        print("⚠ Results differ slightly (floating point precision)")

    return results

if __name__ == "__main__":
    # Configuration
    input_path = "banana.png"
    kernel = ConvSeq.KERNEL_EDGE
    normalize = True
    n_runs = 3

    print("=" * 70)
    print("CONVOLUTION BENCHMARK: Sequential vs Parallel Versions")
    print("=" * 70)

    results = benchmark_convolution(input_path, kernel, normalize, n_runs)

    # Save all results
    Image.fromarray(results['sequential'][0]).save("output_sequential.jpg")
    Image.fromarray(results['parallel'][0]).save("output_parallel.jpg")
    Image.fromarray(results['advanced'][0]).save("output_advanced.jpg")
    print("\n✓ Results saved to output_sequential.jpg, output_parallel.jpg, output_advanced.jpg")

