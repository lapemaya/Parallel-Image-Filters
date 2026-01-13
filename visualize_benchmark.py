#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark_results.json
Creates bar charts comparing execution times across implementations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_path='benchmark_results.json'):
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def organize_data(data):
    """Organize results by implementation, image size, and kernel size."""
    results = {}
    
    for entry in data['results']:
        img_size = entry['image_size']
        kernel_size = entry['kernel_size']
        impl = entry['py_module']
        
        # Convert to milliseconds for comparison
        if impl == 'CUDA':
            time_ms = entry['cuda_ms']
        else:
            time_ms = entry['python_seconds'] * 1000
        
        key = (img_size, kernel_size)
        if key not in results:
            results[key] = {}
        results[key][impl] = time_ms
    
    return results

def plot_benchmark_results(data):
    """Create visualization of benchmark results with execution times and speedups."""
    results = organize_data(data)
    
    # Get unique implementations and sort
    all_impls = set()
    for times in results.values():
        all_impls.update(times.keys())
    implementations = sorted(list(all_impls))
    
    # Get unique combinations and sort
    configs = sorted(results.keys())
    
    # Group by image size
    img_sizes = sorted(set(img_size for img_size, _ in configs))
    n_sizes = len(img_sizes)
    
    # Create figure with 2 rows: execution times and speedups
    # Number of columns = number of different image sizes
    fig = plt.figure(figsize=(8 * n_sizes, 12))
    gs = fig.add_gridspec(2, n_sizes, hspace=0.3, wspace=0.25)
    
    # Row 1: Execution times
    fig.text(0.5, 0.96, 'Image Convolution Benchmark Results', 
             ha='center', fontsize=16, fontweight='bold')
    
    for idx, img_size in enumerate(img_sizes):
        ax = fig.add_subplot(gs[0, idx])
        
        # Filter configs for this image size
        size_configs = [(img, kern) for img, kern in configs if img == img_size]
        kernel_sizes = [kern for _, kern in size_configs]
        
        # Prepare data for grouped bar chart
        x = np.arange(len(kernel_sizes))
        width = 0.25
        
        # Plot bars for each implementation
        for i, impl in enumerate(implementations):
            times = [results[(img_size, kern)].get(impl, 0) for kern in kernel_sizes]
            offset = (i - len(implementations)/2 + 0.5) * width
            bars = ax.bar(x + offset, times, width, label=impl)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Kernel Size', fontsize=11, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'Execution Time - {img_size}x{img_size}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}x{k}' for k in kernel_sizes])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Speedups relative to ConvSeq
    for idx, img_size in enumerate(img_sizes):
        ax = fig.add_subplot(gs[1, idx])
        
        # Filter configs for this image size
        size_configs = [(img, kern) for img, kern in configs if img == img_size]
        kernel_sizes = [kern for _, kern in size_configs]
        
        # Prepare speedup data
        x = np.arange(len(kernel_sizes))
        width = 0.35
        
        # Calculate speedups vs ConvSeq
        for i, impl in enumerate(implementations):
            if impl == 'ConvSeq':
                continue  # Skip baseline
            
            speedups = []
            for kern in kernel_sizes:
                times_dict = results[(img_size, kern)]
                if 'ConvSeq' in times_dict and impl in times_dict:
                    baseline = times_dict['ConvSeq']
                    impl_time = times_dict[impl]
                    speedup = baseline / impl_time if impl_time > 0 else 0
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, speedups, width, label=impl)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}x',
                           ha='center', va='bottom', fontsize=8)
        
        # Add reference line at 1.0x
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (ConvSeq)')
        
        ax.set_xlabel('Kernel Size', fontsize=11, fontweight='bold')
        ax.set_ylabel('Speedup vs ConvSeq', fontsize=11, fontweight='bold')
        ax.set_title(f'Speedup - {img_size}x{img_size}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}x{k}' for k in kernel_sizes])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('benchmark_plot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Execution times plot saved to: benchmark_plot.png")
    plt.show()

def print_summary(data):
    """Print summary statistics."""
    results = organize_data(data)
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    for (img_size, kernel_size), times in sorted(results.items()):
        print(f"\nImage: {img_size}x{img_size} | Kernel: {kernel_size}x{kernel_size}")
        print("-" * 70)
        
        # Sort by time
        sorted_times = sorted(times.items(), key=lambda x: x[1])
        
        for impl, time_ms in sorted_times:
            print(f"  {impl:20s}: {time_ms:8.2f} ms")
        
        # Calculate speedups relative to ConvSeq
        if 'ConvSeq' in times:
            baseline = times['ConvSeq']
            print(f"\n  Speedups vs ConvSeq:")
            for impl, time_ms in sorted_times:
                if impl != 'ConvSeq' and time_ms > 0:
                    speedup = baseline / time_ms
                    print(f"    {impl:20s}: {speedup:6.2f}x")

def main():
    # Load results
    json_path = Path('benchmark_results.json')
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print("Run './benchmark.sh --quick' first to generate results.")
        return
    
    data = load_results(json_path)
    
    # Print summary
    print_summary(data)
    
    # Create visualization
    print("\nGenerating plot...")
    plot_benchmark_results(data)
    
    print("\n✓ Visualization complete!")

if __name__ == '__main__':
    main()
