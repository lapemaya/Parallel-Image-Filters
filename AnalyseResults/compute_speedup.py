#!/usr/bin/env python3
"""
Compute speedup metrics from benchmark results.
Calculates speedup for both CPU parallel and CUDA implementations vs sequential baseline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_benchmark_data(json_path: str = 'benchmark_resultsold.json') -> Tuple[dict, List[dict]]:
    """Load benchmark JSON and return metadata + results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('metadata', {}), data.get('results', [])


def get_baseline_time(results: List[dict], image_size: int, kernel_size: int) -> float | None:
    """Get sequential baseline time for a given configuration."""
    for r in results:
        if (r['py_module'] == 'ConvSeqDumb' and
            r['image_size'] == image_size and
            r['kernel_size'] == kernel_size and
            r['python_seconds'] != -1):
            return r['python_seconds']
    return None


def estimate_macs(image_size: int, kernel_size: int, channels: int = 3) -> int:
    """Estimate MACs (multiply-accumulate operations) for convolution."""
    return image_size * image_size * channels * kernel_size * kernel_size


def compute_cpu_speedup(results: List[dict]) -> List[Dict]:
    """Compute speedup for CPU parallel implementation."""
    speedup_data = []
    
    # Get all unique configurations
    configs = set()
    for r in results:
        if r['py_module'] == 'ConvParallelAdvancedDumb' and r['python_seconds'] != -1:
            configs.add((r['image_size'], r['kernel_size']))
    
    # Compute speedup for each configuration
    for image_size, kernel_size in sorted(configs):
        baseline = get_baseline_time(results, image_size, kernel_size)
        if baseline is None:
            continue
        
        # Get all n_jobs measurements for this config
        parallel_runs = [r for r in results
                        if r['py_module'] == 'ConvParallelAdvancedDumb'
                        and r['image_size'] == image_size
                        and r['kernel_size'] == kernel_size
                        and r['python_seconds'] != -1]
        
        macs = estimate_macs(image_size, kernel_size)
        
        for run in parallel_runs:
            speedup = baseline / run['python_seconds']
            efficiency = speedup / run['n_jobs']
            time_per_mac_ns = (run['python_seconds'] * 1e9) / macs
            
            speedup_data.append({
                'implementation': 'CPU_Parallel',
                'image_size': image_size,
                'kernel_size': kernel_size,
                'n_jobs': run['n_jobs'],
                'tile_size': None,
                'time_seconds': run['python_seconds'],
                'baseline_seconds': baseline,
                'speedup': speedup,
                'efficiency': efficiency,
                'macs': macs,
                'time_per_mac_ns': time_per_mac_ns
            })
    
    return speedup_data


def compute_cuda_speedup(results: List[dict]) -> List[Dict]:
    """Compute speedup for CUDA implementation."""
    speedup_data = []
    
    # Get all unique configurations
    configs = set()
    for r in results:
        if r['py_module'] == 'CUDA' and r['cuda_ms'] != -1:
            configs.add((r['image_size'], r['kernel_size']))
    
    # Compute speedup for each configuration
    for image_size, kernel_size in sorted(configs):
        baseline = get_baseline_time(results, image_size, kernel_size)
        if baseline is None:
            continue
        
        # Get all tile_size measurements for this config
        cuda_runs = [r for r in results
                    if r['py_module'] == 'CUDA'
                    and r['image_size'] == image_size
                    and r['kernel_size'] == kernel_size
                    and r['cuda_ms'] != -1]
        
        macs = estimate_macs(image_size, kernel_size)
        
        for run in cuda_runs:
            cuda_seconds = run['cuda_ms'] / 1000.0
            speedup = baseline / cuda_seconds
            time_per_mac_ns = (cuda_seconds * 1e9) / macs
            
            speedup_data.append({
                'implementation': 'CUDA',
                'image_size': image_size,
                'kernel_size': kernel_size,
                'n_jobs': None,
                'tile_size': run['tile_size'],
                'time_seconds': cuda_seconds,
                'baseline_seconds': baseline,
                'speedup': speedup,
                'efficiency': None,
                'macs': macs,
                'time_per_mac_ns': time_per_mac_ns
            })
    
    return speedup_data


def print_speedup_table(speedup_data: List[Dict], title: str):
    """Print formatted speedup table."""
    if not speedup_data:
        print(f"\n{title}: No data available")
        return
    
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print('='*100)
    
    # Determine columns based on implementation
    impl = speedup_data[0]['implementation']
    
    if impl == 'CPU_Parallel':
        print(f"{'Image':<10} {'Kernel':<10} {'n_jobs':<10} {'Time (s)':<15} "
              f"{'Baseline (s)':<15} {'Speedup':<12} {'Efficiency':<12}")
        print('-'*100)
        
        for d in speedup_data:
            print(f"{d['image_size']:<10} {d['kernel_size']:<10} {d['n_jobs']:<10} "
                  f"{d['time_seconds']:<15.6f} {d['baseline_seconds']:<15.6f} "
                  f"{d['speedup']:<12.2f} {d['efficiency']:<12.2%}")
    
    else:  # CUDA
        print(f"{'Image':<10} {'Kernel':<10} {'Tile':<10} {'Time (s)':<15} "
              f"{'Baseline (s)':<15} {'Speedup':<12}")
        print('-'*100)
        
        for d in speedup_data:
            print(f"{d['image_size']:<10} {d['kernel_size']:<10} {d['tile_size']:<10} "
                  f"{d['time_seconds']:<15.6f} {d['baseline_seconds']:<15.6f} "
                  f"{d['speedup']:<12.2f}")
    
    print('='*100)


def print_best_speedups(cpu_data: List[Dict], cuda_data: List[Dict]):
    """Print best speedup for each configuration."""
    print(f"\n{'='*100}")
    print(f"{'BEST SPEEDUP SUMMARY':^100}")
    print('='*100)
    print(f"{'Image':<12} {'Kernel':<12} {'Best CPU':<20} {'Best CUDA':<20}")
    print('-'*100)
    
    # Get all configs
    all_configs = set()
    for d in cpu_data:
        all_configs.add((d['image_size'], d['kernel_size']))
    for d in cuda_data:
        all_configs.add((d['image_size'], d['kernel_size']))
    
    for image_size, kernel_size in sorted(all_configs):
        # Best CPU
        cpu_matches = [d for d in cpu_data 
                      if d['image_size'] == image_size and d['kernel_size'] == kernel_size]
        if cpu_matches:
            best_cpu = max(cpu_matches, key=lambda x: x['speedup'])
            cpu_str = f"{best_cpu['speedup']:.2f}x (n={best_cpu['n_jobs']})"
        else:
            cpu_str = "---"
        
        # Best CUDA
        cuda_matches = [d for d in cuda_data 
                       if d['image_size'] == image_size and d['kernel_size'] == kernel_size]
        if cuda_matches:
            best_cuda = max(cuda_matches, key=lambda x: x['speedup'])
            cuda_str = f"{best_cuda['speedup']:.2f}x (tile={best_cuda['tile_size']})"
        else:
            cuda_str = "---"
        
        print(f"{image_size:<12} {kernel_size:<12} {cpu_str:<20} {cuda_str:<20}")
    
    print('='*100)


def save_speedup_json(cpu_data: List[Dict], cuda_data: List[Dict], output_path: str = 'speedup_results.json'):
    """Save speedup results to JSON file."""
    output = {
        'cpu_parallel_speedup': cpu_data,
        'cuda_speedup': cuda_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Speedup results saved to: {output_path}")


def main():
    # Parse arguments
    json_path = 'benchmark_results.json'
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    if not Path(json_path).exists():
        print(f"❌ Error: {json_path} not found!")
        print("Usage: python3 compute_speedup.py [benchmark_results.json]")
        return 1
    
    print(f"📊 Loading benchmark data from: {json_path}")
    metadata, results = load_benchmark_data(json_path)
    
    print(f"   Found {len(results)} benchmark results")
    
    # Compute speedups
    print("\n🔄 Computing speedups...")
    cpu_speedup = compute_cpu_speedup(results)
    cuda_speedup = compute_cuda_speedup(results)
    
    print(f"   CPU Parallel: {len(cpu_speedup)} configurations")
    print(f"   CUDA: {len(cuda_speedup)} configurations")
    
    # Print tables
    print_speedup_table(cpu_speedup, "CPU PARALLEL SPEEDUP (vs Sequential)")
    print_speedup_table(cuda_speedup, "CUDA SPEEDUP (vs Sequential)")
    
    # Print best summary
    print_best_speedups(cpu_speedup, cuda_speedup)
    
    # Save to JSON
    save_speedup_json(cpu_speedup, cuda_speedup)
    
    print("\n✅ Speedup analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
