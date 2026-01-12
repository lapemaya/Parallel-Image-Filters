#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark_results.json
Creates charts showing performance comparisons.
"""
import json
import sys

def print_table(results):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS TABLE")
    print("="*100)
    
    # Header
    print(f"\n{'Image Size':<12} {'Kernel':<8} {'Sequential':<12} {'Parallel CPU':<14} {'CUDA GPU':<12} {'CPU Speedup':<12} {'GPU Speedup':<12}")
    print("-"*100)
    
    for benchmark in results["benchmarks"]:
        size = benchmark["image_size"]
        kernel = benchmark["kernel_size"]
        
        seq = benchmark["results"]["sequential"]
        par = benchmark["results"]["parallel_cpu"]
        cuda = benchmark["results"]["cuda_gpu"]
        
        seq_time = f"{seq['time']:.3f}s" if seq["success"] else "FAIL"
        par_time = f"{par['time']:.3f}s" if par["success"] else "FAIL"
        cuda_time = f"{cuda['time']:.3f}s" if cuda["success"] else "FAIL"
        
        cpu_speedup = f"{benchmark.get('speedup_parallel_vs_sequential', 0):.2f}x" if "speedup_parallel_vs_sequential" in benchmark else "N/A"
        gpu_speedup = f"{benchmark.get('speedup_cuda_vs_sequential', 0):.2f}x" if "speedup_cuda_vs_sequential" in benchmark else "N/A"
        
        print(f"{size}x{size:<6} {kernel:<8} {seq_time:<12} {par_time:<14} {cuda_time:<12} {cpu_speedup:<12} {gpu_speedup:<12}")
    
    print("-"*100)

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    # Calculate averages for each implementation
    for impl_name in ["sequential", "parallel_cpu", "cuda_gpu"]:
        times = [b["results"][impl_name]["time"] 
                for b in results["benchmarks"] 
                if b["results"][impl_name]["success"]]
        
        if times:
            print(f"\n{impl_name.upper().replace('_', ' ')}:")
            print(f"  Average time: {sum(times)/len(times):.3f}s")
            print(f"  Min time: {min(times):.3f}s")
            print(f"  Max time: {max(times):.3f}s")
            print(f"  Successful runs: {len(times)}/{len(results['benchmarks'])}")
    
    # Average speedups
    cpu_speedups = [b["speedup_parallel_vs_sequential"] 
                   for b in results["benchmarks"] 
                   if "speedup_parallel_vs_sequential" in b]
    
    gpu_speedups = [b["speedup_cuda_vs_sequential"] 
                   for b in results["benchmarks"] 
                   if "speedup_cuda_vs_sequential" in b]
    
    if cpu_speedups:
        print(f"\nAVERAGE SPEEDUP (Parallel CPU vs Sequential): {sum(cpu_speedups)/len(cpu_speedups):.2f}x")
    
    if gpu_speedups:
        print(f"AVERAGE SPEEDUP (CUDA GPU vs Sequential): {sum(gpu_speedups)/len(gpu_speedups):.2f}x")

def print_best_performers(results):
    """Print best performing configurations."""
    print("\n" + "="*100)
    print("BEST PERFORMERS")
    print("="*100)
    
    # Find fastest for each kernel size
    kernel_sizes = set(b["kernel_size"] for b in results["benchmarks"])
    
    for kernel in kernel_sizes:
        print(f"\n{kernel} KERNEL:")
        kernel_benchmarks = [b for b in results["benchmarks"] if b["kernel_size"] == kernel]
        
        for impl_name in ["sequential", "parallel_cpu", "cuda_gpu"]:
            successful = [(b["image_size"], b["results"][impl_name]["time"]) 
                         for b in kernel_benchmarks 
                         if b["results"][impl_name]["success"]]
            
            if successful:
                fastest = min(successful, key=lambda x: x[1])
                print(f"  {impl_name.replace('_', ' ').title()}: {fastest[1]:.3f}s ({fastest[0]}x{fastest[0]} image)")

def load_and_visualize(json_file="benchmark_results.json"):
    """Load benchmark results and create visualizations."""
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Run benchmark.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not a valid JSON file.")
        return
    
    print("\n" + "="*100)
    print(f"BENCHMARK RESULTS VISUALIZATION")
    print(f"Timestamp: {results['metadata']['timestamp']}")
    print("="*100)
    
    print_table(results)
    print_summary(results)
    print_best_performers(results)
    
    print("\n" + "="*100)
    print("For graphical plots, consider using matplotlib:")
    print("  pip install matplotlib")
    print("Then modify this script to add plotting functionality.")
    print("="*100 + "\n")

if __name__ == "__main__":
    json_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    load_and_visualize(json_file)
