#!/usr/bin/env python3
"""
Benchmark script for comparing image convolution implementations.
Generates test images of various sizes and tests different kernel sizes.
Results are saved to benchmark_results.json
"""
import numpy as np
from PIL import Image
import json
import time
import os
import subprocess
from datetime import datetime

# Import the implementations
import ConvSeq
import ConvParallelAdvanced

# Define kernels
KERNELS = {
    "3x3": np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float),
    "5x5": np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ], dtype=float),
    "7x7": np.array([
        [1,  6,  15,  20,  15,  6, 1],
        [6,  36,  90, 120,  90, 36, 6],
        [15, 90, 225, 300, 225, 90, 15],
        [20,120, 300, 400, 300,120, 20],
        [15, 90, 225, 300, 225, 90, 15],
        [6,  36,  90, 120,  90, 36, 6],
        [1,  6,  15,  20,  15,  6, 1]
    ], dtype=float)
}

def generate_test_image(size):
    """Generate a random RGB image of given size."""
    # Create random RGB image
    img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return img_array

def save_test_image(img_array, filename):
    """Save image array to file."""
    img = Image.fromarray(img_array)
    img.save(filename)
    return filename

def benchmark_sequential(img_array, kernel, kernel_name):
    """Benchmark sequential implementation."""
    try:
        start_time = time.time()
        result = ConvSeq.apply_convolution(img_array, kernel, normalize=True)
        end_time = time.time()
        execution_time = end_time - start_time
        return {
            "success": True,
            "time": execution_time,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "time": None,
            "error": str(e)
        }

def benchmark_parallel(img_array, kernel, kernel_name):
    """Benchmark parallel CPU implementation."""
    try:
        start_time = time.time()
        result = ConvParallelAdvanced.apply_convolution(
            img_array, kernel, normalize=True, n_jobs=-1
        )
        end_time = time.time()
        execution_time = end_time - start_time
        return {
            "success": True,
            "time": execution_time,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "time": None,
            "error": str(e)
        }

def benchmark_cuda(image_path, kernel_name, cuda_executable="./build/CudaConv"):
    """Benchmark CUDA implementation."""
    try:
        # Check if CUDA executable exists
        if not os.path.exists(cuda_executable):
            return {
                "success": False,
                "time": None,
                "error": f"CUDA executable not found at {cuda_executable}"
            }
        
        # Run CUDA implementation and capture output
        start_time = time.time()
        result = subprocess.run(
            [cuda_executable, image_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = time.time()
        
        if result.returncode != 0:
            return {
                "success": False,
                "time": None,
                "error": f"CUDA execution failed: {result.stderr}"
            }
        
        # Try to extract time from CUDA output (if available)
        # Otherwise use wall clock time
        cuda_time = None
        for line in result.stdout.split('\n'):
            if "Tempo CUDA" in line or "ms" in line:
                try:
                    # Extract time in milliseconds
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "ms" in part or part.isdigit():
                            cuda_time = float(part.replace("ms", "")) / 1000.0
                            break
                except:
                    pass
        
        execution_time = cuda_time if cuda_time is not None else (end_time - start_time)
        
        return {
            "success": True,
            "time": execution_time,
            "error": None,
            "output": result.stdout
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "time": None,
            "error": "Execution timeout (>5 minutes)"
        }
    except Exception as e:
        return {
            "success": False,
            "time": None,
            "error": str(e)
        }

def run_benchmarks():
    """Run all benchmarks and save results."""
    # Generate image sizes: 50, 100, 200, 400, 800, 1600, 3200, 6400
    image_sizes = []
    size = 50
    while size <= 8000:
        image_sizes.append(size)
        size *= 2
    
    print(f"Starting benchmark with image sizes: {image_sizes}")
    print(f"Kernel sizes: {list(KERNELS.keys())}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Create temp directory for test images
    os.makedirs("benchmark_images", exist_ok=True)
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_sizes": image_sizes,
            "kernel_sizes": list(KERNELS.keys()),
            "implementations": ["sequential", "parallel_cpu", "cuda_gpu"]
        },
        "benchmarks": []
    }
    
    # Run benchmarks for each image size and kernel size
    total_tests = len(image_sizes) * len(KERNELS) * 3  # 3 implementations
    current_test = 0
    
    for size in image_sizes:
        print(f"\n{'='*60}")
        print(f"Testing image size: {size}x{size} pixels")
        print(f"{'='*60}")
        
        # Generate test image
        print(f"Generating {size}x{size} test image...")
        img_array = generate_test_image(size)
        img_path = f"benchmark_images/test_{size}x{size}.png"
        save_test_image(img_array, img_path)
        
        for kernel_name, kernel in KERNELS.items():
            print(f"\n  Kernel: {kernel_name}")
            
            benchmark_result = {
                "image_size": size,
                "kernel_size": kernel_name,
                "results": {}
            }
            
            # Sequential
            current_test += 1
            print(f"    [{current_test}/{total_tests}] Running Sequential... ", end="", flush=True)
            seq_result = benchmark_sequential(img_array, kernel, kernel_name)
            benchmark_result["results"]["sequential"] = seq_result
            if seq_result["success"]:
                print(f"✓ {seq_result['time']:.3f}s")
            else:
                print(f"✗ Error: {seq_result['error']}")
            
            # Parallel CPU
            current_test += 1
            print(f"    [{current_test}/{total_tests}] Running Parallel CPU... ", end="", flush=True)
            par_result = benchmark_parallel(img_array, kernel, kernel_name)
            benchmark_result["results"]["parallel_cpu"] = par_result
            if par_result["success"]:
                print(f"✓ {par_result['time']:.3f}s")
            else:
                print(f"✗ Error: {par_result['error']}")
            
            # CUDA GPU
            current_test += 1
            print(f"    [{current_test}/{total_tests}] Running CUDA GPU... ", end="", flush=True)
            cuda_result = benchmark_cuda(img_path, kernel_name)
            benchmark_result["results"]["cuda_gpu"] = cuda_result
            if cuda_result["success"]:
                print(f"✓ {cuda_result['time']:.3f}s")
            else:
                print(f"✗ Error: {cuda_result['error']}")
            
            # Calculate speedups
            if seq_result["success"] and par_result["success"]:
                speedup_cpu = seq_result["time"] / par_result["time"]
                benchmark_result["speedup_parallel_vs_sequential"] = speedup_cpu
                print(f"    Speedup (Parallel/Sequential): {speedup_cpu:.2f}x")
            
            if seq_result["success"] and cuda_result["success"]:
                speedup_gpu = seq_result["time"] / cuda_result["time"]
                benchmark_result["speedup_cuda_vs_sequential"] = speedup_gpu
                print(f"    Speedup (CUDA/Sequential): {speedup_gpu:.2f}x")
            
            results["benchmarks"].append(benchmark_result)
    
    # Save results to JSON
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to {output_file}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total tests run: {current_test}/{total_tests}")
    successful_tests = sum(1 for b in results["benchmarks"] 
                          for impl in b["results"].values() 
                          if impl["success"])
    print(f"Successful tests: {successful_tests}/{current_test * 3}")
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("Image Convolution Benchmark Suite")
    print("="*60)
    
    # Check if CUDA executable exists
    cuda_path = "./build/CudaConv"
    if not os.path.exists(cuda_path):
        print(f"\n⚠️  Warning: CUDA executable not found at {cuda_path}")
        print("CUDA benchmarks will be skipped.")
        print("To include CUDA benchmarks, build the project first:")
        print("  mkdir build && cd build && cmake .. && make")
        print()
        response = input("Continue without CUDA? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            exit(0)
    
    try:
        results = run_benchmarks()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
