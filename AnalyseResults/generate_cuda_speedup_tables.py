#!/usr/bin/env python3
"""
Generate LaTeX tables for CUDA speedup vs sequential.
Creates 3 tables (one per kernel size) showing speedup vs tile_size and image_size.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def load_speedup_data(json_path: str = 'speedup_results.json') -> Tuple[List[Dict], List[Dict]]:
    """Load speedup results from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['cpu_parallel_speedup'], data['cuda_speedup']


def generate_cuda_speedup_tables(cuda_data: List[Dict], output_file: str = 'cuda_speedup_tables.tex'):
    """Generate LaTeX tables for CUDA speedup (one table per kernel size)."""
    
    # Get all unique kernel sizes, image sizes, and tile_sizes
    kernel_sizes = sorted(set(d['kernel_size'] for d in cuda_data))
    image_sizes = sorted(set(d['image_size'] for d in cuda_data))
    tile_sizes = sorted(set(d['tile_size'] for d in cuda_data))
    
    # Organize data by kernel -> image -> tile -> speedup
    data_dict = {}
    for d in cuda_data:
        k = d['kernel_size']
        img = d['image_size']
        tile = d['tile_size']
        speedup = d['speedup']
        
        if k not in data_dict:
            data_dict[k] = {}
        if img not in data_dict[k]:
            data_dict[k][img] = {}
        data_dict[k][img][tile] = speedup
    
    with open(output_file, 'w') as f:
        # LaTeX preamble
        f.write("\\documentclass[11pt]{article}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage{colortbl}\n")
        f.write("\n")
        f.write("\\title{CUDA Speedup Analysis}\n")
        f.write("\\author{}\n")
        f.write("\\date{\\today}\n")
        f.write("\n")
        f.write("\\begin{document}\n")
        f.write("\\maketitle\n")
        f.write("\n")
        
        f.write("\\section{CUDA Speedup vs Sequential CPU}\n")
        f.write("\\noindent Speedup computed as: $S = \\frac{T_{\\text{sequential CPU}}}{T_{\\text{CUDA}}}$\n\n")
        f.write("\\noindent Performance measured at different tile sizes for shared memory optimization.\n\n")
        
        # Generate one table per kernel size
        for kernel_size in kernel_sizes:
            f.write(f"\\subsection{{Kernel Size: {kernel_size}×{kernel_size}}}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{CUDA Speedup for kernel {kernel_size}×{kernel_size} at different image sizes and tile sizes}}\n")
            f.write(f"\\label{{tab:cuda_speedup_k{kernel_size}}}\n")
            
            # Table header
            n_cols = len(tile_sizes)
            f.write("\\begin{tabular}{l" + "c" * n_cols + "}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Image Size} & " + 
                   " & ".join([f"\\textbf{{tile={tile}}}" for tile in tile_sizes]) + " \\\\\n")
            f.write("\\midrule\n")
            
            # Table rows (one row per image size)
            for img_size in image_sizes:
                row = [f"{img_size}×{img_size}"]
                
                for tile in tile_sizes:
                    if kernel_size in data_dict and img_size in data_dict[kernel_size]:
                        speedup = data_dict[kernel_size][img_size].get(tile, None)
                        if speedup is not None:
                            # Color code based on speedup magnitude
                            if speedup >= 100:  # Very high speedup
                                cell = f"\\cellcolor{{green!30}}\\textbf{{{speedup:.1f}×}}"
                            elif speedup >= 50:  # High speedup
                                cell = f"\\cellcolor{{green!15}}{speedup:.1f}×"
                            elif speedup >= 10:  # Moderate speedup
                                cell = f"\\cellcolor{{yellow!20}}{speedup:.1f}×"
                            else:  # Low speedup
                                cell = f"{speedup:.1f}×"
                            row.append(cell)
                        else:
                            row.append("---")
                    else:
                        row.append("---")
                
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("\n")
            
            # Add execution time table
            f.write(f"\\subsection{{CUDA Execution Time for Kernel {kernel_size}×{kernel_size}}}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{CUDA execution time (milliseconds) for kernel {kernel_size}×{kernel_size}}}\n")
            f.write(f"\\label{{tab:cuda_time_k{kernel_size}}}\n")
            
            f.write("\\begin{tabular}{l" + "c" * n_cols + "}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Image Size} & " + 
                   " & ".join([f"\\textbf{{tile={tile}}}" for tile in tile_sizes]) + " \\\\\n")
            f.write("\\midrule\n")
            
            # Find time data from original cuda_data
            time_dict = {}
            for d in cuda_data:
                if d['kernel_size'] == kernel_size:
                    img = d['image_size']
                    tile = d['tile_size']
                    time_ms = d['time_seconds'] * 1000  # Convert to ms
                    if img not in time_dict:
                        time_dict[img] = {}
                    time_dict[img][tile] = time_ms
            
            for img_size in image_sizes:
                row = [f"{img_size}×{img_size}"]
                
                for tile in tile_sizes:
                    if img_size in time_dict and tile in time_dict[img_size]:
                        time_ms = time_dict[img_size][tile]
                        if time_ms < 10:
                            cell = f"\\cellcolor{{green!20}}{time_ms:.2f}"
                        elif time_ms < 100:
                            cell = f"{time_ms:.2f}"
                        else:
                            cell = f"{time_ms:.1f}"
                        row.append(cell)
                    else:
                        row.append("---")
                
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("\n")
            f.write("\\clearpage\n\n")
        
        # Summary section
        f.write("\\section{Summary}\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item \\colorbox{green!30}{Dark Green}: Very high speedup (≥100×)\n")
        f.write("\\item \\colorbox{green!15}{Light Green}: High speedup (50-100×)\n")
        f.write("\\item \\colorbox{yellow!20}{Yellow}: Moderate speedup (10-50×)\n")
        f.write("\\item No color: Lower speedup (<10×)\n")
        f.write("\\end{itemize}\n")
        f.write("\n")
        
        f.write("\\noindent \\textbf{Key Observations:}\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item CUDA provides substantial speedup over sequential CPU, especially for larger images\n")
        f.write("\\item Tile size affects shared memory utilization and occupancy\n")
        f.write("\\item Performance variation with tile size indicates memory access patterns and cache behavior\n")
        f.write("\\item Larger images benefit more from GPU parallelization due to better amortization of overhead\n")
        f.write("\\end{itemize}\n")
        
        # Tile size analysis
        f.write("\n\\subsection{Tile Size Sensitivity}\n")
        f.write("\\noindent The following table shows the performance variation across tile sizes:\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Speedup variation with tile size (\\%)}\n")
        f.write("\\label{tab:tile_sensitivity}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Kernel Size} & \\textbf{Avg. Speedup} & \\textbf{Tile Sensitivity} \\\\\n")
        f.write("\\midrule\n")
        
        for kernel_size in kernel_sizes:
            speedups = []
            for img_size in image_sizes:
                for tile in tile_sizes:
                    if kernel_size in data_dict and img_size in data_dict[kernel_size]:
                        s = data_dict[kernel_size][img_size].get(tile, None)
                        if s is not None:
                            speedups.append(s)
            
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                max_s = max(speedups)
                min_s = min(speedups)
                sensitivity = ((max_s - min_s) / min_s) * 100
                f.write(f"{kernel_size}×{kernel_size} & {avg_speedup:.1f}× & {sensitivity:.1f}\\% \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\n")
        
        f.write("\\noindent \\textbf{Note:} Low tile sensitivity indicates that performance is not heavily bottlenecked ")
        f.write("by shared memory configuration, possibly dominated by memory transfers or compute intensity.\n")
        
        # Time per MAC analysis
        f.write("\n\\subsection{Normalized Performance: Time per MAC}\n")
        f.write("\\noindent This metric normalizes execution time by the number of multiply-accumulate operations, ")
        f.write("providing insight into how efficiently the hardware is utilized regardless of problem size.\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Time per MAC (nanoseconds) for CUDA - Best tile size}\n")
        f.write("\\label{tab:cuda_time_per_mac}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Image Size} & \\textbf{K=3} & \\textbf{K=5} & \\textbf{K=7} \\\\\n")
        f.write("\\midrule\n")
        
        # Collect time_per_mac data
        mac_data = {}
        for d in cuda_data:
            if 'time_per_mac_ns' in d:
                k = d['kernel_size']
                img = d['image_size']
                tile = d['tile_size']
                mac_ns = d['time_per_mac_ns']
                
                if img not in mac_data:
                    mac_data[img] = {}
                if k not in mac_data[img]:
                    mac_data[img][k] = []
                mac_data[img][k].append(mac_ns)
        
        for img_size in image_sizes:
            row = [f"{img_size}×{img_size}"]
            for k in kernel_sizes:
                if img_size in mac_data and k in mac_data[img_size]:
                    # Get best (minimum) time per MAC
                    best_mac = min(mac_data[img_size][k])
                    row.append(f"{best_mac:.4f}")
                else:
                    row.append("---")
            f.write(" & ".join(row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\n")
        
        f.write("\\noindent \\textbf{Observation:} If time/MAC is relatively constant across kernel sizes, ")
        f.write("it suggests the implementation scales well with computational intensity. ")
        f.write("Variations may indicate different bottlenecks (memory vs compute) for different kernel sizes.\n")
        
        f.write("\n\\end{document}\n")
    
    print(f"✓ CUDA speedup tables written to: {output_file}")


def main():
    # Parse arguments
    json_path = 'speedup_results.json'
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    output_file = 'cuda_speedup_tables.tex'
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not Path(json_path).exists():
        print(f"❌ Error: {json_path} not found!")
        print("Run 'python3 compute_speedup.py' first to generate speedup data.")
        return 1
    
    print(f"📊 Loading speedup data from: {json_path}")
    cpu_data, cuda_data = load_speedup_data(json_path)
    
    print(f"   CUDA configurations: {len(cuda_data)}")
    
    print("\n📝 Generating LaTeX tables...")
    generate_cuda_speedup_tables(cuda_data, output_file)
    
    print("\n✅ LaTeX generation complete!")
    print(f"\nTo compile:")
    print(f"  pdflatex {output_file}")
    print(f"  pdflatex {output_file}  # Run twice for references")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
