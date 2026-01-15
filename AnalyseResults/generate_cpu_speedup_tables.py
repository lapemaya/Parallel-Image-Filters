#!/usr/bin/env python3
"""
Generate LaTeX tables for CPU parallel speedup vs sequential.
Creates 3 tables (one per kernel size) showing speedup vs n_jobs and image_size.
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


def generate_cpu_speedup_tables(cpu_data: List[Dict], output_file: str = 'cpu_speedup_tables.tex'):
    """Generate LaTeX tables for CPU parallel speedup (one table per kernel size)."""
    
    # Get all unique kernel sizes, image sizes, and n_jobs
    kernel_sizes = sorted(set(d['kernel_size'] for d in cpu_data))
    image_sizes = sorted(set(d['image_size'] for d in cpu_data))
    n_jobs_list = sorted(set(d['n_jobs'] for d in cpu_data))
    
    # Organize data by kernel -> image -> n_jobs -> speedup
    data_dict = {}
    for d in cpu_data:
        k = d['kernel_size']
        img = d['image_size']
        nj = d['n_jobs']
        speedup = d['speedup']
        
        if k not in data_dict:
            data_dict[k] = {}
        if img not in data_dict[k]:
            data_dict[k][img] = {}
        data_dict[k][img][nj] = speedup
    
    with open(output_file, 'w') as f:
        # LaTeX preamble
        f.write("\\documentclass[11pt]{article}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage{colortbl}\n")
        f.write("\n")
        f.write("\\title{CPU Parallel Speedup Analysis}\n")
        f.write("\\author{}\n")
        f.write("\\date{\\today}\n")
        f.write("\n")
        f.write("\\begin{document}\n")
        f.write("\\maketitle\n")
        f.write("\n")
        
        f.write("\\section{CPU Parallel Speedup vs Sequential}\n")
        f.write("\\noindent Speedup computed as: $S = \\frac{T_{\\text{sequential}}}{T_{\\text{parallel}}(n)}$\n\n")
        f.write("\\noindent where $n$ is the number of parallel jobs (\\texttt{n\\_jobs}).\n\n")
        
        # Generate one table per kernel size
        for kernel_size in kernel_sizes:
            f.write(f"\\subsection{{Kernel Size: {kernel_size}×{kernel_size}}}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Speedup for kernel {kernel_size}×{kernel_size} at different image sizes and n\\_jobs}}\n")
            f.write(f"\\label{{tab:speedup_k{kernel_size}}}\n")
            
            # Table header
            n_cols = len(n_jobs_list)
            f.write("\\begin{tabular}{l" + "c" * n_cols + "}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Image Size} & " + 
                   " & ".join([f"\\textbf{{{nj} jobs}}" for nj in n_jobs_list]) + " \\\\\n")
            f.write("\\midrule\n")
            
            # Table rows (one row per image size)
            for img_size in image_sizes:
                row = [f"{img_size}×{img_size}"]
                
                for nj in n_jobs_list:
                    if kernel_size in data_dict and img_size in data_dict[kernel_size]:
                        speedup = data_dict[kernel_size][img_size].get(nj, None)
                        if speedup is not None:
                            # Color code speedup
                            if speedup >= nj * 0.8:  # >= 80% efficiency
                                cell = f"\\cellcolor{{green!20}}{speedup:.2f}×"
                            elif speedup >= nj * 0.5:  # >= 50% efficiency
                                cell = f"\\cellcolor{{yellow!20}}{speedup:.2f}×"
                            else:
                                cell = f"{speedup:.2f}×"
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
            
            # Add efficiency table
            f.write(f"\\subsection{{Parallel Efficiency for Kernel {kernel_size}×{kernel_size}}}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Efficiency (\\%) for kernel {kernel_size}×{kernel_size}: $E = \\frac{{S}}{{n}} \\times 100$}}\n")
            f.write(f"\\label{{tab:efficiency_k{kernel_size}}}\n")
            
            f.write("\\begin{tabular}{l" + "c" * n_cols + "}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Image Size} & " + 
                   " & ".join([f"\\textbf{{{nj} jobs}}" for nj in n_jobs_list]) + " \\\\\n")
            f.write("\\midrule\n")
            
            for img_size in image_sizes:
                row = [f"{img_size}×{img_size}"]
                
                for nj in n_jobs_list:
                    if kernel_size in data_dict and img_size in data_dict[kernel_size]:
                        speedup = data_dict[kernel_size][img_size].get(nj, None)
                        if speedup is not None:
                            efficiency = (speedup / nj) * 100
                            if efficiency >= 80:
                                cell = f"\\cellcolor{{green!20}}{efficiency:.1f}\\%"
                            elif efficiency >= 50:
                                cell = f"\\cellcolor{{yellow!20}}{efficiency:.1f}\\%"
                            else:
                                cell = f"{efficiency:.1f}\\%"
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
            f.write("\\clearpage\n\n")
        
        # Summary section
        f.write("\\section{Summary}\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item \\colorbox{green!20}{Green}: High efficiency (≥80\\%)\n")
        f.write("\\item \\colorbox{yellow!20}{Yellow}: Moderate efficiency (50-80\\%)\n")
        f.write("\\item No color: Lower efficiency (<50\\%)\n")
        f.write("\\end{itemize}\n")
        f.write("\n")
        
        f.write("\\noindent \\textbf{Key Observations:}\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item Speedup generally increases with the number of jobs but shows diminishing returns\n")
        f.write("\\item Larger images tend to achieve better parallel efficiency\n")
        f.write("\\item Kernel size affects the computational workload and parallel scaling behavior\n")
        f.write("\\end{itemize}\n")
        
        # Time per MAC analysis
        f.write("\n\\subsection{Normalized Performance: Time per MAC}\n")
        f.write("\\noindent This metric normalizes execution time by the number of multiply-accumulate operations ")
        f.write("(MACs = $H \\times W \\times 3 \\times K^2$), providing insight into computational efficiency.\n\n")
        
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Time per MAC (nanoseconds) for CPU Parallel - Best n\\_jobs}\n")
        f.write("\\label{tab:cpu_time_per_mac}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Image Size} & \\textbf{K=3} & \\textbf{K=5} & \\textbf{K=7} \\\\\n")
        f.write("\\midrule\n")
        
        # Collect time_per_mac data from cpu_data
        mac_data_cpu = {}
        for d in cpu_data:
            if 'time_per_mac_ns' in d:
                k = d['kernel_size']
                img = d['image_size']
                njobs = d['n_jobs']
                mac_ns = d['time_per_mac_ns']
                
                if img not in mac_data_cpu:
                    mac_data_cpu[img] = {}
                if k not in mac_data_cpu[img]:
                    mac_data_cpu[img][k] = []
                mac_data_cpu[img][k].append(mac_ns)
        
        for img_size in image_sizes:
            row = [f"{img_size}×{img_size}"]
            for k in kernel_sizes:
                if img_size in mac_data_cpu and k in mac_data_cpu[img_size]:
                    # Get best (minimum) time per MAC
                    best_mac = min(mac_data_cpu[img_size][k])
                    row.append(f"{best_mac:.4f}")
                else:
                    row.append("---")
            f.write(" & ".join(row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("\n")
        
        f.write("\\noindent \\textbf{Interpretation:} Lower ns/MAC values indicate better computational efficiency. ")
        f.write("If time/MAC varies significantly with kernel size, it suggests the implementation may have ")
        f.write("different bottlenecks (e.g., cache effects, memory bandwidth) depending on $K$.\n")
        
        f.write("\n\\end{document}\n")
    
    print(f"✓ CPU speedup tables written to: {output_file}")


def main():
    # Parse arguments
    json_path = 'speedup_results.json'
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    output_file = 'cpu_speedup_tables.tex'
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not Path(json_path).exists():
        print(f"❌ Error: {json_path} not found!")
        print("Run 'python3 compute_speedup.py' first to generate speedup data.")
        return 1
    
    print(f"📊 Loading speedup data from: {json_path}")
    cpu_data, cuda_data = load_speedup_data(json_path)
    
    print(f"   CPU configurations: {len(cpu_data)}")
    
    print("\n📝 Generating LaTeX tables...")
    generate_cpu_speedup_tables(cpu_data, output_file)
    
    print("\n✅ LaTeX generation complete!")
    print(f"\nTo compile:")
    print(f"  pdflatex {output_file}")
    print(f"  pdflatex {output_file}  # Run twice for references")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
