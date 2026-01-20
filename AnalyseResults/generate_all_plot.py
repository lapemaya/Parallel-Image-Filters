#!/usr/bin/env python3
"""
Generate publication-ready plots for the report:
CPU parallel scaling, CUDA scaling, global comparisons, and normalized metrics.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PATHS / I/O
# =============================================================================

current_dir = Path(__file__).parent
json_path = current_dir / "speedup_results.json"
output_dir = current_dir / "plots"
output_dir.mkdir(exist_ok=True)

try:
    with open(json_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: file not found: {json_path}")
    sys.exit(1)

# =============================================================================
# STYLE (Matplotlib-only)
# =============================================================================

plt.style.use("seaborn-v0_8-whitegrid")  # stable builtin style (no seaborn dependency)

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.figsize": (14, 8),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "legend.fontsize": 11,
})

# Color palettes (simple + consistent)
# Feel free to tweak these if you want a specific color scheme.
PALETTE_JOBS = plt.cm.viridis(np.linspace(0.15, 0.85, 5))     # 5 n_jobs values
PALETTE_TILES = plt.cm.magma(np.linspace(0.20, 0.80, 4))      # 4 tile sizes
PALETTE_KERNELS = plt.cm.tab10(np.linspace(0, 1, 3))          # 3 kernels
PALETTE_COMPARE = plt.cm.Set2(np.linspace(0, 1, 3))           # 3 implementations


def save_plot(filename: str) -> None:
    """Save current figure to disk (high quality)."""
    plt.tight_layout()
    out_path = output_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ {filename}")
    plt.close()


# =============================================================================
# Helpers
# =============================================================================

def _get_unique_sorted(items, key):
    return sorted({d[key] for d in items})


def _lookup_entry(items, **conds):
    return next((d for d in items if all(d.get(k) == v for k, v in conds.items())), None)


def _cuda_time_ms(entry):
    """
    Return CUDA time in ms.
    Supports either:
      - entry['time_seconds'] (seconds)
      - entry['cuda_ms'] (ms)
    """
    if entry is None:
        return np.nan
    if "time_seconds" in entry and entry["time_seconds"] is not None:
        return float(entry["time_seconds"]) * 1000.0
    if "cuda_ms" in entry and entry["cuda_ms"] is not None:
        return float(entry["cuda_ms"])
    return np.nan


# =============================================================================
# PLOT 1: CPU speedup (all kernels) — varying image size & n_jobs
# =============================================================================

def plot_cpu_speedup_all_kernels():
    """CPU speedup vs. image size for multiple n_jobs and kernels."""
    kernels = [3, 5, 7]
    job_counts = [4, 8, 12, 16, 20]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    cpu_items = data.get("cpu_parallel_speedup", [])
    if not cpu_items:
        print("WARNING: data['cpu_parallel_speedup'] is missing/empty.")
        plt.close(fig)
        return

    for idx, kernel in enumerate(kernels):
        ax = axes[idx]
        kernel_data = [d for d in cpu_items if d.get("kernel_size") == kernel]
        image_sizes = _get_unique_sorted(kernel_data, "image_size")

        for i, n_jobs in enumerate(job_counts):
            speedups = []
            for img_size in image_sizes:
                entry = _lookup_entry(kernel_data, image_size=img_size, n_jobs=n_jobs)
                speedups.append(entry["speedup"] if entry else np.nan)

            ax.plot(
                image_sizes, speedups,
                marker="o",
                label=f"{n_jobs} processes",
                color=PALETTE_JOBS[i % len(PALETTE_JOBS)]
            )

        ax.set_title(f"Kernel {kernel}×{kernel}", fontweight="bold")
        ax.set_xlabel("Image Size (N×N)")
        ax.set_xticks(image_sizes)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Speedup vs. Sequential")
            ax.legend(title="Parallelism", frameon=True)

    fig.suptitle("CPU Parallel Speedup — Scaling with Image Size", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot("cpu_speedup_all_kernels.png")


# =============================================================================
# PLOT 2: CPU efficiency (kernel 3×3)
# =============================================================================

def plot_cpu_efficiency_k3():
    """CPU parallel efficiency (%) for kernel 3×3."""
    fig, ax = plt.subplots(figsize=(12, 7))

    cpu_items = data.get("cpu_parallel_speedup", [])
    kernel_data = [d for d in cpu_items if d.get("kernel_size") == 3]
    if not kernel_data:
        print("WARNING: no CPU entries for kernel=3.")
        plt.close(fig)
        return

    image_sizes = _get_unique_sorted(kernel_data, "image_size")
    job_counts = [4, 8, 12, 16, 20]

    for i, n_jobs in enumerate(job_counts):
        efficiencies = []
        for img_size in image_sizes:
            entry = _lookup_entry(kernel_data, image_size=img_size, n_jobs=n_jobs)
            # entry['efficiency'] expected in [0,1]
            efficiencies.append((entry["efficiency"] * 100.0) if entry and "efficiency" in entry else np.nan)

        ax.plot(
            image_sizes, efficiencies,
            marker="D",
            label=f"{n_jobs} processes",
            color=PALETTE_JOBS[i % len(PALETTE_JOBS)],
            alpha=0.95
        )

    ax.axhline(100, linestyle="--", alpha=0.6, linewidth=1.8, label="Ideal Efficiency (100%)")
    ax.set_xlabel("Image Size (N×N)", fontweight="bold")
    ax.set_ylabel("Parallel Efficiency (%)", fontweight="bold")
    ax.set_title("CPU Parallel Efficiency — Kernel 3×3", fontweight="bold", fontsize=16)
    ax.set_xticks(image_sizes)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Parallelism", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    save_plot("cpu_efficiency_k3.png")


# =============================================================================
# PLOT 3: CPU strong scaling (fixed image size = 3200×3200)
# =============================================================================

def plot_cpu_strong_scaling(img_size: int = 3200):
    """CPU strong scaling for a fixed image size."""
    fig, ax = plt.subplots(figsize=(12, 7))

    cpu_items = data.get("cpu_parallel_speedup", [])
    kernels = [3, 5, 7]
    job_counts = [4, 8, 12, 16, 20]

    for i, kernel in enumerate(kernels):
        speedups = []
        for n_jobs in job_counts:
            entry = _lookup_entry(cpu_items, image_size=img_size, kernel_size=kernel, n_jobs=n_jobs)
            speedups.append(entry["speedup"] if entry else np.nan)

        ax.plot(
            job_counts, speedups,
            marker="o",
            linewidth=3,
            label=f"Kernel {kernel}×{kernel}",
            color=PALETTE_KERNELS[i % len(PALETTE_KERNELS)]
        )

    # Ideal linear speedup (S=P)
    ax.plot(job_counts, job_counts, linestyle="--", alpha=0.7, linewidth=2, label="Ideal Linear Speedup")

    ax.set_xlabel("Number of Processes (n_jobs)")
    ax.set_ylabel("Speedup vs. Sequential")
    ax.set_title(f"CPU Strong Scaling (Image {img_size}×{img_size})", fontweight="bold", fontsize=16)
    ax.set_xticks(job_counts)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    save_plot("cpu_strong_scaling.png")


# =============================================================================
# PLOT 4: CPU efficiency heatmap (all kernels)
# =============================================================================

def plot_cpu_efficiency_heatmap():
    """Heatmap of CPU parallel efficiency (%) for all kernels."""
    cpu_items = data.get("cpu_parallel_speedup", [])
    if not cpu_items:
        print("WARNING: data['cpu_parallel_speedup'] is missing/empty.")
        return

    kernels = [3, 5, 7]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for idx, kernel in enumerate(kernels):
        ax = axes[idx]
        kernel_data = [d for d in cpu_items if d.get("kernel_size") == kernel]

        image_sizes = _get_unique_sorted(kernel_data, "image_size")
        job_counts = _get_unique_sorted(kernel_data, "n_jobs")

        mat = np.full((len(image_sizes), len(job_counts)), np.nan)
        for i, img_size in enumerate(image_sizes):
            for j, n_jobs in enumerate(job_counts):
                entry = _lookup_entry(kernel_data, image_size=img_size, n_jobs=n_jobs)
                if entry and "efficiency" in entry:
                    mat[i, j] = entry["efficiency"] * 100.0

        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100)

        ax.set_title(f"Kernel {kernel}×{kernel}", fontweight="bold")
        ax.set_xlabel("Number of Processes (n_jobs)")
        if idx == 0:
            ax.set_ylabel("Image Size (N×N)")

        ax.set_xticks(np.arange(len(job_counts)))
        ax.set_xticklabels(job_counts)
        ax.set_yticks(np.arange(len(image_sizes)))
        ax.set_yticklabels(image_sizes)

        # annotate
        for i in range(len(image_sizes)):
            for j in range(len(job_counts)):
                val = mat[i, j]
                txt = "—" if np.isnan(val) else f"{val:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)

        if idx == 2:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Efficiency (%)")

    fig.suptitle("CPU Parallel Efficiency (%) — Heatmap", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot("cpu_efficiency_heatmap.png")


# =============================================================================
# PLOT 5: CUDA speedup (all kernels) — NO log scale
# =============================================================================

def plot_cuda_speedup_all_kernels():
    """CUDA speedup vs. image size for multiple tile sizes and kernels (3, 5, 7)."""
    cuda_items = data.get("cuda_speedup", [])
    if not cuda_items:
        print("WARNING: data['cuda_speedup'] is missing/empty.")
        return

    kernels = [3, 5, 7]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

    for idx, kernel in enumerate(kernels):
        ax = axes[idx]
        kernel_data = [d for d in cuda_items if d.get("kernel_size") == kernel]
        image_sizes = _get_unique_sorted(kernel_data, "image_size")
        tile_sizes = _get_unique_sorted(kernel_data, "tile_size")

        for i, tile in enumerate(tile_sizes):
            speedups = []
            for img_size in image_sizes:
                entry = _lookup_entry(kernel_data, image_size=img_size, tile_size=tile)
                speedups.append(entry["speedup"] if entry else np.nan)

            ax.plot(
                image_sizes, speedups,
                marker="o",
                linewidth=2.5,
                label=f"Tile {tile}×{tile}",
                color=PALETTE_TILES[i % len(PALETTE_TILES)]
            )

        ax.set_title(f"Kernel {kernel}×{kernel}", fontweight="bold")
        ax.set_xlabel("Image Size (N×N)")
        ax.set_xticks(image_sizes)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Speedup vs. Sequential")
            ax.legend(title="Tile Size", frameon=True)

    fig.suptitle("CUDA GPU — Speedup vs. Tile Size and Kernel", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot("cuda_speedup_all_kernels.png")


# =============================================================================
# PLOT 6: CUDA execution time (ms) — log scale makes sense here
# =============================================================================

def plot_cuda_execution_time():
    """CUDA execution time (ms) vs. image size for multiple tiles and kernels."""
    cuda_items = data.get("cuda_speedup", [])
    if not cuda_items:
        print("WARNING: data['cuda_speedup'] is missing/empty.")
        return

    kernels = [3, 5, 7]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for idx, kernel in enumerate(kernels):
        ax = axes[idx]
        kernel_data = [d for d in cuda_items if d.get("kernel_size") == kernel]
        image_sizes = _get_unique_sorted(kernel_data, "image_size")
        tile_sizes = _get_unique_sorted(kernel_data, "tile_size")

        for i, tile in enumerate(tile_sizes):
            times_ms = []
            for img_size in image_sizes:
                entry = _lookup_entry(kernel_data, image_size=img_size, tile_size=tile)
                times_ms.append(_cuda_time_ms(entry))

            ax.plot(
                image_sizes, times_ms,
                marker="^",
                label=f"Tile {tile}×{tile}",
                color=PALETTE_TILES[i % len(PALETTE_TILES)]
            )

        ax.set_title(f"Kernel {kernel}×{kernel}", fontweight="bold")
        ax.set_xlabel("Image Size (N×N)")
        ax.set_xticks(image_sizes)
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Execution Time (ms, log scale)")
            ax.legend(title="Tile Size", frameon=True)

    fig.suptitle("CUDA Execution Time — Scaling Analysis", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot("cuda_execution_time.png")


# =============================================================================
# PLOT 7: Tile sensitivity summary (bars + line)
# =============================================================================

def plot_tile_sensitivity():
    """Summary: average speedup and tile sensitivity per kernel."""
    kernels = ["3×3", "5×5", "7×7"]
    avg_speedup = [376.1, 378.9, 377.5]
    tile_sensitivity = [26.4, 25.9, 22.1]

    x = np.arange(len(kernels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        x - width / 2,
        avg_speedup,
        width,
        color=PALETTE_KERNELS,
        alpha=0.85,
        label="Average Speedup"
    )
    ax1.set_ylabel("Average Speedup (×)", fontweight="bold")
    ax1.set_xlabel("Kernel Size", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(kernels)
    ax1.set_title("CUDA Performance Sensitivity to Tile Size", fontweight="bold", fontsize=16)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        tile_sensitivity,
        color="black",
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Performance Variation (%)"
    )
    ax2.set_ylabel("Performance Variation (%)", fontweight="bold")

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    save_plot("cuda_tile_sensitivity.png")


# =============================================================================
# PLOT 8: Time per MAC — CPU vs CUDA (log y is reasonable)
# =============================================================================

def plot_time_per_mac_comparison():
    """Normalized compute efficiency: time per MAC (ns) for CPU vs CUDA."""
    kernels = [3, 5, 7]
    image_sizes = [800, 1600, 3200, 6400, 8000]

    # From your report tables (kept as-is)
    cpu_data = {
        3: [36.35, 44.22, 44.63, 44.30, 44.24],
        5: [13.58, 16.23, 16.20, 16.39, 16.06],
        7: [7.16, 8.42, 8.31, 8.40, 8.33],
    }
    cuda_data = {
        3: [0.521, 0.564, 0.579, 0.582, 0.581],
        5: [0.188, 0.203, 0.210, 0.210, 0.209],
        7: [0.096, 0.106, 0.108, 0.108, 0.108],
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, k in enumerate(kernels):
        ax.plot(
            image_sizes, cpu_data[k],
            marker="o",
            linewidth=2.5,
            label=f"CPU K={k}×{k}",
            color=PALETTE_KERNELS[i],
            alpha=0.95
        )

    for i, k in enumerate(kernels):
        ax.plot(
            image_sizes, cuda_data[k],
            marker="s",
            linewidth=2.5,
            linestyle="--",
            label=f"CUDA K={k}×{k}",
            color=PALETTE_KERNELS[i],
            alpha=0.95
        )

    ax.set_xlabel("Image Size (N×N)", fontweight="bold")
    ax.set_ylabel("Time per MAC (ns, log scale)", fontweight="bold")
    ax.set_title("Compute Efficiency: Normalized Time per MAC", fontweight="bold", fontsize=16)
    ax.set_xticks(image_sizes)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(frameon=True, ncol=2)

    save_plot("time_per_mac_cpu_vs_cuda.png")


# =============================================================================
# PLOT 9: Global execution time comparison (log y makes sense)
# =============================================================================

def plot_global_comparison_bar():
    """Global time comparison: Sequential vs CPU vs CUDA (log y)."""
    configs = ["800×800\nK=3", "3200×3200\nK=5", "8000×8000\nK=7"]
    sequential_times = [3.82, 60.80, 369.10]
    cpu_times = [0.63, 12.43, 76.40]
    cuda_times = [0.009, 0.161, 1.004]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    r1 = ax.bar(x - width, sequential_times, width, label="Sequential", color=PALETTE_COMPARE[0], alpha=0.85)
    r2 = ax.bar(x, cpu_times, width, label="CPU Parallel (best)", color=PALETTE_COMPARE[1], alpha=0.85)
    r3 = ax.bar(x + width, cuda_times, width, label="CUDA GPU (best)", color=PALETTE_COMPARE[2], alpha=0.85)

    ax.bar_label(r1, fmt="%.2fs", padding=3, fontsize=9)
    ax.bar_label(r2, fmt="%.2fs", padding=3, fontsize=9)
    ax.bar_label(r3, fmt="%.3fs", padding=3, fontsize=9)

    ax.set_xlabel("Configuration (Image Size and Kernel)", fontweight="bold")
    ax.set_ylabel("Execution Time (s, log scale)", fontweight="bold")
    ax.set_title("Global Performance Comparison", fontweight="bold", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(frameon=True)

    save_plot("global_comparison_bar.png")


# =============================================================================
# PLOT 10: Speedup summary (CPU vs CUDA)
# =============================================================================

def plot_speedup_summary():
    """Summary of max speedups (CPU vs CUDA) for a few representative configurations."""
    categories = ["800×800\nK=3", "3200×3200\nK=5", "8000×8000\nK=7"]
    cpu_speedup = [6.09, 4.89, 4.77]
    cuda_speedup = [424.9, 377.7, 370.0]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    b1 = ax.bar(x - width / 2, cpu_speedup, width, label="CPU Parallel (max)", color=PALETTE_COMPARE[1], alpha=0.85)
    b2 = ax.bar(x + width / 2, cuda_speedup, width, label="CUDA GPU (max)", color=PALETTE_COMPARE[2], alpha=0.85)

    ax.bar_label(b1, fmt="%.1f×", padding=3)
    ax.bar_label(b2, fmt="%.0f×", padding=3)

    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_ylabel("Max Speedup vs. Sequential", fontweight="bold")
    ax.set_title("Maximum Achieved Speedups", fontweight="bold", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=True)

    save_plot("speedup_summary.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Generating plots...\n")

    print("CPU analysis:")
    plot_cpu_speedup_all_kernels()
    plot_cpu_efficiency_k3()
    plot_cpu_strong_scaling(img_size=3200)
    plot_cpu_efficiency_heatmap()

    print("\nCUDA analysis:")
    plot_cuda_speedup_all_kernels()
    plot_cuda_execution_time()
    plot_tile_sensitivity()

    print("\nComparisons and normalized metrics:")
    plot_time_per_mac_comparison()
    plot_global_comparison_bar()
    plot_speedup_summary()
