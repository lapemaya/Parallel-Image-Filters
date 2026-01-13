# Calculate theoretical occupancy for different configurations

def calc_occupancy(threads_per_block, shared_mem_per_block, max_threads=1024, max_shared=49152):
    """Calculate theoretical occupancy"""
    if threads_per_block > max_threads:
        return 0, "Exceeds max threads/block"
    if shared_mem_per_block > max_shared:
        return 0, "Exceeds max shared memory"
    
    # RTX 4060 Ti: max 1536 threads per SM, 48KB shared per SM
    max_blocks_threads = 1536 // threads_per_block
    max_blocks_shared = max_shared // shared_mem_per_block if shared_mem_per_block > 0 else 999
    max_blocks = min(max_blocks_threads, max_blocks_shared, 16)  # 16 blocks/SM limit
    
    active_threads = max_blocks * threads_per_block
    occupancy = active_threads / 1536.0
    return occupancy, f"{max_blocks} blocks/SM"

# OLD implementation (tileSize=16, separate channels)
K7_old_sharedDim = 16 + 7 - 1  # 22
K7_old_threads = 22 * 22  # 484
K7_old_shared = (22 + 1) * 22 * 4  # 1 channel: 2,024 bytes

print("="*70)
print("OLD IMPLEMENTATION (separate channel processing, tileSize=16)")
print("="*70)
print(f"Kernel K=7:")
print(f"  Threads/block: {K7_old_threads}")
print(f"  Shared mem: {K7_old_shared} bytes (per channel)")
occ, info = calc_occupancy(K7_old_threads, K7_old_shared)
print(f"  Occupancy: {occ*100:.1f}% ({info})")

# NEW implementation (RGB fused, tileSize=24)
K7_new_sharedDim = 24 + 7 - 1  # 30
K7_new_threads = 30 * 30  # 900
K7_new_shared = 3 * (30 + 1) * 30 * 4  # 3 channels: 11,160 bytes

print("\n" + "="*70)
print("NEW IMPLEMENTATION (RGB fused, tileSize=24)")
print("="*70)
print(f"Kernel K=7:")
print(f"  Threads/block: {K7_new_threads}")
print(f"  Shared mem: {K7_new_shared} bytes (all 3 channels)")
occ, info = calc_occupancy(K7_new_threads, K7_new_shared)
print(f"  Occupancy: {occ*100:.1f}% ({info})")

# Improved: RGB fused but with tileSize=16
K7_better_sharedDim = 16 + 7 - 1  # 22
K7_better_threads = 22 * 22  # 484
K7_better_shared = 3 * (22 + 1) * 22 * 4  # 3 channels: 6,072 bytes

print("\n" + "="*70)
print("PROPOSED: RGB FUSED + tileSize=16 (best of both)")
print("="*70)
print(f"Kernel K=7:")
print(f"  Threads/block: {K7_better_threads}")
print(f"  Shared mem: {K7_better_shared} bytes (all 3 channels)")
occ, info = calc_occupancy(K7_better_threads, K7_better_shared)
print(f"  Occupancy: {occ*100:.1f}% ({info})")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("For small images: RGB fused wins (reduces overhead)")
print("For large images: Need to balance occupancy vs kernel launches")
print("\nBest strategy: Use ADAPTIVE tileSize based on image size!")
print("  - Small images (<2000px): tileSize=24 (maximize per-launch work)")
print("  - Large images (≥2000px): tileSize=16 (maximize occupancy)")
