import json

# Results summary
old_results = {
    (800, 7): 126,
    (1600, 7): 161,
    (3200, 7): 314,
    (6400, 7): 899,
    (12800, 7): 3239
}

adaptive_results = {
    (800, 7): 57,
    (1600, 7): 114,
    (3200, 7): 328,
    (6400, 7): 1157,
    (12800, 7): 4489
}

print("="*80)
print("PERFORMANCE ANALYSIS: Adaptive Strategy Results")
print("="*80)
print()
print(f"{'Size':<12} {'OLD (ms)':<12} {'ADAPTIVE (ms)':<15} {'Speedup':<10} {'Status'}")
print("-"*80)

for size in [800, 1600, 3200, 6400, 12800]:
    old = old_results[(size, 7)]
    new = adaptive_results[(size, 7)]
    speedup = old / new
    status = "✅ FASTER" if speedup > 1.0 else "❌ SLOWER"
    pct = (speedup - 1) * 100 if speedup > 1 else -(1 - speedup) * 100
    
    print(f"{size}x{size:<6} {old:<12} {new:<15} {speedup:.2f}x ({pct:+.0f}%)  {status}")

print()
print("="*80)
print("DIAGNOSIS:")
print("="*80)
print()
print("✅ Small images (800-1600): 1.4-2.2× FASTER")
print("   → RGB fusion + reduced overhead = win!")
print()
print("⚠️  Medium images (3200): ~5% SLOWER")
print("   → Break-even point")
print()
print("❌ Large images (6400-12800): 29-39% SLOWER")
print("   → RGB fusion is HURTING performance!")
print()
print("ROOT CAUSE:")
print("-" * 80)
print("The RGB fused kernel has WORSE memory access patterns for large images:")
print()
print("1. INTERLEAVED LAYOUT: stride-3 access [R,G,B,R,G,B...] hurts coalescing")
print("2. CACHE PRESSURE: 3× data per block increases L1/L2 cache misses")
print("3. REGISTER PRESSURE: Each thread manages 3× registers (sum_r, sum_g, sum_b)")
print("4. MEMORY BANDWIDTH: Large images are bandwidth-bound, not launch-bound")
print()
print("="*80)
print("SOLUTION:")
print("="*80)
print("Use HYBRID approach based on image size:")
print()
print("  if (H * W < threshold):  # e.g., < 4M pixels (2000×2000)")
print("      use RGB_Fused()      # Minimize kernel launch overhead")
print("  else:")
print("      use RGB_Separate()   # Maximize memory coalescing & occupancy")
print()
print("Expected improvement: 2× speedup on small, old performance on large")
print("="*80)
