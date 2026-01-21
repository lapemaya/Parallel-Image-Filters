#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail

# Benchmarks ONLY (implementation-reported times):
# - Python: seconds returned by <module>.apply_convolution_timed()
# - CUDA:   integer milliseconds printed as CUDA_CONVOLVE_RGB_OPT_MS=<ms>
# Output: CSV to stdout (and optional OUT_CSV file)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer active environment's python.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

# Default to ConvSeq (no multiprocessing noise).
PY_MODULES=( "ConvolutionScripts.ConvParallelAdvancedDumb" "ConvolutionScripts.ConvSeqDumb" )  # Test optimized and classic implementations
CUDA_EXE="${CUDA_EXE:-$ROOT_DIR/build/CudaConv}"
OUT_CSV="${OUT_CSV:-}"  # if set, also write CSV to file
OUT_JSON="${OUT_JSON:-benchmark_results.json}"  # JSON output file
SILENT_CSV="${SILENT_CSV:-false}"  # if true, don't print CSV to stdout

SIZES=(800 1600 3200 6400 8000)
KERNELS=(3 5 7)
N_JOBS_LIST=(4 8 12 16 20)  # Number of processes for parallel implementations
TILE_SIZES=(8 16 24 32)  # Tile sizes for CUDA (reduced to avoid configuration errors)

if [[ "${1:-}" == "--quick" ]]; then
  SIZES=(50)
  N_JOBS_LIST=(4 8 12 16 20)
  TILE_SIZES=(8 16 24) 
fi

mkdir -p "$ROOT_DIR/benchmark_images"

gen_image() {
  local size="$1"
  local path="$ROOT_DIR/benchmark_images/test_${size}x${size}.png"
  if [[ -f "$path" ]]; then
    echo "$path"
    return 0
  fi
  "$PYTHON_BIN" - <<PY
import numpy as np
from PIL import Image
size = int("$size")
img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
Image.fromarray(img).save("$path")
print("$path")
PY
}

extract_cuda_ms() {
  local text="$1"
  local ms
  ms="$(printf '%s\n' "$text" | sed -n 's/^CUDA_CONVOLVE_RGB_OPT_MS=\([0-9][0-9]*\)$/\1/p' | head -n1)"
  [[ -n "$ms" ]] || return 1
  printf '%s' "$ms"
}

python_time_s() {
  local img_path="$1"
  local k="$2"
  local module="$3"
  local n_jobs="${4:-1}"  # Default to 1 if not specified
  
  local tmpscript=$(mktemp --suffix=.py)
  trap "rm -f '$tmpscript'" RETURN

  cat >"$tmpscript" <<'PY'
import sys
import os
import numpy as np
from PIL import Image
import importlib
import multiprocessing

if __name__ == '__main__':
    # Set multiprocessing start method to avoid ChildProcessError
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set

    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())

    module_name = sys.argv[1]
    img_path = sys.argv[2]
    k = int(sys.argv[3])
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    mod = importlib.import_module(module_name)
    if not hasattr(mod, "apply_convolution_timed"):
        raise SystemExit(f"apply_convolution_timed not found in module: {module_name}")

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    if k == 3:
        kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float)
    elif k == 5:
        kernel = np.array([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=float)
    elif k == 7:
        kernel = np.array([
            [1,  6,  15,  20,  15,  6, 1],
            [6,  36,  90, 120,  90, 36, 6],
            [15, 90, 225, 300, 225, 90, 15],
            [20,120, 300, 400, 300,120, 20],
            [15, 90, 225, 300, 225, 90, 15],
            [6,  36,  90, 120,  90, 36, 6],
            [1,  6,  15,  20,  15,  6, 1]
        ], dtype=float)
    else:
        raise SystemExit("K must be 3, 5, or 7")

    # warm-up (discard)
    _res, _dt = mod.apply_convolution_timed(img, kernel, normalize=True, n_jobs=n_jobs)

    best = None
    for _ in range(3):
        _res, dt = mod.apply_convolution_timed(img, kernel, normalize=True, n_jobs=n_jobs)
        best = dt if best is None else min(best, dt)

    print(f"{best:.9f}")
PY

  "$PYTHON_BIN" "$tmpscript" "$module" "$img_path" "$k" "$n_jobs"
}

cuda_time_ms() {
  local img_path="$1"
  local k="$2"
  local tile_size="${3:-16}"  # Default to 16 if not specified

  if [[ ! -x "$CUDA_EXE" ]]; then
    echo "ERROR: CUDA executable not found/executable at $CUDA_EXE" >&2
    return 2
  fi

  # warm-up
  "$CUDA_EXE" "$img_path" "$k" "$tile_size" >/dev/null

  local best_ms=""
  local out
  local ms
  for _ in 1 2 3; do
    out="$("$CUDA_EXE" "$img_path" "$k" "$tile_size")"
    ms="$(extract_cuda_ms "$out")" || {
      echo "ERROR: Could not parse CUDA_CONVOLVE_RGB_OPT_MS from output:" >&2
      echo "$out" >&2
      return 3
    }
    if [[ -z "$best_ms" || "$ms" -lt "$best_ms" ]]; then
      best_ms="$ms"
    fi
  done
  printf '%s' "$best_ms"
}

emit() {
  local line="$1"
  if [[ "$SILENT_CSV" != "true" ]]; then
    echo "$line"
  fi
  if [[ -n "$OUT_CSV" ]]; then
    echo "$line" >>"$OUT_CSV"
  fi
}
if [[ -n "$OUT_CSV" ]]; then
  : >"$OUT_CSV"
fi

emit "size,kernel,py_module,n_jobs,tile_size,python_seconds,cuda_ms"

# Collect results in an array for JSON conversion
declare -a RESULTS=()

for size in "${SIZES[@]}"; do
  img_path="$(gen_image "$size")"
  for k in "${KERNELS[@]}"; do
    # Test Python implementations with different process counts
    for py_mod in "${PY_MODULES[@]}"; do
      if [[ "$py_mod" == *"Parallel"* ]]; then
        # Test parallel implementations with different n_jobs
        for n_jobs in "${N_JOBS_LIST[@]}"; do
          py_s="$(python_time_s "$img_path" "$k" "$py_mod" "$n_jobs")"
          emit "${size},${k},${py_mod},${n_jobs},-,${py_s},-"
          RESULTS+=("$size,$k,$py_mod,$n_jobs,-1,$py_s,-1")
        done
      else
        # Sequential implementations: test once with n_jobs=1
        py_s="$(python_time_s "$img_path" "$k" "$py_mod" "1")"
        emit "${size},${k},${py_mod},1,-,${py_s},-"
        RESULTS+=("$size,$k,$py_mod,1,-1,$py_s,-1")
      fi
    done
    
    # Test CUDA with different tile sizes
    for tile_size in "${TILE_SIZES[@]}"; do
      cuda_ms="$(cuda_time_ms "$img_path" "$k" "$tile_size")"
      emit "${size},${k},CUDA,1,${tile_size},-,${cuda_ms}"
      RESULTS+=("$size,$k,CUDA,1,$tile_size,-1,$cuda_ms")
    done
  done
done

# Convert results to JSON
"$PYTHON_BIN" - <<PY
import json
import sys
from datetime import datetime

results = []
for line in """${RESULTS[*]}""".split():
    if not line.strip():
        continue
    parts = line.split(',')
    if len(parts) == 7:
        results.append({
            "image_size": int(parts[0]),
            "kernel_size": int(parts[1]),
            "py_module": parts[2],
            "n_jobs": int(parts[3]),
            "tile_size": int(parts[4]),
            "python_seconds": float(parts[5]),
            "cuda_ms": int(parts[6])
        })

output = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "py_modules": "${PY_MODULES[*]}".split(),
        "image_sizes": [int(s) for s in "${SIZES[*]}".split()],
        "kernel_sizes": [int(k) for k in "${KERNELS[*]}".split()],
        "n_jobs_list": [int(n) for n in "${N_JOBS_LIST[*]}".split()],
        "tile_sizes": [int(t) for t in "${TILE_SIZES[*]}".split()]
    },
    "results": results
}

with open("$OUT_JSON", "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: $OUT_JSON", file=sys.stderr)
PY
