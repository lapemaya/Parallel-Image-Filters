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
PY_MODULES=( "ConvSeqDumb" "ConvParallelAdvancedDumb")  # Test optimized and classic implementations
CUDA_EXE="${CUDA_EXE:-$ROOT_DIR/build/CudaConv}"
OUT_CSV="${OUT_CSV:-}"  # if set, also write CSV to file
OUT_JSON="${OUT_JSON:-benchmark_results.json}"  # JSON output file

SIZES=(800 1600 3200 6400 8000)
KERNELS=(3 5 7)

if [[ "${1:-}" == "--quick" ]]; then
  SIZES=(50)
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
  
  local tmpscript=$(mktemp --suffix=.py)
  trap "rm -f '$tmpscript'" RETURN

  cat >"$tmpscript" <<'PY'
import sys
import os
import numpy as np
from PIL import Image
import importlib

if __name__ == '__main__':
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())

    module_name = sys.argv[1]
    img_path = sys.argv[2]
    k = int(sys.argv[3])

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
    _res, _dt = mod.apply_convolution_timed(img, kernel, normalize=True)

    best = None
    for _ in range(3):
        _res, dt = mod.apply_convolution_timed(img, kernel, normalize=True)
        best = dt if best is None else min(best, dt)

    print(f"{best:.9f}")
PY

  "$PYTHON_BIN" "$tmpscript" "$module" "$img_path" "$k"
}

cuda_time_ms() {
  local img_path="$1"
  local k="$2"

  if [[ ! -x "$CUDA_EXE" ]]; then
    echo "ERROR: CUDA executable not found/executable at $CUDA_EXE" >&2
    return 2
  fi

  # warm-up
  "$CUDA_EXE" "$img_path" "$k" >/dev/null

  local best_ms=""
  local out
  local ms
  for _ in 1 2 3; do
    out="$("$CUDA_EXE" "$img_path" "$k")"
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
  echo "$line"
  if [[ -n "$OUT_CSV" ]]; then
    echo "$line" >>"$OUT_CSV"
  fi
}
if [[ -n "$OUT_CSV" ]]; then
  : >"$OUT_CSV"
fi

emit "size,kernel,py_module,python_seconds,cuda_ms"

# Collect results in an array for JSON conversion
declare -a RESULTS=()

for size in "${SIZES[@]}"; do
  img_path="$(gen_image "$size")"
  for k in "${KERNELS[@]}"; do
    # Test both Python implementations
    for py_mod in "${PY_MODULES[@]}"; do
      py_s="$(python_time_s "$img_path" "$k" "$py_mod")"
      emit "${size},${k},${py_mod},${py_s},-"
      RESULTS+=("$size,$k,$py_mod,$py_s,-1")
    done
    
    # Test CUDA once per size/kernel
    cuda_ms="$(cuda_time_ms "$img_path" "$k")"
    emit "${size},${k},CUDA,-,${cuda_ms}"
    RESULTS+=("$size,$k,CUDA,-1,$cuda_ms")
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
    if len(parts) == 5:
        results.append({
            "image_size": int(parts[0]),
            "kernel_size": int(parts[1]),
            "py_module": parts[2],
            "python_seconds": float(parts[3]),
            "cuda_ms": int(parts[4])
        })

output = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "py_modules": "${PY_MODULES[*]}".split(),
        "image_sizes": [int(s) for s in "${SIZES[*]}".split()],
        "kernel_sizes": [int(k) for k in "${KERNELS[*]}".split()]
    },
    "results": results
}

with open("$OUT_JSON", "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: $OUT_JSON", file=sys.stderr)
PY
