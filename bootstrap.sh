#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] start"

# Set the pip cache dir to a writable location using the pip command
python3 -m pip config set global.cache-dir /workspace/.cache/pip-cache || true

# --------- headless + threads (good defaults on HPC) ----------
export MPLBACKEND=${MPLBACKEND:-Agg}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

# --------- resolve Python paths dynamically ----------
echo "[bootstrap] resolving package locations..."
PY_SITE=$(python3 - <<'PY'
import sysconfig; print(sysconfig.get_paths()["purelib"])
PY
)
MMENGINE_DIR=$(python3 - <<'PY'
import mmengine, os; print(os.path.dirname(mmengine.__file__))
PY
)
MMDET3D_TRANSFORMS_DIR=$(python3 - <<'PY'
import mmdet3d
from pathlib import Path
print(Path(mmdet3d.__file__).parent.joinpath("datasets","transforms"))
PY
)
echo "  site-packages: $PY_SITE"
echo "  mmengine dir : $MMENGINE_DIR"
echo "  mmdet3d/transforms: $MMDET3D_TRANSFORMS_DIR"


# --------- 1) torch-points-kernels health check ----------
echo "[bootstrap] checking torch_points_kernels..."
if ! python3 - <<'PY'
try:
    from torch_points_kernels import instance_iou
    print("tpk:OK")
except Exception as e:
    print("tpk:ERR", e); raise
PY
then
  echo "[bootstrap] repairing torch_points_kernels..."
  python3 -m pip uninstall -y torch-points-kernels || true
  # Use no-build-isolation so it compiles against the current torch/numpy/pybind11
  CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0} \
  FORCE_CUDA=1 \
  python3 -m pip install --no-deps --no-cache-dir --no-build-isolation torch-points-kernels==0.7.0
fi

# --------- 2) torch-cluster CUDA check and repair ----------
echo "[bootstrap] checking torch-cluster CUDA..."
if ! python3 - <<'PY'
import sys, torch
try:
    import torch_cluster
    if torch.cuda.is_available():
        import torch
        import torch_cluster
        import torch as T
        x = T.rand(16, 3, device='cuda')
        b = T.zeros(16, dtype=T.long, device='cuda')
        # will raise if CPU-only build
        torch_cluster.fps(x, b, ratio=0.5)
    print("cluster:OK")
except Exception as e:
    print("cluster:ERR", e); raise
PY
then
  echo "[bootstrap] repairing torch-cluster with matching PyG wheel..."
  python3 -m pip uninstall -y torch-cluster || true
  TORCH_VER=$(python3 - <<'PY'
import torch; print(torch.__version__)
PY
)
  # PyG index is keyed by full torch tag e.g. 1.13.1+cu117
  PYGI_URL="https://data.pyg.org/whl/torch-${TORCH_VER}.html"
  echo "  using PyG index: ${PYGI_URL}"
  python3 -m pip install -U --no-cache-dir -f "${PYGI_URL}" torch-cluster
fi

# --------- 3) copy your patched files (path-safe) ----------
echo "[bootstrap] applying local overrides..."
if [ -f /workspace/replace_mmdetection_files/loops.py ]; then
  install -D -m 0644 /workspace/replace_mmdetection_files/loops.py \
    "${MMENGINE_DIR}/runner/loops.py"
fi
if [ -f /workspace/replace_mmdetection_files/base_model.py ]; then
  install -D -m 0644 /workspace/replace_mmdetection_files/base_model.py \
    "${MMENGINE_DIR}/model/base_model/base_model.py"
fi
if [ -f /workspace/replace_mmdetection_files/transforms_3d.py ]; then
  mkdir -p "${MMDET3D_TRANSFORMS_DIR}"
  install -D -m 0644 /workspace/replace_mmdetection_files/transforms_3d.py \
    "${MMDET3D_TRANSFORMS_DIR}/transforms_3d.py"
fi

# --------- 4) segmentator symlink ----------
echo "[bootstrap] ensuring segmentator build symlink..."
mkdir -p /workspace/segmentator/csrc
ln -sfn /segmentator/csrc/build /workspace/segmentator/csrc/build

echo "[bootstrap] done; running command: $*"
exec "$@"
