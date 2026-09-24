#!/usr/bin/env bash
# Reproduce the PALADIN README install (github.com/kmboehm/paladin, "1. Install")
# into envs/paladin/.venv without cloning PALADIN into this repository.
#
# Usage (from anywhere):
#   bash envs/paladin/setup.sh                 # GPU build (torch 2.4.1+cu121), package from pinned git commit
#   PALADIN_DEVICE=cpu bash envs/paladin/setup.sh
#   PALADIN_SRC=/path/to/paladin bash envs/paladin/setup.sh   # editable install from your own checkout
#
# PALADIN's Hydra configs (conf/) and scripts/ live at the repository root, not in the
# installed package. Training or running `src/paladin/run.py --config-name ...` therefore
# needs a checkout; point PALADIN_SRC at it. Inference entry points (paladin.infer_paladin,
# paladin.infer_aeon) import from the installed package. Weights are MSK-internal.
set -euo pipefail

PALADIN_COMMIT="${PALADIN_COMMIT:-3686f49e8cfe709694905056ccb39f8c9ea9f95b}"  # main, 2026-09-24
PALADIN_DEVICE="${PALADIN_DEVICE:-cu121}"
ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ENV_DIR}/.venv"

case "${PALADIN_DEVICE}" in
  cu121) TORCH_SPEC=(torch==2.4.1+cu121 torchvision==0.19.1+cu121)
         TORCH_INDEX=https://download.pytorch.org/whl/cu121 ;;
  cpu)   TORCH_SPEC=(torch==2.4.1+cpu torchvision==0.19.1+cpu)
         TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
  *) echo "PALADIN_DEVICE must be cu121 or cpu (got ${PALADIN_DEVICE})" >&2; exit 2 ;;
esac

# Shared HPC: keep build temp files out of /tmp.
export TMPDIR="${TMPDIR:-${ENV_DIR}/.tmp_build}"
mkdir -p "${TMPDIR}"

# --seed installs pip/setuptools: lightning 2.3 resolves lightning.pytorch via pkg_resources.
uv venv --seed --python 3.11 "${VENV}"
export VIRTUAL_ENV="${VENV}"

uv pip install "${TORCH_SPEC[@]}" --index-url "${TORCH_INDEX}"
uv pip install lightning==2.3.0 torchmetrics==1.4.2 numpy==1.26.4 "setuptools<81"
# nn-template-core pins lightning==2.0.*; PALADIN installs it without dependencies.
uv pip install --no-deps nn-template-core==0.4.0

if [[ -n "${PALADIN_SRC:-}" ]]; then
  uv pip install -e "${PALADIN_SRC}[dev]"
  echo "PALADIN installed editable from ${PALADIN_SRC} ($(git -C "${PALADIN_SRC}" rev-parse HEAD 2>/dev/null || echo 'unknown commit'))"
else
  uv pip install "paladin[dev] @ git+https://github.com/kmboehm/paladin@${PALADIN_COMMIT}"
  echo "PALADIN installed from git commit ${PALADIN_COMMIT}"
fi

"${VENV}/bin/python" - <<'PY'
import lightning, numpy, torch, torchmetrics
import paladin
print("paladin", getattr(paladin, "__version__", "?"), "| torch", torch.__version__,
      "| lightning", lightning.__version__, "| torchmetrics", torchmetrics.__version__,
      "| numpy", numpy.__version__, "| cuda", torch.cuda.is_available())
PY
rm -rf "${ENV_DIR}/.tmp_build"
