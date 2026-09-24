# envs/paladin

PALADIN/Aeon runs on Mussel H-optimus-0 features (stage 4b in `docs/backends-plan.md`).
This environment is **not locked**: `setup.sh` replays the install section of the upstream
README (github.com/kmboehm/paladin) exactly, into `envs/paladin/.venv`:

| Step | Upstream README | `setup.sh` |
|---|---|---|
| venv | `uv venv --seed --python 3.11` | same, at `envs/paladin/.venv` |
| torch | `torch==2.4.1+cu121 torchvision==0.19.1+cu121` (cu121 index) | same (`PALADIN_DEVICE=cpu` → `+cpu`) |
| lightning | `lightning==2.3.0 torchmetrics==1.4.2 numpy==1.26.4 "setuptools<81"` | same |
| nn_core | `uv pip install --no-deps nn-template-core==0.4.0` | same |
| paladin | `uv pip install -e ".[dev]"` from a clone | `paladin[dev] @ git+https://github.com/kmboehm/paladin@<commit>` (default pin `3686f49e`), or editable from `PALADIN_SRC` |

```bash
bash envs/paladin/setup.sh                               # GPU (cu121), pinned git commit
PALADIN_SRC=/path/to/paladin bash envs/paladin/setup.sh  # editable from your checkout
PALADIN_COMMIT=<sha> bash envs/paladin/setup.sh          # different pin
```

Notes:
- PALADIN is a setuptools/setuptools-scm package (`src/paladin`), so the git install gives the
  importable package (`paladin.infer_paladin`, `paladin.infer_aeon`). Its Hydra configs
  (`conf/`) and `scripts/` sit at the repository root and are **not** installed: to train or
  run `src/paladin/run.py --config-name ...`, use a checkout outside this repository and set
  `PALADIN_SRC`.
- Model weights are MSK-internal; the upstream inference script hardcodes an internal
  checkpoint path. Nothing here downloads weights.
- Never clone PALADIN into this repository; `envs/*/.venv/` is gitignored.
