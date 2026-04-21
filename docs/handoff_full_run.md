# Handoff — Full Extraction → Aggregation → Training Run

**Status at handoff:** branch `lazyslide-refactor` @ `3f96ee4` on GitHub. All
validation tests pass (static + pipeline + model registry). All 11 models used
in `scripts/extract.sh` verified to load. Ready for full-cohort run on GPUs.

**You need:** GPU access on the Whitehead Slurm cluster. The repo already
ran on CPU for testing; GPUs are required for feature extraction at cohort
scale.

**You must NOT:** commit, push, change branches, modify .env, request
access to new HF-gated models, or delete anything outside your own logs.
Return control if you hit something not covered here.

---

## TL;DR execution path

```bash
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
git status                        # expect: lazyslide-refactor, clean or just CLAUDE.md
eval "$(conda shell.bash hook)" && conda activate argo

argo env                          # sanity: HF_HOME on /lab, HF_TOKEN set
argo models --check --non-gated-only    # quick availability smoke

# Launch the full pipeline (SLURM):
sbatch scripts/extract.sh         # ~7 days walltime, 2-group array
# ... wait for extraction to finish, then:
sbatch scripts/aggregate.sh       # model-parallel, ~1 hr
sbatch scripts/train.sh           # model-parallel, ~30 min
```

Full details below.

---

## 0. Environment sanity

Run these first. Stop if anything looks wrong.

```bash
cd /lab/barcheese01/mdiberna/ARGO-DeepMSI
git rev-parse --abbrev-ref HEAD     # must be: lazyslide-refactor
git log -1 --oneline                # latest should be 3f96ee4 or newer
eval "$(conda shell.bash hook)" && conda activate argo

# These come from argo_deepmsi/__init__.py auto-defaults
argo env
# Expect:
#   HF_HOME=/lab/barcheese01/mdiberna/ARGO-DeepMSI/.huggingface_cache  (~34TB free)
#   HF_TOKEN=(set)

# Slide + clinical tables from prior ingest run
ls -la results/data/slide_table.csv results/data/clinical_table.csv
wc -l results/data/slide_table.csv   # expect ~800 slides
```

If `HF_TOKEN=(unset)` even after activating the env, `.env` isn't being read.
Check that `/lab/barcheese01/mdiberna/ARGO-DeepMSI/.env` exists and has a
`HF_TOKEN=hf_...` line. Do NOT paste a token anywhere other than this file.

If the slide table is missing, stop and ping the user — don't re-run `argo
ingest` (it pulls from REDCap, you shouldn't touch that).

---

## 1. Pre-flight: model availability

Optional but cheap. Verifies every model in `scripts/extract.sh` can be
instantiated before you burn SLURM time.

```bash
argo models --check
```

Expected, from this session's audit:
- **22 ok** — all 11 models actively used in `scripts/extract.sh`, plus QC
  models we're not using yet.
- **10 fail** — all superseded-by-newer-version (uni, virchow, h0-mini,
  hibou-l) or optional/not-in-extract.sh (medsiglip, path_orchestra, histoplus,
  rosie, omiclip, conch). These failures are expected and do NOT block the run.

If any of the 11 active-extract models fail, stop and hand back — that's a
regression.

---

## 2. Full extraction

```bash
sbatch scripts/extract.sh
```

What happens:
- 2-group SLURM array: `extract_%A_0.out` + `extract_%A_1.out`
- Partition: `nvidia-A6000-20` (80GB VRAM), 128G RAM, 16 CPUs, 168h walltime
- Each group processes ~half the slide_table rows with **all 11 models per
  slide** (`uni2, virchow2, conch_v1.5, h-optimus-1, gigapath, hibou-b, musk,
  chief, ctranspath, phikonv2, plip`)
- Output: a `<slide>.zarr` next to each SVS, with `<model>_tiles` AnnData tables
- Per-slide, per-model failures are caught and logged (`extract_features_
  single_slide` returns `None` on exception); they do NOT kill the group.
- Extraction is incremental: if a zarr already has some models extracted, only
  the missing ones run. Rerunning the script is safe.

### Monitoring

```bash
squeue -u $USER                       # jobs in queue / running
tail -f scripts/logs/extract_<jobid>_0.out   # live progress
ls -la /lab/barcheese01/.../*.zarr | wc -l    # zarrs produced so far
```

### First-run cost

HF caches are populated from prior CPU testing for most models, but some
gated weights may still need to download. All downloads land on
`/lab/barcheese01/.../.huggingface_cache/hub/` (~26 GB already there). Don't
worry if the first slides are slow — subsequent slides reuse the cached models.

### Known issues / expected outcomes

- **musk**: has `sentencepiece` as a runtime dep (added in `3f96ee4`) and
  needs `huggingface_hub.login()` via our package init hook. Both are
  in place. If musk extraction fails at runtime, check
  `scripts/logs/extract_*_*.out` for the root cause — don't assume gating.
- **PLIP**: had sporadic failures historically. Individual slide failures
  are fine; a systemic failure warrants investigation.
- **7-day walltime cap**: if not done by then, the script is rerunable
  thanks to incremental extraction — just `sbatch scripts/extract.sh`
  again and it'll pick up where it left off.

---

## 3. Aggregation

After extraction, aggregate patch features to slide-level embeddings.

```bash
sbatch scripts/aggregate.sh
```

What happens:
- Array `0-8%5` → 9 models × mean pooling, 5 concurrent
- Partition `short`, no GPU, 32G RAM, 4h walltime per task
- Output: `results/embeddings/<model>_mean/` containing:
  - `embeddings.npy` — (n_slides × n_features) float32
  - `metadata.csv` — slide_id, patient_id, site, n_tiles, zarr_path
  - `embeddings.h5ad` — AnnData for scverse interop

Monitoring:
```bash
squeue -u $USER
ls results/embeddings/
# Each dir should have all three files
for d in results/embeddings/*/; do
    echo "=== $d ==="
    ls -la "$d"
done
```

### Neural encoders (optional, skip for this handoff)

`prism`, `titan`, `chief`, `madeleine` are neural slide encoders that
produce alternative slide embeddings. They're NOT in `aggregate.sh` by
default. If the user wants them, they need to be run manually — leave that
decision for the next pass.

---

## 4. Training

After aggregation, train classifiers on each model's embeddings.

```bash
sbatch scripts/train.sh
```

What happens:
- Array `0-11%5` → 12 embedding dirs × (LogReg, RandomForest, SVM), 5 concurrent
- Partition `short`, no GPU, 16G RAM, 2h walltime per task
- **Critical:** uses `StratifiedGroupKFold(groups=patient_id)` — prevents
  leakage when a patient has multiple slides. This is enforced in code; don't
  let it silently degrade to slide-level CV.
- Output: `results/models/<embedding_dir>/classifier_comparison.csv` with
  mean/std AUROC per classifier, plus `training_data.csv`.

Monitoring:
```bash
squeue -u $USER
cat results/models/*/classifier_comparison.csv
```

Best AUROC per model is the headline result.

---

## 5. Results summary

After everything finishes, produce a summary:

```bash
# Per-model best AUROC across classifiers
for f in results/models/*/classifier_comparison.csv; do
    model=$(basename $(dirname $f))
    best=$(python -c "import pandas as pd; d=pd.read_csv('$f'); r=d.sort_values('auroc_mean',ascending=False).iloc[0]; print(f\"{r['classifier']:25s} auroc={r['auroc_mean']:.3f}±{r['auroc_std']:.3f}\")")
    printf "%-20s  %s\n" "$model" "$best"
done | tee results/models/SUMMARY.txt
```

Save that `SUMMARY.txt` and hand it back to the user, along with pointers to:
- Any slides that failed extraction across multiple models (grep extract logs
  for `Failed to process`).
- Any aggregation or training tasks that errored.

---

## 6. If things go wrong

| Symptom | Action |
|---|---|
| `argo env` shows `HF_TOKEN=(unset)` | `.env` issue, stop and hand back |
| A specific model fails on every slide | Rerun `argo models --check` on the argo env; likely an HF/network issue |
| All slides OOM in extraction | Likely a new very-large model; drop its line from `scripts/extract.sh`'s `MODELS=()` and rerun |
| `sbatch` rejected by partition | `curl -s http://slurmstatus.wi.mit.edu/limits.html` to check limits |
| Aggregation empty for a model | Extraction didn't produce that model's table — check extract logs before rerunning aggregate |
| Training errors with `groups must be provided` | Bug regression; stop and hand back |

Do not:
- Delete zarrs to "start fresh" — extraction is incremental and safe to rerun.
- Commit anything.
- Change branches.
- Skip the StratifiedGroupKFold check in training. If the test suite regresses
  on that, hand it back.

---

## 7. Expected timeline

Rough, from cold-cache state:

| Stage | Walltime | Wall-clock with SLURM wait |
|---|---|---|
| Extraction | 2–5 days | plan for a week |
| Aggregation | ~30 min/model × 9 models / 5 concurrent ≈ 1 hr | ~2 hr with queuing |
| Training | ~10 min/embedding × 12 / 5 concurrent ≈ 30 min | ~1 hr with queuing |

Hand back to the user when:
1. All expected `<model>_mean/` embeddings exist.
2. All expected `results/models/*/classifier_comparison.csv` exist.
3. `SUMMARY.txt` is produced.

---

## 8. Context if you need it

- Branch `lazyslide-refactor` is the result of ~9 commits of correctness +
  efficiency + ecosystem-integration work against the pre-existing pipeline.
  See `docs/code_review.md`, `docs/efficiency_analysis.md`,
  `docs/lazyslide_gap_analysis.md`, `docs/lazyslide_reference_guide.md` for
  the analyses that drove it, and `docs/refactor_status.md` for a full
  cross-reference of every recommendation against what landed (plus what's
  explicitly deferred — none of it blocks the full run).
- Tests: `tests/test_lazyslide_api.py` (18 CPU), `tests/test_argo_pipeline.py`
  (12 end-to-end), `tests/test_model_registry.py` (6 static + 32 opt-in
  download). Run with `pytest tests/ -v` (static + pipeline only).
- HF cache + token are auto-configured by `argo_deepmsi/__init__.py` — no
  manual export needed when working from this checkout.
