# MSK-cluster runbook — handoff ledger for the final-cohort run

> File name kept as `iris-runbook.md`; the target is whichever MSK cluster (CDSI or IRIS)
> hosts the Mosaic embeddings and code — to be confirmed by the collaborators.

Give this file (plus `AGENTS.md` and `CLAUDE.md`) to the Claude Code session on IRIS. It
states what to run, in what order, with which encoders, what is blocked on collaborators,
and what "done" means at each step. Evidence behind every choice:
[negative-results ledger](negative-results-ledger.md),
[IRIS-prep ledger](iris-prep-ledger.md), [domain-shift evidence](domain-shift-evidence.md),
[literature landscape](research/domain-shift-landscape.md).

## Kick-off prompt (paste into the IRIS session)

> Read AGENTS.md, CLAUDE.md and docs/iris-runbook.md. We are on the MSK cluster with the final
> Nigerian cohort. Execute the runbook phase by phase; stop at every gate marked 🔒 and
> report. Do not change the cohort, labels, or sealed-set membership without asking. Keep
> docs/iris-runbook.md's status table current as you go.

## Status

| Phase | State | Blocked on |
|---|---|---|
| P0 environment + reproduction | not started | cluster choice (CDSI vs IRIS) + data placement (HPC contact, Mosaic contact) |
| P1 data freeze v3 | not started | updated REDCap labels (end of September) |
| P2 extraction | not started | P0; H-optimus version + Mussel settings (PALADIN team); TITAN settings (Mosaic embedding contact) |
| P3 pretrained evaluation | not started | P2; PALADIN inference path (PALADIN team) |
| P4 site smoothing + domain-shift aim | not started | P2 |
| P5 learning beyond the reference | not started | P3 results |

### Pipeline stages and environments

Built on Whitehead 2026-09-24 (branch `backends`); every stage still has to be brought up on
the MSK cluster in P0.

| Stage | Env | Whitehead state | MSK cluster |
|---|---|---|---|
| 1. setup (`argo setup`, `argo envs`) | core | ✅ core + `envs/lazyslide` + `envs/mussel` synced; GPU smoke passed | not started |
| 2. slide tasks (ingest, pyramidal, QC, cohort freeze) | core | ✅ 808-slide cohort | not started (P1) |
| 3a. extract, LazySlide (default) | `envs/lazyslide` | ✅ existing zarrs; `--backend lazyslide` | not started (P2) |
| 3b. extract, Mussel | `envs/mussel` | ✅ smoke on 2 slides (H-optimus-0, TITAN); default preset, provisional | blocked on MSK engineering params |
| 4a. analyses | core | ✅ | not started (P3/P4) |
| 4b. PALADIN | `envs/paladin` | `setup.sh` ready; `argo paladin` is a stub | blocked on MSK weights / inference path |

## What depends on the email replies

| Question | Who | Changes |
|---|---|---|
| CDSI vs IRIS for this project; data-placement rules and backup for partner-site slides | IT/HPC contact | which cluster; TheSpot request if IRIS; where slides live |
| Which cluster/path hosts the Mosaic TITAN / H-optimus embeddings and code | Mosaic embedding contact | decides the cluster (co-locate with the MSK-side embeddings) |
| Where the Mosaic CRC TITAN embeddings are; their exact patching (CONCH v1.5 at 512 px, 20×? Trident or Mussel?); which patients | Mosaic embedding contact | P2 TITAN tiling generation; Aim-2 comparison cohort |
| PALADIN's encoder (H-optimus-0 vs -1), Mussel tiling (tile µm, mpp, tissue filter), CRC MSI sub-model availability, whether MSK runs inference on our features or shares weights, where the code/configs live internally | PALADIN developer, via the K43 postdoc | P2 H-optimus extraction settings; whether P3 includes PALADIN |
| Is the collaboration's primary representation TITAN (slide-level, CONCH) or H-optimus/PALADIN (tile-level, task-trained)? | PI + K43 postdoc | Which encoder is P2 priority 1 after CTransPath |
| Updated MSI labels + remaining prospective slides | Nigerian REDCap team | P1 freeze; size of the sealed temporal set |
| OAUTHC specimen type (biopsy/resection), fixation, sectioning, block selection | Nigerian team, via the K43 postdoc | explains the OAUTHC failure; stratification for P3/P4 |

Until answered, P2 extracts the answer-independent encoders first (CTransPath, Mascaret,
Phaet) and queues H-optimus and CONCH at the settings given below as provisional.

## Encoders to extract (and why)

Evidence from the 217-patient cohort decides the list; extraction is incremental per zarr.

| Priority | Encoder | Tiling | Why | Evidence |
|---|---|---|---|---|
| 1 | **CTransPath** | 256 px @ 0.5 mpp (Wagner's grid) | Input of the only model with reliable signal (Wagner 0.717) and of all site-smoothing experiments | ledger; `site_smoothing/` |
| 1 | **Mascaret** (Waiv) | 256 px @ 0.5 mpp | Only raw encoder whose embedding ignores the staining lab (restain FD 1.05× null vs 4–9× others) | `embed_shift.csv` |
| 1 | **Phaet** (Waiv) | 256 px @ 0.5 mpp | Best learned heads (nested MLP 0.632; nested LR race 0.619) | `dann_*`, B2 |
| 2 | **H-optimus-0** (confirm) | Mussel settings (20×, 112 µm tiles per preprint — confirm) | PALADIN input; strongest available pretrained CRC MSI model (internal 0.97 / external 0.93) | research landscape |
| 2 | **CONCH v1.5 → TITAN** | **512 px @ 20×** (TITAN's native patching) | Only for matching the MSK TITAN cohort (Aim 2). Our existing TITAN used 256 px tiles — non-canonical; do not compare it to MSK's | P1 (TITAN least stain-robust), nested 0.536 |
| — | UNI2, Virchow2, PRISM, CONCH mean | — | Drop. No nested advantage, largest site shift, costly. Re-add only if a collaborator needs them | `dann_*`, `embed_shift.csv` |

## Phases

### P0 — Environment and exact reproduction
0. Read the MSK cluster docs (cluster info, user-data policy, user docs; links in the local
   correspondence notes) for partition names, GPU limits, CUDA driver and HF/internet access.
   A user-level `uv` environment is fine.
1. `uv sync --frozen --extra dev` (core), then `uv run argo setup` (syncs `envs/lazyslide`
   and `envs/mussel` from their locks, reports HF token, `.env` keys, and data paths without
   printing values) and `uv run argo envs`. Add `--env paladin` only once PALADIN access is
   settled. Accept gated HF licences (CONCH/TITAN, UNI if needed, Waiv, H-optimus) under the
   cluster account.
2. Acceptance: `uv run ruff check argo_deepmsi tests scripts/extract_dask.py`,
   `uv run pytest -q`, `uv run --frozen --project envs/lazyslide pytest -q`,
   `uv run argo self-test`, `uv lock --check` for the root and for `--project envs/lazyslide`
   and `--project envs/mussel`, **and on a GPU node** the GPU smoke in each tool env:
   `uv run --frozen --project envs/<lazyslide|mussel> python -c "import torch; torch.zeros(1).cuda()"`
   plus `uv run --frozen --project envs/lazyslide pytest -m gpu -q`. Torch is pinned to the
   cu128 index (core, `envs/lazyslide`) and cu121 (`envs/mussel`); both need a CUDA ≥ 12.x
   driver (`nvidia-smi`). Only if that fails, repoint the `[tool.uv.sources]` torch index in
   the affected env project and re-lock it.
3. Transfer slides + existing zarrs through the transfer node (`data/` is ~531 GB, 965 image
   files incl. pyramidal conversions, zarrs alongside); write a checksum manifest.
4. `argo ingest` to regenerate slide tables with IRIS paths (tables store absolute paths).
5. 🔒 Reproduce Wagner on the 803 slides: max/√n 0.717 ± 0.005 and patient-mean 0.659,
   same 217 patients (`python -m argo_deepmsi.eval.domain_shift slidecount`).

### P1 — Data freeze v3
1. Pull labels (`scripts/audit_redcap_freshness.py`, `scripts/build_jhu_crc_pathology_crosswalk.py`).
2. `argo pyramidal` new slides (MPP stamping for MPP-less TIFFs); drop unreadable SVS.
3. 🔒 Freeze roles before any scoring: development = current 217; **sealed temporal set** =
   newly labelled patients; unlabelled slides = self-supervision pool. Keep `Indeterminate`
   as its own label state. Record in `results/data/cohort_manifest.json`.

### P2 — Extraction
`argo extract-dask` (one model per reopen, per-slide isolation; runs in `envs/lazyslide`) in
the priority order above. Done when every slide has every priority-1 table and
`uv run --frozen --project envs/lazyslide argo doctor --strict` passes.

**Mussel / PALADIN path.** H-optimus-0 for PALADIN is extracted with Mussel, not LazySlide:
`sbatch --array=0-N scripts/extract_mussel.sh <slide_table.csv>` (or
`argo extract <table> --backend mussel --model hoptimus0 --indices ...`) writes
`<slide_dir>/<stem>.mussel/OPTIMUS.features.{h5,pt}` + provenance. The config
`configs/backends/mussel-hoptimus0.toml` holds Mussel's default preset (224 px @ 0.5 mpp for
H-optimus-0) and is provisional; never use `seg_config=stain` (32 tiles/slide cap). Then
`argo paladin` in `envs/paladin` once MSK supplies weights or runs inference on our features.

**When MSK engineering sends the Mosaic/PALADIN extraction parameters** (seg preset, mpp,
patch size, overlap, tissue filter, Mussel version): put them in
`configs/backends/mussel-hoptimus0.toml` (and `mussel-titan.toml` if they cover TITAN), bump
the Mussel pin in `envs/mussel` only if they name a different version, re-extract with
`--overwrite` on the comparison slides, and rerun
`argo compare-backends --slides <table> --n 8 --model hoptimus0` (the X1 study,
`docs/experiments/X1-backend-equivalence.md`). Record the new verdict before extracting the
full cohort with Mussel.

### P3 — Pretrained evaluation (pre-registered)
1. Write the analysis plan (estimand, sites, operating point, comparisons) before scoring
   the sealed set.
2. Score development + sealed sets: Wagner (locked), PALADIN (if available), their rank
   average. Patient AUROC with bootstrap CI, spec @ sens 0.95, per site. **Primary pooling
   is slide-count-neutral (patient mean); report max/√n and a slide-count-only control
   alongside** — at OAUTHC slide count alone scored 0.626, equal to Wagner.
3. 🔒 Report before any fitting.

### P4 — Site smoothing and the domain-shift aim
1. Re-run `python -m argo_deepmsi.eval.domain_shift embed|image` and
   `scripts/domain_shift/site_smoothing.py wagner` on the full cohort — the label-free
   site smoothing variants apply directly to the sealed set without refitting.
2. Carry forward only variants that improved pooled development AUROC *and* did not reduce
   any site's AUROC (see `results/analysis/domain_shift/site_smoothing/`).
3. Measure shift with the three natural contrasts (restain, re-cut, site) at image and
   embedding level; these are the preliminary data for the domain-shift aim.

### P5 — Learning beyond the reference (only after P3)
Ordered: (a) distil Wagner/PALADIN into a Mascaret/Phaet student using all tiles incl.
unlabelled slides; (b) in-domain self-supervised adaptation on Nigerian tiles, audited for
site predictability; (c) low-capacity MIL heads only, nested, paired bootstrap vs reference.
No from-scratch transformers at this label budget.

## Rules carried over
Patient-grouped splits; nested selection; no sealed-set labels in any decision; one
hypothesis per `[run].name`; negative results go into the ledger.
