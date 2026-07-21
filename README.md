# PaleoHebrew-Seals: Code & Data Supplement (Anonymous)

Anonymous reproducibility package for the submission
**“Preserving Low-Resource Cultural Heritage with AI: A Dataset and Synthetic-to-Real Study of Paleo-Hebrew Seals.”**

This is a **code-and-data supplement** for the AAAI AISI track: YAML-driven
detector and classifier sweeps, Stage A generation, the real Stage B diffusion
stylization pipeline used to build the 200k corpus, OCR baselines, group-disjoint
splitting, and frozen paper tables. It is intended to facilitate follow-up work.

> **Evaluation split.** The 150-image real partition used in all tables is an
> **evaluation / comparative validation split**. It participated in checkpoint
> selection in parts of the project. It is **not** a sealed blind test set. The
> same wording is used in the paper, table captions, and `results/run_manifest.json`.

## What is and is not reproducible from this repository

| Component | Status |
|-----------|--------|
| Detector / classifier training & evaluation (A, A+R, B+R, R, R40) | Fully reproducible from YAML + scripts |
| Stage A structural generator | Included (`paleo_ocr/synthetic_v_2_generator.py`) |
| Stage B diffusion stylization (SD1.5 + ControlNet Canny + IP-Adapter) | Included (`scripts/style_adapt_sd15_controlnet_ip_multigpu.py`) — the script that produced the 200k corpus |
| Classic CV stylization helper | Included (`paleo_ocr/style_adapt.py`); its `diffusion` mode is a **non-functional scaffold** and is **not** the Stage B corpus pipeline |
| Frozen paper tables | Included under `results/` |
| Real / Stage A / Stage B images | Distributed as a separate data release under `data/` (see `data/README.md`) |

**Regime naming (aligned with code and paper):**

| Code | Training meaning |
|------|------------------|
| `A` | Stage A structural synth only |
| `A+R` | Stage A pretrain → real finetune |
| `B+R` | Phase 0 on **Stage B styled images only** → real finetune. Stage B images are produced from Stage A. Paper shorthand: **A+B+R** (generation pipeline + real), **not** joint training on A and B together |
| `R` / `R40` | Real only |

## Repository layout

```
.
├── paleo_ocr/                 # Python package (train, eval, Stage A, OCR)
├── configs/experiments/       # YAML regimes (paths under data/)
├── scripts/                   # sweeps, Stage B diffusion, verify, reproduce_tables
├── fonts/                     # open-source Paleo-Hebrew TTFs (Stage A)
├── data/                      # place the data release here
├── results/                   # frozen tables matching the paper
├── REPRODUCIBILITY.md
├── SYSTEM_REQUIREMENTS.md
├── REPRODUCE_TABLES.md
└── ORIGIN.md                  # notebook lineage (Paleo_OCR.ipynb → YAML sweeps)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# CUDA torch/torchvision: see SYSTEM_REQUIREMENTS.md

# Place the anonymized data release under data/ (see data/README.md), then:
python scripts/verify_data_release.py --root data --sha256

# Reproduce the published tables from frozen results:
python scripts/reproduce_tables.py --output results
```

### Detector Stage A + real (one backbone)

```bash
DETECTOR_GPUS="0" DETECTOR_MODELS="yolov8m.pt" \
  bash scripts/run_parallel_detector_stagea_sweep.sh
```

### Classifier Stage A + 120 real epochs

```bash
CLASSIFIER_GPUS="0 1" CLASSIFIER_PHASE1_EPOCHS=120 \
  bash scripts/run_parallel_classifier_stagea_backbones.sh
```

### Stage B regeneration (optional; heavy)

```bash
python scripts/style_adapt_sd15_controlnet_ip_multigpu.py --mode launcher \
  --gpus 0,1,2,3,4,5,6,7 \
  --in-syn data/stage_a \
  --out data/stage_b_regen \
  --extract-dir data/real/images \
  --batch-size 128 --prompt-mode auto \
  --controlnet-scale 0.70 --ip-scale 1.2 \
  --guidance 6.0 --steps 20 --seed 123
```

Default hyperparameters match the run that produced the 200k Stage B corpus
(SD1.5, ControlNet Canny, IP-Adapter, 512², guidance 6, 20 steps). See
`ORIGIN.md` and `SYSTEM_REQUIREMENTS.md`.

## Classifier grid (16 complete sweeps)

Included: ConvNeXt-T/S/B/L, EfficientNet-B0/B1/B2/B3-NS, ResNet-34/101,
Swin-T/S/B, SwinV2-T, ViT-S/B.

Excluded from cross-architecture analysis (incomplete regime): ResNet-50,
SwinV2-B, SwinV2-S.

## License

Code: MIT. Fonts under `fonts/`: SIL OFL / GPL with Font Exception. Real images
and synthetic outputs: terms in the data release (typically CC-BY 4.0).

## AI-use disclosure

Generative AI assisted drafting and language editing of the accompanying paper.
The authors verified and remain responsible for all claims, results, and code.
