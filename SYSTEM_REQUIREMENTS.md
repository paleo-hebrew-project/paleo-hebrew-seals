# System requirements

## Hardware (paper runs)

- 6–8 × NVIDIA H100 80 GB (or equivalent ≥40 GB for smaller backbones)
- ≥256 GB host RAM recommended for parallel classifier sweeps
- Local SSD for YOLO shared export cache (`DETECTOR_SHARED_CACHE_ROOT`)

## Software

| Component | Paper / recommended |
|-----------|---------------------|
| OS | Linux x86_64 |
| Python | 3.10–3.11 |
| CUDA toolkit | 12.4+ (match driver) |
| NVIDIA driver | supporting CUDA 12.4+ |
| PyTorch | 2.1+ with CUDA (paper: 2.10+cu128) |
| torchvision | matching torch build (**required**; used by `train_classify.py`) |
| ultralytics | ≥8.2 (paper: 8.4.x) |
| timm | ≥1.0 |
| System Tesseract | binary `tesseract` + `heb` language pack |
| Kraken | ≥3.1 + Hebrew manuscript checkpoints listed below |

Install PyTorch/torchvision from the official index for your CUDA version, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Do **not** rely on the PyPI package named `tesseract` for OCR baselines. The
code invokes the system binary via subprocess. On Debian/Ubuntu:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-heb
```

## Stage B regeneration (optional)

To re-run `scripts/style_adapt_sd15_controlnet_ip_multigpu.py`:

```bash
pip install "diffusers>=0.27" "transformers>=4.40" "accelerate>=0.30" "controlnet_aux>=0.0.7"
```

Model IDs downloaded on first run (Hugging Face Hub):

| Role | ID / file |
|------|-----------|
| Base | `runwayml/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/sd-controlnet-canny` |
| IP-Adapter | `h94/IP-Adapter` → `ip-adapter_sd15.safetensors` |

Corpus run defaults: resolution 512², guidance 6.0, steps 20, strength~0.78,
controlnet_scale 0.70, ip_scale 1.2, seed 123, style bank = real seal photos
(disjoint from the evaluation split).

## OCR baseline model identifiers

| Engine | Identifier |
|--------|------------|
| Tesseract | `heb` |
| Kraken | MiDRASH_Gen_01, BiblIA_01, Ashkenazi_01, Italian_01, Sephardi_01 |

Exact checkpoint URLs/checksums should be recorded in the data release under
`data/ocr_baselines/SHA256SUMS` when shipping the full supplement.

## Environment files

- `requirements.txt` — lower bounds for a fresh install
- `environment.yml` — conda skeleton (Python + pip deps)
- Prefer pinning exact builds in a private `requirements-lock.txt` generated
  from the machine that produced the paper tables (`pip freeze`).
