# Manifest (anonymous ZIP contents)

Suggested OpenReview upload name: `paleo-hebrew-seals-aaai27.zip`

Build from a tagged commit (example):

```bash
git tag aaai27-anonymous-v1
git archive --format=zip --prefix=paleo-hebrew-seals/ \
  -o paleo-hebrew-seals-aaai27.zip aaai27-anonymous-v1
```

Or, for a clean tree without `.git`:

```bash
rsync -a --exclude .git --exclude '__pycache__' --exclude '*.pt' \
  ./ /tmp/paleo-hebrew-seals/
(cd /tmp && zip -r paleo-hebrew-seals-aaai27.zip paleo-hebrew-seals)
```

## Must include

- `paleo_ocr/`, `configs/`, `scripts/`, `fonts/`
- `results/` (frozen tables)
- `README.md`, `REPRODUCIBILITY.md`, `REPRODUCE_TABLES.md`, `ORIGIN.md`,
  `SYSTEM_REQUIREMENTS.md`, `LICENSE`, `requirements.txt`, `environment.yml`
- `data/README.md` (+ full data release in a separate ZIP if size requires)

## Must exclude

- `.git/`
- `runs/`, `logs/`, weight blobs (`*.pt`, `*.pth`, `*.ckpt`)
- absolute paths, author names, emails, employer names, `/home/...`

## Pre-upload grep

```bash
rg -n -i 'gorbulev|mr3vial|aigorbulev|humonen|golyadkin|makarov|/home/jovyan|@gmail|@hse|theta|gigacode|axxx' .
```

Should return no hits in the ZIP contents.
