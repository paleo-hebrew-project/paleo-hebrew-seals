# Data release layout

Place the anonymized data release **directly under this `data/` directory**.
YAML configs reference these paths only (no `notebooks/` or absolute paths).

```
data/
├── real/
│   ├── manifest.jsonl
│   ├── manifest_train.jsonl
│   ├── manifest_val.jsonl
│   ├── manifest_group_split.jsonl
│   └── images/                  # or paths inside manifests may be relative here
├── stage_a/
│   ├── manifest.jsonl
│   └── images/
├── stage_b/
│   ├── manifest.jsonl
│   └── images/
├── lexicons/                    # Stage A text pack
└── SHA256SUMS
```

## Verify

```bash
python scripts/verify_data_release.py --root data --sha256
```

Checks: required files, record counts (307 / 157 / 150), entry-disjoint
`row_id` between train and evaluation, optional group-split overlap, SHA-256.

## Mapping from legacy internal layout

If you already have the project-internal trees, symlink:

```bash
mkdir -p data/real data/stage_a data/stage_b
ln -sfn /path/to/manifest_hebrew_unambiguous.jsonl data/real/manifest.jsonl
ln -sfn /path/to/yolo_22_from_val_split_singlecls/manifest_train.jsonl data/real/manifest_train.jsonl
ln -sfn /path/to/yolo_22_from_val_split_singlecls/manifest_val.jsonl data/real/manifest_val.jsonl
ln -sfn /path/to/manifest_hebrew_unambiguous_group_split_rowid.jsonl data/real/manifest_group_split.jsonl
ln -sfn /path/to/synthetic_v2_parallel_advanced_20260222/all_manifest.jsonl data/stage_a/manifest.jsonl
ln -sfn /path/to/syn_v2_styled_advanced_20260222/manifest.jsonl data/stage_b/manifest.jsonl
```

Image paths inside manifests must still resolve (either rewrite paths in the
JSONL records or keep image trees where the manifests point).

## Leakage

Stage B style sources must be disjoint from evaluation images at the catalog
entry level. The audit script/output belongs in the data release.

## License

Real images and synthetic outputs: typically CC-BY 4.0 (see release card).
Fonts: SIL OFL / GPL with Font Exception. Code: MIT.
