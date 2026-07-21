# Data

The real benchmark images, the synthetic corpus (Stage A and Stage B), and the
text lexicons used by the Stage A generator are distributed as a separate
anonymized data supplement (the file sizes and provenance make them unsuitable
for a code repository). Place the release under this `data/` directory, or
symlink the manifests and image trees to the relative paths referenced by
`configs/experiments/*.yaml`.

## Expected layout (relative to repository root)

```
notebooks/manifest_hebrew_unambiguous.jsonl              # real benchmark manifest
notebooks/manifest_hebrew_unambiguous_group_split_rowid.jsonl  # entry-disjoint split
notebooks/paleo_ocr_part2/yolo_22_from_val_split_singlecls/manifest_train.jsonl
notebooks/paleo_ocr_part2/yolo_22_from_val_split_singlecls/manifest_val.jsonl
notebooks/syn_v2_styled_advanced_20260222/manifest.jsonl # Stage B manifest + images
runs/synthetic_v2_parallel_advanced_20260222/all_manifest.jsonl  # Stage A manifest + images
notebooks/paleo_ocr_texts_pack/                          # lexicons for Stage A generation
```

Each manifest is a JSONL file with one record per image, containing the image
path, character sequence, character-level bounding boxes, synchronized text
variants, split label, and (for real records) the catalog `row_id` grouping.

## Real benchmark

- 307 real seal photographs from 129 catalog entries.
- 2,963 manually localized signs.
- 157 train / 150 evaluation images (27 / 102 entries), entry-disjoint.
- Character inventory: 22 Paleo-Hebrew letters (no distinct final forms).

## Synthetic corpus

- 200,000 images total.
- Stage A: structural, lexicon-aware, font-based renders with exact supervision.
- Stage B: structure-preserving diffusion stylization of Stage A renders,
  retaining the Stage A text and boxes.
- Per-sample generation metadata (font, stroke width, blur, noise, taper,
  depth-like perturbation, contrast, color enhancement) is stored alongside
  each record.

## Leakage control

Stage B style-source manifests were audited against the evaluation image
identifiers; no evaluation image is used as a style source. Group identifiers
(`row_id`) expose multiple views of one artifact so entry-level overlap can be
checked. The audit script and its output are included in the data supplement.

## License

Real images and synthetic outputs are released under CC-BY 4.0; generation
code under MIT; fonts under their respective open-source licenses (SIL OFL,
GPL with Font Exception). See the data release documentation for asset-level
terms.
