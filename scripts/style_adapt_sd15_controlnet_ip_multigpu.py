#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style_adapt_sd15_controlnet_ip_multigpu.py

Multi-GPU (multi-process) neural style adaptation:
SD1.5 + ControlNet Canny + IP-Adapter, batched, with shard-by-uid parallelism.

Why multi-process sharding:
- easiest stable way to use multiple GPUs for SD inference: one pipeline per GPU
  (diffusers community recommendation). :contentReference[oaicite:2]{index=2}
- diffusers also has distributed inference docs, but sharding is simpler here. :contentReference[oaicite:3]{index=3}

ControlNet weight is controlled by `controlnet_conditioning_scale`. :contentReference[oaicite:4]{index=4}
IP-Adapter influence controlled by `set_ip_adapter_scale`. :contentReference[oaicite:5]{index=5}
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# -------------------------
# Utils
# -------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write_jsonl(path: Path, lines: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")
    os.replace(tmp, path)

def safe_decode(b: Optional[bytes]) -> str:
    if not b:
        return ""
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return str(b)

def md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def in_shard(uid: str, shard_id: int, n_shards: int) -> bool:
    if n_shards <= 1:
        return True
    return (md5_int(uid) % n_shards) == shard_id

def detect_any_images(root: Path) -> bool:
    for ext in IMG_EXTS:
        if any(root.rglob(f"*{ext}")):
            return True
    return False

# -------------------------
# Prompt routing
# -------------------------

@dataclass
class PromptPack:
    prompt_seal: str
    prompt_plain: str
    prompt_lineart: str
    neg: str

def pick_prompt(
    meta: Dict[str, Any],
    edge_density: float,
    mode: str,
    pack: PromptPack,
) -> str:
    """
    mode:
      - fixed: always seal prompt
      - by_doc_kind: meta.doc_kind (seal/plain)
      - auto: if edges dense => lineart, else doc_kind (seal/plain) fallback
    """
    if mode == "fixed":
        return pack.prompt_seal

    doc_kind = (meta.get("doc_kind") or "").strip().lower()
    if mode == "by_doc_kind":
        return pack.prompt_plain if doc_kind == "plain" else pack.prompt_seal

    # auto
    if edge_density >= 0.18:  # heuristic threshold on canny edge map density
        return pack.prompt_lineart
    return pack.prompt_plain if doc_kind == "plain" else pack.prompt_seal

# -------------------------
# Extraction + style bank
# -------------------------

def ensure_extracted(real_zip: Path, extract_dir: Path) -> None:
    ensure_dir(extract_dir)
    if detect_any_images(extract_dir):
        return
    if not real_zip.exists():
        raise FileNotFoundError(f"REAL_DATA_ZIP not found: {real_zip}")
    print(f"📦 Extracting {real_zip} -> {extract_dir} ...", flush=True)
    with zipfile.ZipFile(real_zip, "r") as zf:
        zf.extractall(extract_dir)
    print("✅ Extracted.", flush=True)

def collect_style_paths(
    mode: str,
    style_bank_n: int,
    manifest_test: Optional[Path],
    extract_dir: Path,
    work: Path,
) -> List[str]:
    def _resolve_real(p: str) -> Optional[str]:
        if not p:
            return None
        pp = Path(str(p))
        if pp.is_absolute() and pp.exists():
            return str(pp)

        roots = [
            extract_dir,
            extract_dir / "seals_images_jpeg",
            work / "data" / "seals_images_jpeg",
            work / "data" / "seals_images_jpeg" / "seals_images_jpeg",
            work,
            work / "seals_images_jpeg_downloaded",
            Path("/content"),
            Path("/content/work"),
        ]
        if not pp.is_absolute():
            for root in roots:
                cand = (root / pp)
                if cand.exists():
                    return str(cand.resolve())

        parts = pp.parts
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "seals_images_jpeg":
            pp2 = Path(*parts[2:])
            for root in roots:
                cand = (root / pp2)
                if cand.exists():
                    return str(cand.resolve())
        return None

    paths: List[str] = []
    if mode == "hebrew_test" and manifest_test and manifest_test.exists():
        recs = list(iter_jsonl(manifest_test))
        for r in recs:
            img = r.get("image") or {}
            p = img.get("abs_path") or img.get("path") or r.get("image_path")
            rp = _resolve_real(p) if p else None
            if rp and Path(rp).suffix.lower() in IMG_EXTS:
                paths.append(rp)
        print("style bank (hebrew_test) candidates:", len(paths), flush=True)
    else:
        paths = [str(p) for p in extract_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if not paths:
            paths = [str(p) for p in (extract_dir / "seals_images_jpeg").rglob("*") if p.suffix.lower() in IMG_EXTS]
        print("style bank (all_seals) candidates:", len(paths), flush=True)

    paths = sorted(set(paths))
    if len(paths) > int(style_bank_n):
        random.Random(42).shuffle(paths)
        paths = paths[: int(style_bank_n)]
    return paths

# -------------------------
# Main worker (diffusers)
# -------------------------

def worker_main(args: argparse.Namespace) -> None:
    # Import heavy deps inside worker
    import torch
    import numpy as np
    import cv2
    from PIL import Image

    import diffusers
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, EulerAncestralDiscreteScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[worker {args.shard_id}/{args.n_shards}] device={device} dtype={dtype} pid={os.getpid()}", flush=True)

    work = Path(args.work).resolve()
    in_syn = Path(args.in_syn).resolve()
    man_in = in_syn / "all_manifest.jsonl"
    if not man_in.exists():
        raise FileNotFoundError(f"Missing synthetic manifest: {man_in}")

    out_root = Path(args.out).resolve()
    ensure_dir(out_root)
    ensure_dir(out_root / "images")

    # per-worker manifest & debug dirs
    man_shard = out_root / f"manifest_shard_{args.shard_id:03d}.jsonl"
    debug_dir = out_root / "debug" / f"shard_{args.shard_id:03d}"
    ensure_dir(debug_dir)

    # load style bank paths + images (each worker has its own memory)
    extract_dir = Path(args.extract_dir).resolve()
    style_paths = collect_style_paths(
        mode=args.style_bank_mode,
        style_bank_n=args.style_bank_n,
        manifest_test=Path(args.manifest_test).resolve() if args.manifest_test else None,
        extract_dir=extract_dir,
        work=work,
    )
    if not style_paths:
        raise RuntimeError("Style bank is empty. Check extract_dir / manifest_test.")

    def load_style_image(path: str, size=224):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2))
        im = im.resize((size, size), Image.BICUBIC)
        return im

    style_bank = [load_style_image(p, size=int(args.style_size)) for p in style_paths]
    print(f"[worker {args.shard_id}] style bank size={len(style_bank)}", flush=True)

    # Build pipeline
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if device == "cuda" and args.enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print(f"[worker {args.shard_id}] ✅ xformers enabled", flush=True)
        except Exception as e:
            print(f"[worker {args.shard_id}] ⚠️ xformers enable failed: {e!r}", flush=True)

    # memory helpers
    if device == "cuda":
        if args.enable_attention_slicing:
            try:
                pipe.enable_attention_slicing("max")
            except Exception:
                pass
        pipe.enable_vae_slicing()
        pipe.to(device)
    else:
        pipe.to(device)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    # IP-Adapter
    ip_ok = False
    if args.use_ip_adapter and hasattr(pipe, "load_ip_adapter"):
        for subfolder, weight in [
            ("models", "ip-adapter_sd15.safetensors"),
            ("models", "ip-adapter_sd15.bin"),
            ("models", "ip-adapter-plus_sd15.safetensors"),
            ("models", "ip-adapter-plus_sd15.bin"),
        ]:
            try:
                pipe.load_ip_adapter(args.ip_repo, subfolder=subfolder, weight_name=weight)
                ip_ok = True
                print(f"[worker {args.shard_id}] ✅ IP-Adapter loaded: {weight}", flush=True)
                break
            except Exception:
                continue

    def set_ip_scale(scale: float):
        try:
            pipe.set_ip_adapter_scale(scale)
        except Exception:
            pipe.set_ip_adapter_scale([scale])

    if ip_ok:
        set_ip_scale(float(args.ip_scale))

    # already done (global images directory + merged manifest if exists)
    already_done = set()
    merged_manifest = out_root / "manifest.jsonl"
    if merged_manifest.exists():
        for r in iter_jsonl(merged_manifest):
            u = r.get("uid")
            if u:
                already_done.add(u)
    for p in (out_root / "images").glob("*.png"):
        already_done.add(p.stem)
    for p in (out_root / "images").glob("*.jpg"):
        already_done.add(p.stem)

    # helpers
    def canny_control(init_im: Image.Image, low: int, high: int):
        arr = np.array(init_im.convert("RGB"))
        edges = cv2.Canny(arr, int(low), int(high))
        edges = np.stack([edges]*3, axis=-1)
        return Image.fromarray(edges), float(edges.mean() / 255.0)  # density

    def out_path_for_uid(uid: str) -> Path:
        ext = ".png" if args.save_format.lower() == "png" else ".jpg"
        return out_root / "images" / f"{uid}{ext}"

    def save_result(uid: str, res_im: Image.Image) -> Path:
        out_path = out_path_for_uid(uid)
        if args.save_format.lower() == "jpg":
            res_im.save(out_path, quality=int(args.jpeg_quality), subsampling=0, optimize=True)
        else:
            res_im.save(out_path)
        return out_path

    def save_grid(path: Path, imgs: List[Image.Image]):
        W, H = imgs[0].size
        pad = 8
        cols = len(imgs)
        canvas = Image.new("RGB", (cols*W + (cols+1)*pad, H + 2*pad), (245,245,245))
        x = pad
        for im in imgs:
            canvas.paste(im, (x, pad))
            x += W + pad
        canvas.save(path)

    def img_diff(a: Image.Image, b: Image.Image) -> float:
        aa = np.asarray(a).astype(np.float32)
        bb = np.asarray(b).astype(np.float32)
        return float(np.mean(np.abs(aa - bb)))

    pack = PromptPack(
        prompt_seal=args.prompt_seal,
        prompt_plain=args.prompt_plain,
        prompt_lineart=args.prompt_lineart,
        neg=args.neg,
    )

    # batching runners
    def run_pipe_batch_ip(syn_list, ctrl_list, style_list, seeds_list, prompt_list):
        bsz = len(syn_list)
        negs = [pack.neg] * bsz
        gen_device = "cuda" if device == "cuda" else "cpu"
        gens = [torch.Generator(device=gen_device).manual_seed(int(s)) for s in seeds_list]

        out = pipe(
            prompt=prompt_list,
            negative_prompt=negs,
            image=syn_list,
            control_image=ctrl_list,
            controlnet_conditioning_scale=float(args.controlnet_scale),
            strength=float(args.strength_ip),
            guidance_scale=float(args.guidance),
            num_inference_steps=int(args.steps),
            ip_adapter_image=[style_list],  # correct batching for 1 adapter
            generator=gens,
        )
        return out.images

    def run_pipe_batch_noip(syn_list, ctrl_list, style_list, seeds_list, prompt_list):
        bsz = len(syn_list)
        negs = [pack.neg] * bsz
        gen_device = "cuda" if device == "cuda" else "cpu"
        gens = [torch.Generator(device=gen_device).manual_seed(int(s)) for s in seeds_list]

        style_init = [im.resize((512,512), Image.BICUBIC) for im in style_list]
        out = pipe(
            prompt=prompt_list,
            negative_prompt=negs,
            image=style_init,
            control_image=ctrl_list,
            controlnet_conditioning_scale=float(args.controlnet_scale),
            strength=float(args.strength_noip),
            guidance_scale=float(args.guidance),
            num_inference_steps=int(args.steps),
            generator=gens,
        )
        return out.images

    def run_batch(items: List[Dict[str, Any]]):
        syn_list   = [it["syn"] for it in items]
        ctrl_list  = [it["ctrl"] for it in items]
        style_list = [it["style"] for it in items]
        seeds_list = [it["seed"] for it in items]
        prompt_list = [it["prompt"] for it in items]

        with torch.inference_mode():
            if ip_ok:
                return run_pipe_batch_ip(syn_list, ctrl_list, style_list, seeds_list, prompt_list)
            else:
                return run_pipe_batch_noip(syn_list, ctrl_list, style_list, seeds_list, prompt_list)

    def run_batch_with_fallback(items: List[Dict[str, Any]]):
        try:
            return run_batch(items)

        except torch.cuda.OutOfMemoryError as e:
            print(f"[worker {args.shard_id}] ❌ OOM batch={len(items)}: {e}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if args.batch_retry_halve_on_oom and len(items) > 1:
                mid = len(items) // 2
                left  = run_batch_with_fallback(items[:mid])
                right = run_batch_with_fallback(items[mid:])
                return left + right
            else:
                outs = []
                for it in items:
                    outs.extend(run_batch_with_fallback([it]))
                return outs

        except Exception as e:
            print(f"[worker {args.shard_id}] ❌ batch failed: {e!r}", flush=True)
            traceback.print_exc()
            outs = []
            for it in items:
                try:
                    outs.extend(run_batch([it]))
                except Exception as ee:
                    print(f"[worker {args.shard_id}]   ❌ single failed {it.get('uid')}: {ee!r}", flush=True)
                    traceback.print_exc()
            return outs

    # Stream tasks
    run_limit = None if (args.n_style in (None, 0)) else int(args.n_style)
    processed = 0
    debug_left = int(args.debug_ab_first_k)

    # write shard manifest in append mode
    fp_mode = "a" if man_shard.exists() else "w"
    with open(man_shard, fp_mode, encoding="utf-8") as f_out:
        batch: List[Dict[str, Any]] = []

        for idx, r in enumerate(iter_jsonl(man_in), 1):
            uid = r.get("uid")
            if not uid:
                continue
            if uid in already_done:
                continue
            if not in_shard(uid, int(args.shard_id), int(args.n_shards)):
                continue
            if run_limit is not None and processed >= run_limit:
                break

            # input path resolution
            img = r.get("image") or {}
            in_path = img.get("abs_path") or img.get("path")
            if not in_path:
                rp = img.get("rel_path")
                if rp:
                    in_path = str(in_syn / rp)
            if (not in_path) or (not Path(in_path).exists()):
                continue

            # deterministic style + seed
            hbytes = hashlib.md5(uid.encode("utf-8")).digest()
            style_idx = hbytes[0] % len(style_bank)
            seed_i = int.from_bytes(hbytes[:4], "little", signed=False)
            seed_val = int(args.seed + seed_i)

            syn_im = Image.open(in_path).convert("RGB").resize((512, 512), Image.BICUBIC)
            ctrl_im, ed = canny_control(syn_im, args.canny_low, args.canny_high)

            meta = r.get("meta") or {}
            prompt = pick_prompt(meta, ed, args.prompt_mode, pack)

            outp = out_path_for_uid(uid)
            if outp.exists() and outp.stat().st_size > 0:
                already_done.add(uid)
                continue

            batch.append({
                "uid": uid,
                "row": r,
                "in_path": in_path,
                "seed": seed_val,
                "syn": syn_im,
                "ctrl": ctrl_im,
                "style": style_bank[int(style_idx)],
                "style_idx": int(style_idx),
                "edge_density": float(ed),
                "prompt": prompt,
            })

            if len(batch) < int(args.batch_size):
                continue

            imgs = run_batch_with_fallback(batch)

            # optional AB debug on first K items (per worker)
            if ip_ok and debug_left > 0:
                take = min(debug_left, len(batch))
                sub = batch[:take]
                try:
                    set_ip_scale(0.0)
                    imgs0 = run_pipe_batch_ip(
                        [x["syn"] for x in sub],
                        [x["ctrl"] for x in sub],
                        [x["style"] for x in sub],
                        [x["seed"] for x in sub],
                        [x["prompt"] for x in sub],
                    )
                finally:
                    set_ip_scale(float(args.ip_scale))

                for j in range(take):
                    uidj = sub[j]["uid"]
                    d = img_diff(imgs[j], imgs0[j])
                    grid_path = debug_dir / f"{uidj}_grid.png"
                    save_grid(grid_path, [
                        sub[j]["syn"],
                        sub[j]["style"].resize((512,512), Image.BICUBIC),
                        imgs[j],
                        imgs0[j],
                    ])
                    print(f"[worker {args.shard_id}] 🧪 AB diff({uidj})={d:.3f} edges={sub[j]['edge_density']:.3f}", flush=True)
                debug_left -= take

            # Save + shard manifest
            for it, imj in zip(batch, imgs):
                uidj = it["uid"]
                out_img_path = save_result(uidj, imj)

                rr = dict(it["row"])
                rr["image"] = dict(rr.get("image") or {})
                rr["image"]["path"] = str(out_img_path)
                rr["image"]["abs_path"] = str(out_img_path)
                rr.setdefault("meta", {})
                rr["meta"]["styled"] = True
                rr["meta"]["save_format"] = args.save_format.lower()
                rr["meta"]["shard_id"] = int(args.shard_id)
                rr["meta"]["n_shards"] = int(args.n_shards)
                rr["meta"]["ip_adapter"] = bool(ip_ok)
                rr["meta"]["ip_scale"] = float(args.ip_scale) if ip_ok else None
                rr["meta"]["controlnet_scale"] = float(args.controlnet_scale)
                rr["meta"]["prompt_mode"] = args.prompt_mode
                rr["meta"]["prompt_used"] = it["prompt"]
                rr["meta"]["edge_density"] = float(it["edge_density"])

                f_out.write(json.dumps(rr, ensure_ascii=False) + "\n")
                already_done.add(uidj)
                processed += 1

            f_out.flush()
            batch = []

        # tail
        if batch:
            imgs = run_batch_with_fallback(batch)
            for it, imj in zip(batch, imgs):
                uidj = it["uid"]
                out_img_path = save_result(uidj, imj)

                rr = dict(it["row"])
                rr["image"] = dict(rr.get("image") or {})
                rr["image"]["path"] = str(out_img_path)
                rr["image"]["abs_path"] = str(out_img_path)
                rr.setdefault("meta", {})
                rr["meta"]["styled"] = True
                rr["meta"]["save_format"] = args.save_format.lower()
                rr["meta"]["shard_id"] = int(args.shard_id)
                rr["meta"]["n_shards"] = int(args.n_shards)
                rr["meta"]["ip_adapter"] = bool(ip_ok)
                rr["meta"]["ip_scale"] = float(args.ip_scale) if ip_ok else None
                rr["meta"]["controlnet_scale"] = float(args.controlnet_scale)
                rr["meta"]["prompt_mode"] = args.prompt_mode
                rr["meta"]["prompt_used"] = it["prompt"]
                rr["meta"]["edge_density"] = float(it["edge_density"])

                f_out.write(json.dumps(rr, ensure_ascii=False) + "\n")
                processed += 1

            f_out.flush()

    print(f"[worker {args.shard_id}] ✅ done: processed={processed}, shard_manifest={man_shard}", flush=True)

# -------------------------
# Launcher + merge
# -------------------------

def merge_manifests(out_root: Path) -> int:
    shard_mans = sorted(out_root.glob("manifest_shard_*.jsonl"))
    if not shard_mans:
        return 0

    # merge unique by uid (last write wins)
    by_uid: Dict[str, str] = {}
    for mp in shard_mans:
        for line in mp.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("uid")
            if uid:
                by_uid[uid] = json.dumps(r, ensure_ascii=False)

    merged = out_root / "manifest.jsonl"
    lines = [by_uid[k] for k in sorted(by_uid.keys())]
    atomic_write_jsonl(merged, lines)
    return len(lines)

def launch_multi_gpu(args: argparse.Namespace) -> None:
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        gpus = ["0"]

    # ensure real zip extracted once (avoid races)
    if args.real_zip:
        ensure_extracted(Path(args.real_zip).resolve(), Path(args.extract_dir).resolve())

    procs = []
    n = len(gpus)
    for shard_id, gpu in enumerate(gpus):
        env = os.environ.copy()
        # one visible gpu per worker => it will use cuda:0
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--mode", "worker",
            "--shard-id", str(shard_id),
            "--n-shards", str(n),
        ] + args._pass_through

        procs.append((shard_id, gpu, subprocess.Popen(cmd, env=env)))

    rc = 0
    for shard_id, gpu, p in procs:
        r = p.wait()
        if r != 0:
            rc = r
            print(f"❌ worker shard={shard_id} gpu={gpu} exited rc={r}", flush=True)

    if rc != 0:
        raise SystemExit(rc)

    out_root = Path(args.out).resolve()
    nrec = merge_manifests(out_root)
    print(f"✅ merged {nrec} -> {out_root/'manifest.jsonl'}", flush=True)

# -------------------------
# Argparse
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", type=str, default="launcher", choices=["launcher", "worker"])

    # multi-gpu
    p.add_argument("--gpus", type=str, default="0", help='e.g. "0,1,2,3" (launcher only)')
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)

    # paths
    p.add_argument("--work", type=str, default=".", help="base work dir for resolving test manifests")
    p.add_argument("--in-syn", type=str, required=True, help="synthetic dataset root with all_manifest.jsonl")
    p.add_argument("--out", type=str, required=True, help="output root")
    p.add_argument("--real-zip", type=str, default="", help="zip with real style images (optional if already extracted)")
    p.add_argument("--extract-dir", type=str, required=True, help="dir with extracted real images (style bank source)")
    p.add_argument("--manifest-test", type=str, default="", help="optional hebrew_test manifest jsonl")

    # style bank
    p.add_argument("--style-bank-mode", type=str, default="all_seals", choices=["all_seals", "hebrew_test"])
    p.add_argument("--style-bank-n", type=int, default=751)
    p.add_argument("--style-size", type=int, default=224)

    # models
    p.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--controlnet", type=str, default="lllyasviel/sd-controlnet-canny")

    # ip-adapter
    p.add_argument("--use-ip-adapter", type=int, default=1)
    p.add_argument("--ip-repo", type=str, default="h94/IP-Adapter")
    p.add_argument("--ip-scale", type=float, default=1.0)

    # params
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength-ip", type=float, default=0.78)
    p.add_argument("--strength-noip", type=float, default=0.75)
    p.add_argument("--controlnet-scale", type=float, default=0.75)

    # canny
    p.add_argument("--canny-low", type=int, default=80)
    p.add_argument("--canny-high", type=int, default=160)

    # batching
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batch-retry-halve-on-oom", type=int, default=1)

    # prompts
    p.add_argument("--prompt-mode", type=str, default="auto", choices=["fixed", "by_doc_kind", "auto"])
    p.add_argument("--prompt-seal", type=str, default="archaeological clay seal impression photo, ancient artifact, clay texture, cracks, dust, grain, realistic surface, museum lighting, macro")
    p.add_argument("--prompt-plain", type=str, default="ancient Hebrew manuscript fragment, parchment or papyrus texture, inked inscription, archival photo, museum lighting, macro, realistic")
    p.add_argument("--prompt-lineart", type=str, default="archaeological inscription rubbing on paper, graphite rubbing, high-contrast, museum documentation photo, realistic, macro")
    p.add_argument("--neg", type=str, default="modern printed text, watermark, logo, latin letters, extra symbols, deformed glyphs, low quality, oversharp")

    # debug / limits
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-style", type=int, default=0, help="per-run limit; 0=all remaining")
    p.add_argument("--debug-ab-first-k", type=int, default=0, help="save first K AB grids per worker (0 disables)")

    # memory toggles
    p.add_argument("--enable-xformers", type=int, default=0)
    p.add_argument("--enable-attention-slicing", type=int, default=0)

    # output format
    p.add_argument("--save-format", type=str, default="png", choices=["png", "jpg"])
    p.add_argument("--jpeg-quality", type=int, default=92)

    args, unknown = p.parse_known_args()
    args.use_ip_adapter = bool(int(args.use_ip_adapter))
    args.enable_xformers = bool(int(args.enable_xformers))
    args.enable_attention_slicing = bool(int(args.enable_attention_slicing))
    args.batch_retry_halve_on_oom = bool(int(args.batch_retry_halve_on_oom))
    args._pass_through = [
        "--work", args.work,
        "--in-syn", args.in_syn,
        "--out", args.out,
        "--extract-dir", args.extract_dir,
        "--manifest-test", args.manifest_test,
        "--style-bank-mode", args.style_bank_mode,
        "--style-bank-n", str(args.style_bank_n),
        "--style-size", str(args.style_size),
        "--base-model", args.base_model,
        "--controlnet", args.controlnet,
        "--use-ip-adapter", "1" if args.use_ip_adapter else "0",
        "--ip-repo", args.ip_repo,
        "--ip-scale", str(args.ip_scale),
        "--guidance", str(args.guidance),
        "--steps", str(args.steps),
        "--strength-ip", str(args.strength_ip),
        "--strength-noip", str(args.strength_noip),
        "--controlnet-scale", str(args.controlnet_scale),
        "--canny-low", str(args.canny_low),
        "--canny-high", str(args.canny_high),
        "--batch-size", str(args.batch_size),
        "--batch-retry-halve-on-oom", "1" if args.batch_retry_halve_on_oom else "0",
        "--prompt-mode", args.prompt_mode,
        "--prompt-seal", args.prompt_seal,
        "--prompt-plain", args.prompt_plain,
        "--prompt-lineart", args.prompt_lineart,
        "--neg", args.neg,
        "--seed", str(args.seed),
        "--n-style", str(args.n_style),
        "--debug-ab-first-k", str(args.debug_ab_first_k),
        "--enable-xformers", "1" if args.enable_xformers else "0",
        "--enable-attention-slicing", "1" if args.enable_attention_slicing else "0",
        "--save-format", args.save_format,
        "--jpeg-quality", str(args.jpeg_quality),
    ]
    if args.real_zip:
        args._pass_through += ["--real-zip", args.real_zip]
    return args

def main():
    args = parse_args()
    if args.mode == "launcher":
        launch_multi_gpu(args)
    else:
        # worker: if zip provided, still ok to call ensure_extracted (idempotent if already extracted)
        if args.real_zip:
            ensure_extracted(Path(args.real_zip).resolve(), Path(args.extract_dir).resolve())
        worker_main(args)

if __name__ == "__main__":
    main()