"""Reproducibility: seeds, env snapshot, git hash."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort deterministic algorithms (may reduce speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def try_git_hash(cwd: Optional[Path] = None) -> str:
    """Return short git SHA or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path.cwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out[:12]
    except Exception:
        return "unknown"


def capture_environment() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("CUDA", "PYTHON", "OMP", "MKL"))},
    }


def write_run_metadata(
    out_dir: Path,
    *,
    config_dict: Dict[str, Any],
    argv: Optional[list] = None,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": seed,
        "git_hash": try_git_hash(),
        "git_hash_note": "If unknown, repo may not be a git checkout.",
        "environment": capture_environment(),
        "argv": argv or sys.argv,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config_resolved.json").write_text(json.dumps(config_dict, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (out_dir / "seed.txt").write_text(str(seed), encoding="utf-8")
