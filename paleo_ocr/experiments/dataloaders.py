"""Sampling utilities for mixed-source classifier training and fairness counters."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler


@dataclass
class SourceBatchStats:
    """Counts samples seen per logical source tag (e.g. real / synth_a)."""

    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, tag: str, n: int = 1) -> None:
        self.counts[tag] = self.counts.get(tag, 0) + n

    def to_dict(self) -> Dict[str, int]:
        return dict(self.counts)


def _infer_concat_ranges(datasets: List[Dataset]) -> List[Tuple[int, int, int]]:
    """Return list of (start_idx, end_idx), dataset index for ConcatDataset."""
    out: List[Tuple[int, int, int]] = []
    offset = 0
    for i, ds in enumerate(datasets):
        n = len(ds)
        out.append((offset, offset + n, i))
        offset += n
    return out


def weighted_sampler_for_concat(
    datasets: List[Dataset],
    weights: List[float],
    num_samples: int,
    seed: int = 42,
) -> WeightedRandomSampler:
    """
    datasets: order matches ConcatDataset order.
    weights: one positive weight per subdataset (same length as datasets).
    """
    if len(datasets) != len(weights):
        raise ValueError("length mismatch")
    w = [float(x) for x in weights]
    if any(x <= 0 for x in w):
        raise ValueError("weights must be positive")

    sample_weights: List[float] = []
    for ds, wt in zip(datasets, w):
        sw = wt / float(len(ds))
        sample_weights.extend([sw] * len(ds))

    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True, generator=gen)


def curriculum_synth_fraction(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    t = epoch / max(1, total_epochs - 1)
    return float(start + (end - start) * t)


class AlternatingBatchIterator:
    """
    Yields batches of indices for ConcatDataset: one real batch, then K synth batches.
    real_range / synth_range are (start, end) index ranges in ConcatDataset flat index space.
    """

    def __init__(
        self,
        sampler_real: Sampler[int],
        sampler_synth: Sampler[int],
        batch_size: int,
        k_synth: int,
        seed: int = 42,
    ):
        self.sampler_real = sampler_real
        self.sampler_synth = sampler_synth
        self.batch_size = batch_size
        self.k_synth = max(1, int(k_synth))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch * 1000003)
        it_r = iter(self.sampler_real)
        it_s = iter(self.sampler_synth)
        while True:
            batch: List[int] = []
            try:
                for _ in range(self.batch_size):
                    batch.append(next(it_r))
            except StopIteration:
                return
            yield batch
            for _ in range(self.k_synth):
                b2: List[int] = []
                try:
                    for _ in range(self.batch_size):
                        b2.append(next(it_s))
                except StopIteration:
                    if b2:
                        yield b2
                    return
                yield b2


@dataclass
class StepStats:
    optimizer_steps: int = 0
    real_samples: int = 0
    synth_samples: int = 0
