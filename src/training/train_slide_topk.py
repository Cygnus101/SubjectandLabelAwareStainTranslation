#!/usr/bin/env python3
"""Slide-only training entrypoint that samples patches from top10 pools per slide."""

from __future__ import annotations

from typing import Sequence

import torch

from . import train_augment as base


class SlideTopPoolDataset(base.AugmentedSlideDataset):
    """AugmentedSlideDataset variant enforcing the top10 sampling policy per slide."""

    @staticmethod
    def _allowed_pool(total: int, flags: Sequence[int]) -> list[int]:
        pool = [idx for idx, flag in enumerate(flags) if int(flag) > 0]
        if pool:
            return pool
        if total <= 0:
            raise RuntimeError("Slide contains no patches.")
        return list(range(total))

    def _sample_indices(
        self,
        total: int,
        flags: Sequence[int],
        deterministic: bool,
    ) -> list[int]:
        if total <= 0:
            raise RuntimeError("Slide contains no patches.")
        pool = self._allowed_pool(total, flags)
        if not pool:
            raise RuntimeError("Slide contains no eligible patches.")
        count = self.patches_per_slide
        if deterministic:
            expanded: list[int] = []
            while len(expanded) < count:
                expanded.extend(pool)
            return expanded[:count]
        if len(pool) >= count:
            order = torch.randperm(len(pool))[:count].tolist()
            return [pool[idx] for idx in order]
        picks = torch.randint(0, len(pool), (count,), dtype=torch.long).tolist()
        return [pool[idx] for idx in picks]


# Ensure the imported training pipeline uses the new dataset implementation.
base.AugmentedSlideDataset = SlideTopPoolDataset


def train(args):
    """Delegate to the base training routine with the top10 dataset injected."""
    return base.train(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = base.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
