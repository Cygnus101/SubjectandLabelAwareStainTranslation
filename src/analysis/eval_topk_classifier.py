#!/usr/bin/env python3
"""
Quick sanity script to probe a trained top-k classifier on a handful of slides.

Steps:
1. Pick N slides (default 12) spanning different labels.
2. For each slide, grab rows from the attention CSV where is_top10 == 1.
3. Deterministically sample 16 patches without replacement and load their embeddings.
4. Pool embeddings using (a) attention-weighted sum and (b) plain mean.
5. Run a frozen classifier checkpoint on both pooled vectors and record logits/probs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, resolve_path  # noqa: E402
from training.train_topk_classifier import SlideMLP, EmbeddingPathMapper  # noqa: E402

ensure_project_root_on_syspath()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-csv", type=str, required=True)
    parser.add_argument("--classifier-ckpt", type=str, required=True)
    parser.add_argument("--embedding-prefix", type=str, default=None)
    parser.add_argument("--embedding-root", type=str, default=None)
    parser.add_argument("--slide-sample", type=int, default=12)
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--allow-replacement", action="store_true",
                        help="If a slide has fewer than --bag-size candidates, sample with replacement instead of skipping.")
    parser.add_argument("--min-candidates", type=int, default=None,
                        help="Minimum candidates required to consider a slide eligible. Defaults to --bag-size.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_slides(df: pd.DataFrame, sample_per_class: int, total: int, seed: int) -> List[str]:
    """Sample slide_ids from the already-filtered dataframe.

    Note: `sample_per_class` is kept for API compatibility but not used here.
    """
    rng = random.Random(seed)
    slide_ids = df["slide_id"].unique().tolist()
    rng.shuffle(slide_ids)
    return slide_ids[:total]


def load_embeddings(paths: List[str], mapper: EmbeddingPathMapper) -> np.ndarray:
    vecs = []
    for path in paths:
        resolved = mapper.map(path)
        vecs.append(np.load(resolved).astype(np.float32).reshape(-1))
    return np.stack(vecs, axis=0)


def pool_embeddings(vecs: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    attn_pool = (weights[:, None] * vecs).sum(axis=0)
    mean_pool = vecs.mean(axis=0)
    return attn_pool, mean_pool


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mapper = EmbeddingPathMapper(
        Path(resolve_path(args.embedding_prefix)) if args.embedding_prefix else None,
        Path(resolve_path(args.embedding_root)) if args.embedding_root else None,
    )

    df = pd.read_csv(resolve_path(args.attention_csv))
    required = {"slide_id", "label", "attention_weight", "embedding_path", "is_top10"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df = df[df["is_top10"] == 1]

    min_cand = args.min_candidates if args.min_candidates is not None else args.bag_size
    counts = df.groupby("slide_id").size()
    eligible_ids = counts[counts >= min_cand].index.tolist()
    if len(eligible_ids) == 0:
        logging.error(
            "No slides have >= %d candidates after filtering (is_top10==1). "
            "Try lowering --bag-size/--min-candidates or pass --allow-replacement.",
            min_cand,
        )
        return
    df = df[df["slide_id"].isin(eligible_ids)]

    slide_ids = sample_slides(df, sample_per_class=2, total=args.slide_sample, seed=args.seed)

    # infer embedding dim from first entry
    first_path = mapper.map(df.iloc[0]["embedding_path"])
    feat_dim = np.load(first_path).reshape(-1).shape[0]

    model = SlideMLP(feat_dim, hidden_dim=256, num_classes=df["label"].nunique())
    state = torch.load(resolve_path(args.classifier_ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    results: List[Dict[str, Any]] = []
    for slide_id in slide_ids:
        sub = df[df["slide_id"] == slide_id]
        if len(sub) < args.bag_size and not args.allow_replacement:
            logging.warning(
                "Slide %s has only %d candidates (< bag_size=%d); skipping (enable --allow-replacement to include).",
                slide_id,
                len(sub),
                args.bag_size,
            )
            continue
        # Use a per-slide deterministic seed so different slides don't sample identical rows
        slide_seed = (hash(slide_id) ^ args.seed) & 0xFFFFFFFF
        sub = sub.sample(n=args.bag_size, replace=(len(sub) < args.bag_size), random_state=slide_seed)
        embeddings = load_embeddings(sub["embedding_path"].tolist(), mapper)
        weights = sub["attention_weight"].to_numpy(dtype=np.float32)
        attn_vec, mean_vec = pool_embeddings(embeddings, weights)
        label = int(sub["label"].iloc[0])

        for variant, vec in (("attention", attn_vec), ("mean", mean_vec)):
            tensor = torch.from_numpy(vec).to(device).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
            results.append(
                {
                    "slide_id": slide_id,
                    "label": label,
                    "variant": variant,
                    "prediction": pred,
                    "confidence": float(probs.max().item()),
                    "logits": logits.squeeze(0).cpu().tolist(),
                }
            )
            logging.info(
                "Slide %s | variant=%s | label=%d | pred=%d | conf=%.3f",
                slide_id,
                variant,
                label,
                pred,
                probs.max().item(),
            )

    if args.output_json:
        if len(results) == 0:
            logging.warning(
                "No results were produced (likely all sampled slides were skipped). "
                "Try lowering --bag-size/--min-candidates or enabling --allow-replacement."
            )
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        logging.info("Saved results to %s", args.output_json)


if __name__ == "__main__":
    main()
