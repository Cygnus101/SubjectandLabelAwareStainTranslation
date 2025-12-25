import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _build_split_dataframe(splits: dict, slides: list[dict]) -> pd.DataFrame:
    if not isinstance(splits, dict):
        raise ValueError("Split payload must be a JSON object.")
    if not isinstance(slides, list):
        raise ValueError("Slides payload must be a JSON list.")

    rows: list[dict[str, str]] = []

    def _stain_id(entry: dict) -> str:
        return str(
            entry.get("ret_stain_id")
            or entry.get("he_stain_id")
            or entry.get("stain_id")
            or ""
        ).strip()

    for split_name in ("train", "val", "test"):
        key = f"{split_name}_indices"
        for idx in splits.get(key, []):
            if not isinstance(idx, int) or not (0 <= idx < len(slides)):
                continue
            stain = _stain_id(slides[idx])
            if stain:
                rows.append({"stain_id": stain, "split": split_name})

    if not rows:
        raise RuntimeError("No slide indices found; ensure augmented_splits.json is populated.")

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect augmented_splits.json and report per-case split purity."
    )
    parser.add_argument(
        "--augmented-splits",
        type=Path,
        default=Path("augmented_splits.json"),
        help="Path to augmented_splits.json (train/val/test indices).",
    )
    parser.add_argument(
        "--augmented-slides",
        type=Path,
        default=Path("augmented_slides.json"),
        help="Path to augmented_slides.json used to map indices to stain_ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_payload = json.loads(args.augmented_splits.read_text("utf-8"))
    slides_payload = json.loads(args.augmented_slides.read_text("utf-8"))
    df = _build_split_dataframe(splits_payload, slides_payload)

    def middle(stain_id: str) -> str | None:
        m = re.match(r"[A-Z]_(\d+)_\d+", stain_id)
        return m.group(1) if m else None

    df["middle"] = df["stain_id"].apply(middle)

    middle_splits = df.groupby("middle")["split"].apply(lambda s: sorted(set(s)))

    pure_train = [m for m, sp in middle_splits.items() if sp == ["train"]]
    pure_val = [m for m, sp in middle_splits.items() if sp == ["val"]]
    pure_test = [m for m, sp in middle_splits.items() if sp == ["test"]]

    print("Pure TRAIN cases:", len(pure_train), pure_train)
    print("Pure VAL cases:", len(pure_val), pure_val)
    print("Pure TEST cases:", len(pure_test), pure_test)

    counts = df.groupby(["middle", "split"]).size().unstack(fill_value=0)
    majority_train: list[str | None] = []
    majority_val: list[str | None] = []
    majority_test: list[str | None] = []
    majority_mixed: list[str | None] = []
    for m, row in counts.iterrows():
        splits = row.to_dict()
        max_count = max(splits.values())
        max_splits = [k for k, v in splits.items() if v == max_count and max_count > 0]
        if len(max_splits) == 1:
            if max_splits[0] == "train":
                majority_train.append(m)
            elif max_splits[0] == "val":
                majority_val.append(m)
            elif max_splits[0] == "test":
                majority_test.append(m)
        else:
            majority_mixed.append(m)

    print("Majority TRAIN cases:", len(majority_train), majority_train)
    print("Majority VAL cases:", len(majority_val), majority_val)
    print("Majority TEST cases:", len(majority_test), majority_test)
    print("Mixed / no clear majority cases:", len(majority_mixed), majority_mixed)

    def _ret_id_from_index(idx: int) -> str:
        entry = slides_payload[idx]
        return str(entry.get("ret_stain_id") or entry.get("he_stain_id") or f"idx_{idx}")

    val_ids = sorted({_ret_id_from_index(idx) for idx in splits_payload.get("val_indices", [])})
    test_ids = sorted({_ret_id_from_index(idx) for idx in splits_payload.get("test_indices", [])})

    print("VAL stain_ids represented in augmented_splits:", len(val_ids), val_ids)
    print("TEST stain_ids represented in augmented_splits:", len(test_ids), test_ids)


if __name__ == "__main__":
    main()
