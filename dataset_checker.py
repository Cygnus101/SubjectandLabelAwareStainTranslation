import pandas as pd
import re

df = pd.read_csv("stain_id_splits.csv")

# extract middle number from stain_id
def middle(stain_id):
    m = re.match(r"[A-Z]_(\d+)_\d+", stain_id)
    return m.group(1) if m else None

df["middle"] = df["stain_id"].apply(middle)

# group by middle (case number), collect unique splits
middle_splits = df.groupby("middle")["split"].apply(lambda s: sorted(set(s)))

# identify pure groups
pure_train = [m for m, sp in middle_splits.items() if sp == ["train"]]
pure_val   = [m for m, sp in middle_splits.items() if sp == ["val"]]
pure_test  = [m for m, sp in middle_splits.items() if sp == ["test"]]

print("Pure TRAIN cases:", len(pure_train), pure_train)
print("Pure VAL cases:", len(pure_val), pure_val)
print("Pure TEST cases:", len(pure_test), pure_test)

# --- Majority split per middle ---
counts = df.groupby(["middle", "split"]).size().unstack(fill_value=0)
majority_train = []
majority_val = []
majority_test = []
majority_mixed = []
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

df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

val_both_stain_ids = sorted([sid for sid, grp in df_val.groupby("stain_id") if grp["stain"].nunique()>=2])
test_both_stain_ids = sorted([sid for sid, grp in df_test.groupby("stain_id") if grp["stain"].nunique()>=2])

print("VAL stain_ids with both H&E and Reticulin:", len(val_both_stain_ids), val_both_stain_ids)
print("TEST stain_ids with both H&E and Reticulin:", len(test_both_stain_ids), test_both_stain_ids)