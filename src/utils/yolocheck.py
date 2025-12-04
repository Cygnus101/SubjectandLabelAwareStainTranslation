from pathlib import Path

labels_root = Path("outputs/yolo_masks/labels")  # or your labels root

bad_files = []

for txt in labels_root.rglob("*.txt"):
    with txt.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # empty line = no object (YOLO can handle, but we can flag if whole file empty)

            parts = line.split()
            # At least: class + 3 (x,y) pairs => 1 + 3*2 = 7 values
            if len(parts) < 7:
                bad_files.append((txt, lineno, "too few values"))
                continue

            cls = parts[0]
            coords = parts[1:]

            # 1) class index integer?
            try:
                c = int(cls)
            except ValueError:
                bad_files.append((txt, lineno, f"non-int class: {cls}"))
                continue
            if c not in (0, 1):  # you only have he/reticulin
                bad_files.append((txt, lineno, f"out-of-range class: {c}"))

            # 2) even number of coord values
            if len(coords) % 2 != 0:
                bad_files.append((txt, lineno, f"odd number of coords: {len(coords)}"))
                continue

            # 3) coords in [0, 1] and finite
            try:
                vals = [float(v) for v in coords]
            except ValueError:
                bad_files.append((txt, lineno, "non-float coord"))
                continue

            for v in vals:
                if not (0.0 <= v <= 1.0):
                    bad_files.append((txt, lineno, f"coord out of range: {v}"))
                    break

if bad_files:
    print("Found potential bad labels:")
    for p, ln, msg in bad_files[:50]:
        print(p, "line", ln, "->", msg)
    print(f"... total bad entries: {len(bad_files)}")
else:
    print("All labels look syntactically OK.")