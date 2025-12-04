from pathlib import Path

labels_root = Path("outputs/yolo_masks/labels")  # adjust if needed

bad_entries = []

for txt in labels_root.rglob("*.txt"):
    lines = txt.read_text().strip().splitlines()
    if not lines:
        continue

    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cls = parts[0]
        coords_str = parts[1:]

        # 1) basic structure checks
        try:
            c = int(cls)
        except ValueError:
            bad_entries.append((txt, lineno, "non-int class"))
            continue
        if c < 0 or c > 10:  # you probably only use 0/1 anyway
            bad_entries.append((txt, lineno, f"class out of range: {c}"))
            continue

        if len(coords_str) < 6 or len(coords_str) % 2 != 0:
            bad_entries.append((txt, lineno, f"bad coord count: {len(coords_str)}"))
            continue

        # 2) coords as floats in [0,1]
        try:
            coords = [float(v) for v in coords_str]
        except ValueError:
            bad_entries.append((txt, lineno, "non-float coord"))
            continue

        bad_coord = False
        for v in coords:
            if not (0.0 <= v <= 1.0):
                bad_entries.append((txt, lineno, f"coord out of range: {v}"))
                bad_coord = True
                break
        if bad_coord:
            continue

        # 3) bbox from polygon must have positive area
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        if x_max <= x_min or y_max <= y_min:
            bad_entries.append(
                (txt, lineno, f"degenerate bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            )
            continue

if bad_entries:
    print("Found potential bad labels:")
    for p, ln, msg in bad_entries[:200]:
        print(p, "line", ln, "->", msg)
    print(f"... total bad entries: {len(bad_entries)}")
else:
    print("All labels look structurally OK.")