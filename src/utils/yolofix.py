from pathlib import Path

labels_root = Path("outputs/yolo_masks/labels")

for txt in labels_root.rglob("*.txt"):
    lines = txt.read_text().strip().splitlines()
    if not lines:
        continue

    changed = False
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cls = parts[0]
        coords_str = parts[1:]

        # skip weird lines
        if len(coords_str) < 6 or len(coords_str) % 2 != 0:
            new_lines.append(line)
            continue

        coords = []
        for v in coords_str:
            x = float(v)
            orig = x
            if x < 0.0:
                x = 0.0
            if x > 1.0:
                x = 1.0
            if x != orig:
                changed = True
            coords.append(x)

        new_line = " ".join([cls] + [f"{v:.6f}" for v in coords])
        new_lines.append(new_line)

    if changed:
        print(f"Fixed {txt}")
        txt.write_text("\n".join(new_lines) + "\n")