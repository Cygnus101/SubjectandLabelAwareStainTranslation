#!/usr/bin/env python3
"""
Show top-level contents of the folder where this script resides.
"""

from pathlib import Path

def show_top_level():
    # get the directory where this script file is located
    script_dir = Path(__file__).resolve().parent
    print(f"\n📂 {script_dir}")
    print("-" * (len(str(script_dir)) + 4))

    # list immediate contents (no recursion)
    for item in sorted(script_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        if item.is_dir():
            print(f"📁 {item.name}/")
        else:
            print(f"📄 {item.name}")

if __name__ == "__main__":
    show_top_level()