#!/usr/bin/env python3
"""Fill PRUTOK (6th column) with 9.0000 for lines 3-367 (1-based).

Creates a backup next to the original CSV before editing.
"""
from pathlib import Path
import shutil


DATA = Path(__file__).resolve().parents[1] / "data" / "QD_109000_Data.csv"
BACKUP = DATA.with_suffix(DATA.suffix + ".bak")


def main(start_line: int = 3, end_line: int = 367) -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Data file not found: {DATA}")

    text = DATA.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Ensure indices are within the file
    start_idx = max(0, start_line - 1)
    end_idx = min(end_line, len(lines))

    for i in range(start_idx, end_idx):
        line = lines[i]
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            # Preserve a few leading spaces like the original formatting
            parts[5] = "   9.0000"
            lines[i] = ",".join(parts)

    DATA.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {DATA} lines {start_line}-{end_line}. Backup at {BACKUP}")


if __name__ == "__main__":
    main()
