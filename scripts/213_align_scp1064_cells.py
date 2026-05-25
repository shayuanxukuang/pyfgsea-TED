from __future__ import annotations

from scp1064_utils import build_cell_alignment


def main() -> None:
    alignment, qc = build_cell_alignment()
    print(f"SCP1064 cell alignment rows: {len(alignment)}")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
