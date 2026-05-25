from __future__ import annotations

from scp1064_utils import build_perturbation_metadata


def main() -> None:
    metadata = build_perturbation_metadata()
    print(f"SCP1064 perturbation metadata rows: {len(metadata)}")


if __name__ == "__main__":
    main()
