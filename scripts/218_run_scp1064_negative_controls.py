from __future__ import annotations

from scp1064_utils import run_negative_controls


def main() -> None:
    outputs = run_negative_controls()
    print("SCP1064 negative controls complete")
    for name, df in outputs.items():
        print(f"{name}: {getattr(df, 'shape', '')}")


if __name__ == "__main__":
    main()
