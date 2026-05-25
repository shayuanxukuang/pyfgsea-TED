from __future__ import annotations

from scp1064_utils import compare_author_effects


def main() -> None:
    outputs = compare_author_effects()
    print("SCP1064 author effect comparison complete")
    for name, df in outputs.items():
        print(f"{name}: {getattr(df, 'shape', '')}")


if __name__ == "__main__":
    main()
