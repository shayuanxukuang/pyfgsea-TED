from __future__ import annotations

from scp1064_utils import prepare_protein_outcomes


def main() -> None:
    outputs = prepare_protein_outcomes()
    print("SCP1064 protein outcome preparation complete")
    for name, df in outputs.items():
        print(f"{name}: {getattr(df, 'shape', '')}")


if __name__ == "__main__":
    main()
