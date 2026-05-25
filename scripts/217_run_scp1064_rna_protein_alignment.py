from __future__ import annotations

from scp1064_utils import run_rna_protein_alignment


def main() -> None:
    outputs = run_rna_protein_alignment()
    print("SCP1064 RNA/protein alignment complete")
    for name, df in outputs.items():
        print(f"{name}: {getattr(df, 'shape', '')}")


if __name__ == "__main__":
    main()
