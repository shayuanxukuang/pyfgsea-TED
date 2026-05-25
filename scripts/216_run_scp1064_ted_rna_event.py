from __future__ import annotations

from scp1064_utils import run_rna_event_scoring


def main() -> None:
    outputs = run_rna_event_scoring()
    print("SCP1064 RNA event scoring complete")
    for name, df in outputs.items():
        print(f"{name}: {getattr(df, 'shape', '')}")


if __name__ == "__main__":
    main()
