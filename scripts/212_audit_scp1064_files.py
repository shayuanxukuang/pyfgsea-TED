from __future__ import annotations

from scp1064_utils import audit_files


def main() -> None:
    outputs = audit_files()
    print("SCP1064 file audit complete")
    for name, df in outputs.items():
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
