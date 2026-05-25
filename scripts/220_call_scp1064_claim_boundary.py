from __future__ import annotations

from scp1064_utils import call_claim_boundary


def main() -> None:
    claim = call_claim_boundary()
    print(claim.to_string(index=False))


if __name__ == "__main__":
    main()
