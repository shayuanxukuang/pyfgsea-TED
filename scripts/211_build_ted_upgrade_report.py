from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "ted_upgrade_benchmark_registry.tsv"
DEFAULT_TABLES = ROOT / "results" / "ted_known_source_validation" / "tables"
DEFAULT_GATA1 = ROOT / "results" / "gata1_cross_dataset_support" / "tables"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def count_pass(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return "not_available"
    values = df[column].astype(str).str.lower()
    return f"{int(values.isin(['true', 'pass']).sum())}/{len(df)}"


def audit_report(registry: pd.DataFrame) -> list[str]:
    lines = [
        "# TED Known-Source Dataset Freeze Report",
        "",
        "## Dataset Registry",
        "",
        "| Dataset | Role | Access | Expected event | Claim if pass | Claim if fail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in registry.to_dict("records"):
        lines.append(
            "| {dataset} | {analysis_role} | {access_status} | {expected_event} | {claim_if_pass} | {claim_if_fail} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Freeze Decision",
            "",
            "- Every dataset has an explicit role, access status, expected event, pass claim and fail claim.",
            "- SCP1064 is frozen as access-limited metadata only until a portal-authenticated download is available.",
            "- GATA1/T21 cross-dataset rows are frozen as mechanism support only; they cannot promote GSE271399 to Level 4.",
        ]
    )
    return lines


def final_report(known_source: Path, gata1: Path, registry: pd.DataFrame) -> list[str]:
    event = read_tsv(known_source / "ted_event_recovery_summary.tsv")
    outcome = read_tsv(known_source / "ted_outcome_alignment_summary.tsv")
    claims = read_tsv(known_source / "ted_dataset_level_claim_boundary.tsv")
    neg = read_tsv(known_source / "ted_negative_control_summary.tsv")
    gata1_summary = read_tsv(gata1 / "gata1_cross_dataset_support_summary.tsv")
    gata1_claim = read_tsv(gata1 / "gata1_claim_boundary_decision.tsv")

    lines = [
        "# TED Upgrade Final Report",
        "",
        "## Bottom Line",
        "",
        "- TED methodology can be strengthened with known-source/outcome and matched-intervention validation.",
        "- GATA1/T21 remains Level 3.5: cross-dataset mechanism support is added, but same-system matched rescue is absent.",
        "",
        "## Known-Source Validation",
        "",
        f"- Event recovery rows: {len(event)}",
        f"- Outcome alignment rows: {len(outcome)}",
        f"- Dataset claim rows: {len(claims)}",
        f"- Negative-control pass: {count_pass(neg, 'negative_control_pass')}",
        "",
    ]
    if not claims.empty:
        lines.extend(["| Dataset | Claim boundary | Status | Reason |", "| --- | --- | --- | --- |"])
        for row in claims.to_dict("records"):
            lines.append(
                f"| {row.get('dataset')} | {row.get('claim_boundary')} | {row.get('status')} | {row.get('reason')} |"
            )
    lines.extend(["", "## GATA1 Cross-Dataset Support", ""])
    if not gata1_summary.empty:
        lines.extend(["| Metric | Value |", "| --- | --- |"])
        for row in gata1_summary.to_dict("records"):
            lines.append(f"| {row.get('metric')} | {row.get('value')} |")
    if not gata1_claim.empty:
        lines.extend(["", "| Decision | Value |", "| --- | --- |"])
        for key, value in gata1_claim.iloc[0].to_dict().items():
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Registry Snapshot", ""])
    lines.extend(audit_report(registry)[4:])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit-only", "final"], default="final")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--known-source", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--gata1", type=Path, default=DEFAULT_GATA1)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "ted_known_source_validation" / "reports" / "ted_upgrade_final_report.md",
    )
    args = parser.parse_args()

    registry = read_tsv(args.registry)
    if registry.empty:
        raise FileNotFoundError(f"Registry is missing or empty: {args.registry}")

    lines = audit_report(registry) if args.mode == "audit-only" else final_report(args.known_source, args.gata1, registry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
