from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scp1064_utils import GLOBAL_FIGURES, GLOBAL_TABLES, RESULTS


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def main() -> None:
    outcome = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    author = read_tsv(RESULTS / "scp1064_author_effect_summary.tsv")
    claim = read_tsv(RESULTS / "scp1064_claim_boundary.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if not outcome.empty and "spearman" in outcome:
        plot = outcome.dropna(subset=["spearman"]).copy()
        plot["label"] = plot["level"].astype(str) + ":" + plot["axis"].astype(str)
        ax.barh(plot["label"].head(18), plot["spearman"].head(18), color="#4C78A8")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("RNA event/protein Spearman")
    ax.set_title("SCP1064 outcome alignment")
    ax = axes[1]
    if not author.empty and "max_abs_spearman" in author:
        ax.barh(author["axis"], author["max_abs_spearman"], color="#F58518")
    ax.set_xlabel("Author reference max |rho|")
    ax.set_title("Author effect concordance")
    title = "SCP1064 validation"
    if not claim.empty:
        title += f": {claim.iloc[0]['claim_boundary']}"
    fig.suptitle(title)
    GLOBAL_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(GLOBAL_FIGURES / "supplementary_figure_scp1064.png", dpi=180, bbox_inches="tight")
    fig.savefig(GLOBAL_FIGURES / "supplementary_figure_scp1064.pdf", bbox_inches="tight")
    plt.close(fig)

    claims = read_tsv(GLOBAL_TABLES / "ted_dataset_level_claim_boundary.tsv")
    gse153 = read_tsv(GLOBAL_TABLES / "gse153056_pdl1_outcome_alignment.tsv")
    gse937 = read_tsv(GLOBAL_TABLES / "gse93735_reversal_index.tsv")
    scp = read_tsv(RESULTS / "scp1064_outcome_alignment_summary.tsv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.axis("off")
    if not claims.empty:
        subset = claims[claims["dataset"].isin(["GSE153056", "GSE93735", "SCP1064"])]
        lines = ["Dataset       Claim boundary"]
        for _, row in subset.iterrows():
            lines.append(f"{row['dataset']:<12} {row['claim_boundary']}")
        ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("A. Claim-boundary registry")

    ax = axes[0, 1]
    if not gse153.empty:
        ax.scatter(gse153["event_effect_size"], gse153["pdl1_protein_effect_size"], s=18, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("TED IFNG/PD-L1 event")
    ax.set_ylabel("PD-L1 protein outcome")
    ax.set_title("B. GSE153056")

    ax = axes[1, 0]
    if not gse937.empty:
        plot = gse937.sort_values("reversal_fraction")
        ax.barh(plot["axis"], plot["reversal_fraction"], color="#5B8FF9")
    ax.axvline(0.2, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Reversal fraction")
    ax.set_title("C. GSE93735")

    ax = axes[1, 1]
    if not scp.empty:
        plot = scp.dropna(subset=["spearman"]).copy().head(12)
        labels = plot["level"].astype(str) + ":" + plot["axis"].astype(str) + ":" + plot["protein_name"].astype(str)
        ax.barh(labels, plot["spearman"], color="#54A24B")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("RNA event/protein rho")
    ax.set_title("D. SCP1064")
    fig.suptitle("TED Known-Source Validation")
    fig.tight_layout()
    fig.savefig(GLOBAL_FIGURES / "figure2_known_source_validation.png", dpi=180, bbox_inches="tight")
    fig.savefig(GLOBAL_FIGURES / "figure2_known_source_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {GLOBAL_FIGURES / 'supplementary_figure_scp1064.png'}")


if __name__ == "__main__":
    main()
