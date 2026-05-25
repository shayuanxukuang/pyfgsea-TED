from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWN = ROOT / "results" / "ted_known_source_validation" / "tables"
DEFAULT_GATA1 = ROOT / "results" / "gata1_cross_dataset_support" / "tables"
DEFAULT_OUT = ROOT / "results" / "ted_known_source_validation" / "figures"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def known_source_figure(known: Path, out: Path) -> None:
    gse153 = read_tsv(known / "gse153056_pdl1_outcome_alignment.tsv")
    gse937 = read_tsv(known / "gse93735_reversal_index.tsv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    if not gse153.empty:
        ax.scatter(gse153["event_effect_size"], gse153["pdl1_protein_effect_size"], s=16, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("TED IFNG/PD-L1 event effect")
    ax.set_ylabel("PD-L1 protein effect")
    ax.set_title("GSE153056")

    ax = axes[1]
    if not gse937.empty:
        plot = gse937.sort_values("reversal_fraction")
        ax.barh(plot["axis"], plot["reversal_fraction"], color="#5B8FF9")
    ax.axvline(0.2, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Reversal fraction")
    ax.set_title("GSE93735")
    fig.suptitle("TED Known-Source Validation")
    save(fig, out / "figure2_known_source_validation.png")


def gata1_figure(gata1: Path, out: Path) -> None:
    df = read_tsv(gata1 / "gata1_axis_direction_consistency.tsv")
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No GATA1 support table", ha="center", va="center")
        ax.axis("off")
        save(fig, out / "figure4_gata1_cross_dataset_support.png")
        return
    pivot = df.pivot_table(index="dataset", columns="axis", values="effect_size", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("GATA1 Cross-Dataset Axis Direction")
    fig.colorbar(im, ax=ax, label="Effect size")
    save(fig, out / "figure4_gata1_cross_dataset_support.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-source", type=Path, default=DEFAULT_KNOWN)
    parser.add_argument("--gata1", type=Path, default=DEFAULT_GATA1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    known_source_figure(args.known_source, args.out)
    gata1_figure(args.gata1, args.out)
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
