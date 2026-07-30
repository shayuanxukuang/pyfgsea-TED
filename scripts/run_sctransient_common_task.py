from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import wavelet_pseudotime.process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()

    counts = sparse.load_npz(input_dir / "raw_counts.npz")
    cells = pd.read_csv(input_dir / "cell_metadata.tsv", sep="\t")
    matrix = counts.toarray().astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    adata = ad.AnnData(matrix)
    adata.var_names = [f"gene_{i + 1:05d}" for i in range(matrix.shape[1])]
    adata.obs["psupertime"] = cells["ordered_coordinate"].to_numpy(dtype=float)
    adata.obs["phase"] = 1

    started = time.perf_counter()
    _waves, scores, _signals, _trimmed = wavelet_pseudotime.process.pipeline3(
        adata, scoring_threshold=0.2
    )
    elapsed = time.perf_counter() - started
    output = pd.DataFrame(
        {
            "gene": adata.var_names,
            "native_score": [float(scores.get(name, np.nan)) for name in adata.var_names],
        }
    )
    output["status"] = np.where(np.isfinite(output["native_score"]), "ok", "not_estimable")
    output.to_csv(input_dir / "sctransient_native_gene.tsv", sep="\t", index=False)
    pd.DataFrame([{"method": "scTransient", "elapsed_seconds": elapsed}]).to_csv(
        input_dir / "sctransient_runtime.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
