"""Outcome-blind expression preprocessing for formal T21 inference.

The functions in this module deliberately accept only a sparse raw-count
matrix, a frozen analysis-cell mask, and ordered gene identifiers.  They do
not inspect condition labels, pathway outcomes, or any other biological
metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


FORMAL_EXPRESSION_TARGET_SUM = 10_000.0
FORMAL_EXPRESSION_DTYPE = np.dtype(np.float32)
FORMAL_EXPRESSION_CONTRACT_SCHEMA = "t21_formal_expression_preprocessing"
FORMAL_EXPRESSION_CONTRACT_VERSION = "1.0.0"
FORMAL_EXPRESSION_CSR_SEMANTIC_SCHEMA = "t21_formal_expression_csr_semantic_sha256_v1"
POOLED_GENE_SUPPORT_SCHEMA = "t21_pooled_analysis_gene_support"
POOLED_GENE_SUPPORT_VERSION = "1.0.0"

_NORMALIZATION_ROW_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class PooledGeneSupport:
    """Pooled nonzero-gene support over a frozen analysis-cell mask."""

    ordered_gene_ids: tuple[str, ...]
    supported_mask: tuple[bool, ...]
    n_cells: int
    n_analysis_cells: int
    gene_order_sha256: str
    analysis_cell_mask_sha256_uint8: str
    support_mask_sha256_uint8: str
    gene_order_bound_support_sha256: str
    support_contract_sha256: str

    @property
    def n_genes(self) -> int:
        """Return the number of genes in the frozen expression order."""

        return len(self.ordered_gene_ids)

    @property
    def n_supported_genes(self) -> int:
        """Return the number of genes that are nonzero in the analysis pool."""

        return int(sum(self.supported_mask))

    @property
    def supported_gene_ids(self) -> tuple[str, ...]:
        """Return supported genes in the frozen expression-matrix order."""

        return tuple(
            gene
            for gene, supported in zip(self.ordered_gene_ids, self.supported_mask)
            if supported
        )

    def as_mask(self) -> np.ndarray:
        """Return an independent NumPy boolean mask for matrix indexing."""

        return np.asarray(self.supported_mask, dtype=bool)

    def to_contract_dict(self) -> dict[str, Any]:
        """Return the compact, stable support provenance contract."""

        result = _support_contract_payload(
            n_cells=self.n_cells,
            n_analysis_cells=self.n_analysis_cells,
            n_genes=self.n_genes,
            n_supported_genes=self.n_supported_genes,
            gene_order_sha256=self.gene_order_sha256,
            analysis_cell_mask_sha256_uint8=(self.analysis_cell_mask_sha256_uint8),
            support_mask_sha256_uint8=self.support_mask_sha256_uint8,
            gene_order_bound_support_sha256=(self.gene_order_bound_support_sha256),
        )
        result["support_contract_sha256"] = self.support_contract_sha256
        return result


@dataclass(frozen=True)
class FilteredGeneSets:
    """Gene sets filtered against pooled expression support before scoring."""

    gene_sets: dict[str, tuple[str, ...] | dict[str, float]]
    audit: tuple[dict[str, Any], ...]
    support_contract_sha256: str
    min_size: int
    max_size: int | None

    def as_mapping(self) -> dict[str, tuple[str, ...] | dict[str, float]]:
        """Return an independent mapping suitable for a pathway scorer."""

        copied: dict[str, tuple[str, ...] | dict[str, float]] = {}
        for pathway, members in self.gene_sets.items():
            copied[pathway] = dict(members) if isinstance(members, dict) else members
        return copied


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(values: Any, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    header = _canonical_json(
        {"dtype": np.dtype(dtype).str, "shape": list(map(int, array.shape))}
    ).encode("utf-8")
    return sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _strict_positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _strict_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty with no edge whitespace")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL characters")
    return value


def _validate_raw_count_values(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contain non-finite values")
    if np.any(values < 0):
        raise ValueError(f"{label} contain negative values")
    if (
        np.issubdtype(values.dtype, np.floating)
        and not np.equal(values, np.rint(values)).all()
    ):
        raise ValueError(f"{label} contain non-integer values")


def _canonicalize_raw_counts(counts: Any) -> sparse.csr_matrix:
    if not sparse.issparse(counts):
        raise TypeError("counts must be a scipy sparse raw-count matrix")
    if getattr(counts, "ndim", None) != 2:
        raise ValueError("counts must be two-dimensional")
    if counts.shape[0] < 1 or counts.shape[1] < 1:
        raise ValueError("counts must contain at least one cell and one gene")

    dtype = np.dtype(counts.dtype)
    if (
        dtype == np.dtype(bool)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise TypeError("counts must contain real numeric integer values")

    # Validate stored entries before duplicate collapse so that a negative or
    # non-finite duplicate cannot be hidden by another entry at the same index.
    # Common compressed/coordinate formats expose a numeric ndarray directly;
    # avoiding ``tocoo(copy=True)`` here prevents an unnecessary whole-matrix
    # allocation for production-scale inputs.
    stored = getattr(counts, "data", None)
    if isinstance(stored, np.ndarray) and np.issubdtype(stored.dtype, np.number):
        raw_values = stored
    else:
        raw_values = np.asarray(counts.tocoo(copy=False).data)
    _validate_raw_count_values(raw_values, "counts")

    already_canonical = (
        sparse.isspmatrix_csr(counts)
        and counts.has_canonical_format
        and counts.has_sorted_indices
        and not np.any(counts.data == 0)
    )
    if already_canonical:
        # All later operations are read-only, so a canonical CSR can be reused
        # without doubling the raw count matrix in memory.
        canonical = counts
    else:
        canonical = sparse.csr_matrix(counts, copy=True)
        canonical.sum_duplicates()
        canonical.sort_indices()
        canonical.eliminate_zeros()
        canonical.prune()

    values = np.asarray(canonical.data)
    _validate_raw_count_values(values, "counts after duplicate collapse")
    if np.any(values <= 0):
        raise ValueError("canonical raw-count nonzeros must be strictly positive")

    totals = np.asarray(canonical.sum(axis=1, dtype=np.float64)).ravel()
    if not np.isfinite(totals).all():
        raise ValueError("counts contain a row with a non-finite total")
    invalid_rows = np.flatnonzero(totals <= 0)
    if invalid_rows.size:
        preview = invalid_rows[:10].astype(int).tolist()
        raise ValueError(f"counts contain rows with total <= 0: {preview}")
    return canonical


def _normalized_log1p_data(counts: sparse.csr_matrix) -> np.ndarray:
    totals = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    scales = FORMAL_EXPRESSION_TARGET_SUM / totals
    result = np.empty(counts.nnz, dtype=FORMAL_EXPRESSION_DTYPE)
    indptr = counts.indptr
    for row_start in range(0, counts.shape[0], _NORMALIZATION_ROW_CHUNK_SIZE):
        row_stop = min(row_start + _NORMALIZATION_ROW_CHUNK_SIZE, counts.shape[0])
        data_start = int(indptr[row_start])
        data_stop = int(indptr[row_stop])
        repeats = np.diff(indptr[row_start : row_stop + 1])
        entry_scales = np.repeat(scales[row_start:row_stop], repeats)
        normalized = (
            np.asarray(counts.data[data_start:data_stop], dtype=np.float64)
            * entry_scales
        )
        result[data_start:data_stop] = np.log1p(normalized).astype(
            FORMAL_EXPRESSION_DTYPE, copy=False
        )
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(
            "float32 normalization produced a non-finite or zero stored value; "
            "the raw-count dynamic range is outside the formal contract"
        )
    return result


def _csr_row_sums(indptr: np.ndarray, values: np.ndarray) -> np.ndarray:
    n_rows = len(indptr) - 1
    result = np.zeros(n_rows, dtype=np.float64)
    nonempty = np.flatnonzero(np.diff(indptr) > 0)
    if nonempty.size:
        starts = indptr[nonempty]
        result[nonempty] = np.add.reduceat(np.asarray(values, dtype=np.float64), starts)
    return result


def formal_expression_float32_row_tolerances(
    expression: sparse.csr_matrix,
) -> np.ndarray:
    """Return per-row absolute tolerances for reconstructed totals.

    The bound sums, per stored value, the larger exponential deviation from
    the two float32 rounding-cell midpoints.  It then adds conservative
    float64 division, multiplication, transcendental-evaluation, and summation
    guards.  This is a value- and sparsity-dependent error envelope, not an
    arbitrary decimal ``rtol``.
    """

    if not sparse.isspmatrix_csr(expression):
        raise TypeError("expression must be a scipy.sparse.csr_matrix")
    if expression.dtype != FORMAL_EXPRESSION_DTYPE:
        raise TypeError("expression must have dtype float32")
    if not expression.has_canonical_format:
        raise ValueError("expression CSR indices must be canonical")
    values32 = np.asarray(expression.data, dtype=np.float32)
    if not np.isfinite(values32).all() or np.any(values32 <= 0):
        raise ValueError("expression stored values must be finite and positive")

    previous32 = np.nextafter(values32, np.float32(-np.inf))
    following32 = np.nextafter(values32, np.float32(np.inf))
    values64 = values32.astype(np.float64)
    lower_midpoint = (previous32.astype(np.float64) + values64) * 0.5
    upper_midpoint = (values64 + following32.astype(np.float64)) * 0.5
    reconstructed = np.expm1(values64)
    lower = np.expm1(lower_midpoint)
    upper = np.expm1(upper_midpoint)
    rounding_bound = np.maximum(reconstructed - lower, upper - reconstructed)
    row_rounding_bound = _csr_row_sums(expression.indptr, rounding_bound)

    epsilon = np.finfo(np.float64).eps
    row_nnz = np.diff(expression.indptr).astype(np.float64)
    operation_count = row_nnz + 8.0
    gamma = operation_count * epsilon / (1.0 - operation_count * epsilon)
    reconstructed_totals = _csr_row_sums(expression.indptr, reconstructed)
    arithmetic_guard = gamma * np.maximum(
        FORMAL_EXPRESSION_TARGET_SUM, reconstructed_totals
    )
    transcendental_entry_guard = (
        8.0 * epsilon * (np.abs(values64) + 1.0) * (reconstructed + 1.0)
    )
    transcendental_guard = _csr_row_sums(expression.indptr, transcendental_entry_guard)
    floor = 8.0 * np.spacing(np.float64(FORMAL_EXPRESSION_TARGET_SUM))
    return row_rounding_bound + arithmetic_guard + transcendental_guard + floor


def formal_expression_csr_semantic_sha256(expression: Any) -> str:
    """Hash the platform-independent semantic bytes of a formal CSR matrix.

    The digest input is exactly the UTF-8 schema tag followed by one NUL byte,
    then ``shape`` as two int64-le values, canonical ``indptr`` and ``indices``
    as int64-le, and ``data`` as float32-le.  Array lengths are unambiguous from
    the shape and terminal indptr value.  Native CSR index width therefore does
    not affect the digest.
    """

    if not sparse.isspmatrix_csr(expression):
        raise TypeError("expression must be a scipy.sparse.csr_matrix")
    dtype = np.dtype(expression.dtype)
    if dtype.kind != "f" or dtype.itemsize != 4:
        raise TypeError("expression must have dtype float32")
    if not expression.has_canonical_format or not expression.has_sorted_indices:
        raise ValueError("expression must be canonical sorted CSR")
    if not np.isfinite(expression.data).all():
        raise ValueError("expression contains non-finite values")

    digest = sha256()
    digest.update(FORMAL_EXPRESSION_CSR_SEMANTIC_SCHEMA.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        np.ascontiguousarray(np.asarray(expression.shape, dtype="<i8")).tobytes()
    )
    digest.update(
        np.ascontiguousarray(np.asarray(expression.indptr, dtype="<i8")).tobytes()
    )
    digest.update(
        np.ascontiguousarray(np.asarray(expression.indices, dtype="<i8")).tobytes()
    )
    digest.update(
        np.ascontiguousarray(np.asarray(expression.data, dtype="<f4")).tobytes()
    )
    return digest.hexdigest()


def _validate_prepared_expression(
    counts: sparse.csr_matrix,
    expression: Any,
    *,
    expected_data: np.ndarray | None = None,
) -> dict[str, Any]:
    if not sparse.isspmatrix_csr(expression):
        raise TypeError("expression must be a scipy.sparse.csr_matrix")
    if expression.dtype != FORMAL_EXPRESSION_DTYPE:
        raise TypeError("expression must have dtype float32")
    if expression.shape != counts.shape:
        raise ValueError("expression shape differs from raw counts")
    if not expression.has_canonical_format or not expression.has_sorted_indices:
        raise ValueError("expression must be canonical sorted CSR")
    values = np.asarray(expression.data)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("expression contains negative or non-finite values")
    if np.any(values == 0):
        raise ValueError("expression contains explicit stored zeros")
    if not np.array_equal(expression.indptr, counts.indptr) or not np.array_equal(
        expression.indices, counts.indices
    ):
        raise ValueError("expression zero pattern differs from raw counts")

    if expected_data is None:
        expected_data = _normalized_log1p_data(counts)
    if not np.array_equal(values, expected_data):
        raise ValueError(
            "expression values differ from the deterministic normalize_total/log1p "
            "transform"
        )

    reconstructed_values = np.expm1(values.astype(np.float64))
    reconstructed_totals = _csr_row_sums(expression.indptr, reconstructed_values)
    errors = np.abs(reconstructed_totals - FORMAL_EXPRESSION_TARGET_SUM)
    tolerances = formal_expression_float32_row_tolerances(expression)
    failing = np.flatnonzero(errors > tolerances)
    if failing.size:
        row = int(failing[0])
        raise ValueError(
            "expression reconstructed row total exceeds its float32 error "
            f"envelope at row {row}: error={errors[row]:.17g}, "
            f"tolerance={tolerances[row]:.17g}"
        )

    return {
        "schema_name": FORMAL_EXPRESSION_CONTRACT_SCHEMA,
        "schema_version": FORMAL_EXPRESSION_CONTRACT_VERSION,
        "shape": [int(expression.shape[0]), int(expression.shape[1])],
        "nnz": int(expression.nnz),
        "dtype": "float32",
        "target_sum": FORMAL_EXPRESSION_TARGET_SUM,
        "max_reconstructed_total_absolute_error": float(np.max(errors)),
        "max_float32_absolute_tolerance": float(np.max(tolerances)),
        "row_tolerances_sha256_float64_le": _array_sha256(tolerances, "<f8"),
        "expression_csr_semantic_sha256": (
            formal_expression_csr_semantic_sha256(expression)
        ),
        "contract_sha256": formal_expression_preprocessing_contract_sha256(),
        "implementation_source_sha256": (
            formal_expression_preprocessing_source_sha256()
        ),
    }


def normalize_t21_formal_expression(counts: Any) -> sparse.csr_matrix:
    """Normalize sparse raw integer counts to canonical float32 log1p CSR.

    Each cell is independently scaled to a raw-count total of exactly 10,000
    in real arithmetic.  Scaling and ``log1p`` use float64 intermediates; the
    stored matrix is rounded once to float32.  The input is never mutated.
    """

    canonical_counts = _canonicalize_raw_counts(counts)
    normalized_data = _normalized_log1p_data(canonical_counts)
    result = sparse.csr_matrix(
        (
            normalized_data,
            canonical_counts.indices.copy(),
            canonical_counts.indptr.copy(),
        ),
        shape=canonical_counts.shape,
        copy=False,
    )
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    result.prune()
    _validate_prepared_expression(
        canonical_counts, result, expected_data=normalized_data
    )
    return result


def validate_t21_formal_expression(counts: Any, expression: Any) -> dict[str, Any]:
    """Validate a formal expression matrix against its sparse raw counts."""

    canonical_counts = _canonicalize_raw_counts(counts)
    return _validate_prepared_expression(canonical_counts, expression)


def _strict_gene_ids(gene_ids: Sequence[str], n_genes: int) -> tuple[str, ...]:
    if isinstance(gene_ids, (str, bytes)):
        raise TypeError("gene_ids must be an ordered sequence, not a string")
    values = tuple(
        _strict_identifier(value, f"gene_ids[{index}]")
        for index, value in enumerate(gene_ids)
    )
    if len(values) != n_genes:
        raise ValueError("gene_ids length must equal the count-matrix column count")
    if len(values) != len(set(values)):
        raise ValueError("gene_ids must be unique in expression-matrix order")
    return values


def _strict_analysis_cell_mask(
    analysis_cell_mask: Sequence[bool] | np.ndarray, n_cells: int
) -> np.ndarray:
    mask = np.asarray(analysis_cell_mask)
    if mask.dtype != np.dtype(bool):
        raise TypeError("analysis_cell_mask must contain only booleans")
    if mask.ndim != 1 or len(mask) != n_cells:
        raise ValueError("analysis_cell_mask must have one value per count row")
    if not np.any(mask):
        raise ValueError("analysis_cell_mask must select at least one cell")
    return np.ascontiguousarray(mask)


def _support_contract_payload(
    *,
    n_cells: int,
    n_analysis_cells: int,
    n_genes: int,
    n_supported_genes: int,
    gene_order_sha256: str,
    analysis_cell_mask_sha256_uint8: str,
    support_mask_sha256_uint8: str,
    gene_order_bound_support_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_name": POOLED_GENE_SUPPORT_SCHEMA,
        "schema_version": POOLED_GENE_SUPPORT_VERSION,
        "support_rule": "positive_count_in_at_least_one_frozen_analysis_cell",
        "n_cells": int(n_cells),
        "n_analysis_cells": int(n_analysis_cells),
        "n_genes": int(n_genes),
        "n_supported_genes": int(n_supported_genes),
        "gene_order_sha256": gene_order_sha256,
        "analysis_cell_mask_sha256_uint8": analysis_cell_mask_sha256_uint8,
        "support_mask_sha256_uint8": support_mask_sha256_uint8,
        "gene_order_bound_support_sha256": gene_order_bound_support_sha256,
    }


def _finish_pooled_gene_support(
    *,
    supported: np.ndarray,
    mask: np.ndarray,
    genes: tuple[str, ...],
    n_cells: int,
) -> PooledGeneSupport:
    supported_tuple = tuple(bool(value) for value in supported)
    gene_order_hash = _json_sha256(
        {"schema": "ordered_gene_ids_v1", "ordered_gene_ids": list(genes)}
    )
    analysis_mask_hash = _array_sha256(mask.astype(np.uint8), "|u1")
    support_mask_hash = _array_sha256(supported.astype(np.uint8), "|u1")
    gene_order_bound_hash = _json_sha256(
        {
            "schema": "gene_order_bound_pooled_support_v1",
            "gene_order_sha256": gene_order_hash,
            "n_genes": len(genes),
            "support_mask_sha256_uint8": support_mask_hash,
        }
    )
    payload = _support_contract_payload(
        n_cells=n_cells,
        n_analysis_cells=int(np.sum(mask)),
        n_genes=len(genes),
        n_supported_genes=int(np.sum(supported)),
        gene_order_sha256=gene_order_hash,
        analysis_cell_mask_sha256_uint8=analysis_mask_hash,
        support_mask_sha256_uint8=support_mask_hash,
        gene_order_bound_support_sha256=gene_order_bound_hash,
    )
    return PooledGeneSupport(
        ordered_gene_ids=genes,
        supported_mask=supported_tuple,
        n_cells=n_cells,
        n_analysis_cells=int(np.sum(mask)),
        gene_order_sha256=gene_order_hash,
        analysis_cell_mask_sha256_uint8=analysis_mask_hash,
        support_mask_sha256_uint8=support_mask_hash,
        gene_order_bound_support_sha256=gene_order_bound_hash,
        support_contract_sha256=_json_sha256(payload),
    )


def _accumulate_supported_columns(
    counts: sparse.csr_matrix,
    selected_rows: np.ndarray,
    supported: np.ndarray,
) -> None:
    for row in np.flatnonzero(selected_rows):
        start = int(counts.indptr[row])
        stop = int(counts.indptr[row + 1])
        supported[counts.indices[start:stop]] = True


def compute_pooled_gene_support(
    counts: Any,
    analysis_cell_mask: Sequence[bool] | np.ndarray,
    gene_ids: Sequence[str],
) -> PooledGeneSupport:
    """Compute pooled nonzero-gene support from a frozen analysis-cell mask.

    Only the supplied boolean mask is consulted.  No condition, donor, pathway,
    or outcome metadata is accepted by this API.
    """

    canonical_counts = _canonicalize_raw_counts(counts)
    mask = _strict_analysis_cell_mask(analysis_cell_mask, canonical_counts.shape[0])
    genes = _strict_gene_ids(gene_ids, canonical_counts.shape[1])
    supported = np.zeros(canonical_counts.shape[1], dtype=bool)
    _accumulate_supported_columns(canonical_counts, mask, supported)
    return _finish_pooled_gene_support(
        supported=supported,
        mask=mask,
        genes=genes,
        n_cells=canonical_counts.shape[0],
    )


def compute_pooled_gene_support_chunked(
    row_reader: Callable[[int, int], Any],
    *,
    n_cells: int,
    n_genes: int,
    analysis_cell_mask: Sequence[bool] | np.ndarray,
    gene_ids: Sequence[str],
    chunk_size: int = _NORMALIZATION_ROW_CHUNK_SIZE,
) -> PooledGeneSupport:
    """Compute pooled support from strict contiguous sparse row chunks.

    ``row_reader(start, stop)`` is called once for every contiguous half-open
    row interval covering ``[0, n_cells)``.  It must return a scipy sparse raw
    integer-count matrix with shape ``(stop - start, n_genes)``; wrappers around
    backed AnnData layers should materialize only that requested block before
    returning it.  Every row is validated, including rows outside the frozen
    analysis mask, but no full count matrix is materialized by this function.
    """

    if not callable(row_reader):
        raise TypeError("row_reader must be callable")
    cell_count = _strict_positive_integer(n_cells, "n_cells")
    gene_count = _strict_positive_integer(n_genes, "n_genes")
    block_size = _strict_positive_integer(chunk_size, "chunk_size")
    mask = _strict_analysis_cell_mask(analysis_cell_mask, cell_count)
    genes = _strict_gene_ids(gene_ids, gene_count)
    supported = np.zeros(gene_count, dtype=bool)

    for start in range(0, cell_count, block_size):
        stop = min(start + block_size, cell_count)
        raw_block = row_reader(start, stop)
        block = _canonicalize_raw_counts(raw_block)
        expected_shape = (stop - start, gene_count)
        if block.shape != expected_shape:
            raise ValueError(
                "row_reader returned shape "
                f"{block.shape}, expected {expected_shape} for rows [{start}, {stop})"
            )
        _accumulate_supported_columns(block, mask[start:stop], supported)

    return _finish_pooled_gene_support(
        supported=supported,
        mask=mask,
        genes=genes,
        n_cells=cell_count,
    )


def _validate_support_contract(support: PooledGeneSupport) -> None:
    if not isinstance(support, PooledGeneSupport):
        raise TypeError("support must be a PooledGeneSupport")
    genes = _strict_gene_ids(support.ordered_gene_ids, len(support.supported_mask))
    mask = np.asarray(support.supported_mask, dtype=np.uint8)
    gene_order_hash = _json_sha256(
        {"schema": "ordered_gene_ids_v1", "ordered_gene_ids": list(genes)}
    )
    support_mask_hash = _array_sha256(mask, "|u1")
    bound_hash = _json_sha256(
        {
            "schema": "gene_order_bound_pooled_support_v1",
            "gene_order_sha256": gene_order_hash,
            "n_genes": len(genes),
            "support_mask_sha256_uint8": support_mask_hash,
        }
    )
    payload = _support_contract_payload(
        n_cells=support.n_cells,
        n_analysis_cells=support.n_analysis_cells,
        n_genes=len(genes),
        n_supported_genes=int(np.sum(mask)),
        gene_order_sha256=gene_order_hash,
        analysis_cell_mask_sha256_uint8=(support.analysis_cell_mask_sha256_uint8),
        support_mask_sha256_uint8=support_mask_hash,
        gene_order_bound_support_sha256=bound_hash,
    )
    if (
        support.gene_order_sha256 != gene_order_hash
        or support.support_mask_sha256_uint8 != support_mask_hash
        or support.gene_order_bound_support_sha256 != bound_hash
        or support.support_contract_sha256 != _json_sha256(payload)
    ):
        raise ValueError("PooledGeneSupport hash contract does not match its contents")


def _strict_weight(value: Any, pathway: str, gene: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(
            f"Weight for gene {gene!r} in pathway {pathway!r} must be real numeric"
        )
    result = float(value)
    if not np.isfinite(result) or result == 0.0:
        raise ValueError(
            f"Weight for gene {gene!r} in pathway {pathway!r} must be finite "
            "and nonzero"
        )
    return result


def filter_gene_sets_for_supported_expression(
    gene_sets: Mapping[str, Sequence[str] | Mapping[str, Real]],
    support: PooledGeneSupport,
    *,
    min_size: int,
    max_size: int | None = None,
) -> FilteredGeneSets:
    """Filter pooled-all-zero genes before size gates and weight denominators.

    Weighted pathways must be mappings from gene ID to finite nonzero weight.
    Unweighted pathways must be unique gene-ID sequences.  Surviving members
    are returned in the frozen expression-gene order.  The audit's absolute
    weight denominator is computed *after* pooled-support filtering.
    """

    if not isinstance(gene_sets, Mapping):
        raise TypeError("gene_sets must be a pathway mapping")
    _validate_support_contract(support)
    minimum = _strict_positive_integer(min_size, "min_size")
    maximum = (
        None if max_size is None else _strict_positive_integer(max_size, "max_size")
    )
    if maximum is not None and maximum < minimum:
        raise ValueError("max_size must be at least min_size")

    rank = {gene: index for index, gene in enumerate(support.ordered_gene_ids)}
    supported = {
        gene
        for gene, is_supported in zip(support.ordered_gene_ids, support.supported_mask)
        if is_supported
    }
    pathway_items: list[tuple[str, Sequence[str] | Mapping[str, Real]]] = []
    seen_pathways: set[str] = set()
    for raw_pathway, members in gene_sets.items():
        pathway = _strict_identifier(raw_pathway, "pathway name")
        if pathway in seen_pathways:
            raise ValueError(f"Duplicate pathway name {pathway!r}")
        seen_pathways.add(pathway)
        pathway_items.append((pathway, members))

    filtered: dict[str, tuple[str, ...] | dict[str, float]] = {}
    audit: list[dict[str, Any]] = []
    for pathway, members in sorted(pathway_items, key=lambda item: item[0]):
        weighted = isinstance(members, Mapping)
        if weighted:
            weighted_members: dict[str, float] = {}
            for raw_gene, raw_weight in members.items():
                gene = _strict_identifier(raw_gene, f"gene in pathway {pathway!r}")
                if gene in weighted_members:
                    raise ValueError(f"Duplicate gene {gene!r} in pathway {pathway!r}")
                weighted_members[gene] = _strict_weight(raw_weight, pathway, gene)
            input_genes = tuple(weighted_members)
        else:
            if isinstance(members, (str, bytes)):
                raise TypeError(
                    f"Pathway {pathway!r} members must be a sequence, not a string"
                )
            input_genes = tuple(
                _strict_identifier(gene, f"gene in pathway {pathway!r}")
                for gene in members
            )
            if len(input_genes) != len(set(input_genes)):
                raise ValueError(f"Pathway {pathway!r} contains duplicate genes")
            weighted_members = {}

        in_gene_order = [gene for gene in input_genes if gene in rank]
        surviving = sorted(
            (gene for gene in in_gene_order if gene in supported), key=rank.__getitem__
        )
        denominator = float(
            sum(abs(weighted_members[gene]) for gene in surviving)
            if weighted
            else len(surviving)
        )
        if len(surviving) < minimum:
            status = "dropped_below_min_size"
        elif maximum is not None and len(surviving) > maximum:
            status = "dropped_above_max_size"
        else:
            status = "retained"
            if denominator <= 0:
                raise RuntimeError("Internal non-positive post-support denominator")
            filtered[pathway] = (
                {gene: weighted_members[gene] for gene in surviving}
                if weighted
                else tuple(surviving)
            )
        audit.append(
            {
                "pathway": pathway,
                "weighted": bool(weighted),
                "n_input_members": len(input_genes),
                "n_absent_from_gene_order": len(input_genes) - len(in_gene_order),
                "n_pooled_all_zero": len(in_gene_order) - len(surviving),
                "n_after_pooled_support": len(surviving),
                "absolute_weight_denominator_after_pooled_support": denominator,
                "min_size_applied_after_pooled_support": minimum,
                "max_size_applied_after_pooled_support": maximum,
                "status": status,
            }
        )

    return FilteredGeneSets(
        gene_sets=filtered,
        audit=tuple(audit),
        support_contract_sha256=support.support_contract_sha256,
        min_size=minimum,
        max_size=maximum,
    )


def formal_expression_preprocessing_source_sha256() -> str:
    """Return SHA-256 of canonical UTF-8/LF implementation source text."""

    source = Path(__file__).read_text(encoding="utf-8")
    canonical_source = source.replace("\r\n", "\n").replace("\r", "\n")
    return sha256(canonical_source.encode("utf-8")).hexdigest()


def formal_expression_preprocessing_contract() -> dict[str, Any]:
    """Return a fresh, canonicalizable formal preprocessing contract."""

    return {
        "schema_name": FORMAL_EXPRESSION_CONTRACT_SCHEMA,
        "schema_version": FORMAL_EXPRESSION_CONTRACT_VERSION,
        "implementation_source_sha256": (
            formal_expression_preprocessing_source_sha256()
        ),
        "implementation_source_hash_serialization": "utf8_text_newlines_canonical_lf",
        "input": {
            "container": "scipy_sparse",
            "semantics": "raw_integer_counts",
            "required_checks": [
                "real_numeric",
                "finite",
                "nonnegative",
                "integer_valued",
                "every_row_total_strictly_positive",
            ],
        },
        "transform": {
            "output_container": "scipy.sparse.csr_matrix",
            "target_sum": FORMAL_EXPRESSION_TARGET_SUM,
            "operation_order": [
                "canonicalize_raw_counts",
                "per_cell_normalize_total_float64",
                "log1p_nonzero_float64",
                "round_once_to_float32",
                "sum_duplicates",
                "sort_indices",
                "eliminate_zeros",
            ],
            "output_dtype": "float32",
            "stored_zero_policy": "forbidden",
            "row_chunk_size": _NORMALIZATION_ROW_CHUNK_SIZE,
        },
        "validation": {
            "shape": "exactly_equal_to_counts",
            "zero_pattern": "canonical_csr_indptr_and_indices_exactly_equal",
            "values": "finite_nonnegative_and_exact_deterministic_transform",
            "row_total_check": "sum_float64_expm1_stored_values_equals_10000",
            "row_total_tolerance": (
                "per_value_float32_rounding_cell_exponential_bound_plus_"
                "float64_arithmetic_and_transcendental_guards"
            ),
            "expression_csr_semantic_sha256": {
                "schema_tag": FORMAL_EXPRESSION_CSR_SEMANTIC_SCHEMA,
                "byte_order": [
                    "schema_tag_utf8_then_nul",
                    "shape_int64_le",
                    "canonical_indptr_int64_le",
                    "canonical_indices_int64_le",
                    "data_float32_le",
                ],
            },
        },
        "gene_support": {
            "selection_input": "frozen_boolean_analysis_cell_mask_only",
            "support_rule": "positive_count_in_at_least_one_selected_cell",
            "hash_binding": "ordered_gene_ids_plus_pooled_support_mask",
            "backed_reader_contract": (
                "contiguous_sparse_raw_count_rows_covering_every_input_row"
            ),
            "gene_set_order": [
                "filter_absent_and_pooled_all_zero_genes",
                "apply_min_size_and_max_size",
                "compute_absolute_weight_denominator",
            ],
        },
        "forbidden_inputs": [
            "condition",
            "donor_condition_assignment",
            "pathway_outcomes",
            "pathway_statistics",
        ],
    }


def formal_expression_preprocessing_contract_sha256() -> str:
    """Return SHA-256 of canonical JSON for the current contract."""

    return _json_sha256(formal_expression_preprocessing_contract())


__all__ = [
    "FORMAL_EXPRESSION_CONTRACT_SCHEMA",
    "FORMAL_EXPRESSION_CONTRACT_VERSION",
    "FORMAL_EXPRESSION_CSR_SEMANTIC_SCHEMA",
    "FORMAL_EXPRESSION_DTYPE",
    "FORMAL_EXPRESSION_TARGET_SUM",
    "FilteredGeneSets",
    "PooledGeneSupport",
    "compute_pooled_gene_support",
    "compute_pooled_gene_support_chunked",
    "filter_gene_sets_for_supported_expression",
    "formal_expression_csr_semantic_sha256",
    "formal_expression_float32_row_tolerances",
    "formal_expression_preprocessing_contract",
    "formal_expression_preprocessing_contract_sha256",
    "formal_expression_preprocessing_source_sha256",
    "normalize_t21_formal_expression",
    "validate_t21_formal_expression",
]
