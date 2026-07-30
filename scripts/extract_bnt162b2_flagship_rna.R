#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(data.table)
  library(jsonlite)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else "scripts/extract_bnt162b2_flagship_rna.R"
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
input <- if (length(args) >= 1L) args[[1L]] else file.path(root, "data_external", "bnt162b2_cite_asap_2023", "source", "PBMC_vaccine_CITE_seuratV5.rds")
outdir <- if (length(args) >= 2L) args[[2L]] else file.path(root, "data_external", "bnt162b2_cite_asap_2023", "rna_masked_export_v1")
donor_override <- if (length(args) >= 3L && nzchar(args[[3L]])) args[[3L]] else NULL
day_override <- if (length(args) >= 4L && nzchar(args[[4L]])) args[[4L]] else NULL

config <- read_yaml(file.path(root, "config", "ted_bnt162b2_flagship_v1.yaml"))
gmt_path <- file.path(root, "results", "ted_bnt162b2_flagship", "protocol_freeze_v1", "locked_pathway_family.gmt")
if (!file.exists(input)) stop("Input RDS not found: ", input)
if (!file.exists(gmt_path)) stop("Primary pathway freeze not found: ", gmt_path)
if (dir.exists(outdir)) stop("RNA masked export is create-only and already exists: ", outdir)

obj <- readRDS(input)
meta <- obj[[]]
meta$cell_barcode <- rownames(meta)
assay_names <- Assays(obj)
rna_candidates <- assay_names[toupper(assay_names) == "RNA"]
if (!length(rna_candidates)) rna_candidates <- assay_names[grepl("RNA", assay_names, ignore.case = TRUE)]
if (length(rna_candidates) != 1L) stop("Could not identify exactly one RNA assay: ", paste(assay_names, collapse = ", "))
rna_assay_name <- rna_candidates[[1L]]
rna_assay <- obj[[rna_assay_name]]
all_rna_layers <- Layers(rna_assay)
count_layers <- all_rna_layers[grepl("^counts($|\\.)", all_rna_layers)]
if (!length(count_layers)) stop("No raw RNA counts layer found in assay ", rna_assay_name)
count_parts <- lapply(count_layers, function(layer_name) {
  LayerData(rna_assay, layer = layer_name, fast = FALSE)
})
counts <- if (length(count_parts) == 1L) count_parts[[1L]] else do.call(cbind, count_parts)
if (anyDuplicated(colnames(counts))) stop("RNA count layers contain duplicate cell names")
missing_cells <- setdiff(colnames(obj), colnames(counts))
if (length(missing_cells)) stop("RNA count layers omit ", length(missing_cells), " object cells")
counts <- counts[, colnames(obj), drop = FALSE]
if (!identical(colnames(counts), rownames(meta))) stop("RNA counts and metadata are not aligned")

parse_day <- function(value) {
  text <- as.character(value)
  direct <- suppressWarnings(as.integer(text))
  # The paper reports the nominal recovery draw as day 10 with 1--2 days of
  # scheduling flexibility, whereas the frozen Zenodo object labels that draw
  # Day11.  This metadata-only alignment was recorded before any assay layer
  # value was requested; it does not alter the frozen contrast weight.
  normalized <- tolower(trimws(text))
  nominal_lookup <- c(day0 = 0L, day2 = 2L, day10 = 10L, day11 = 10L, day28 = 28L)
  nominal <- unname(nominal_lookup[normalized])
  direct[is.na(direct)] <- nominal[is.na(direct)]
  day_match <- regmatches(text, regexpr("(?i)(?:day|d)[ _-]?(0|2|10|28)(?:$|[^0-9])", text, perl = TRUE))
  parsed <- suppressWarnings(as.integer(gsub("[^0-9]", "", day_match)))
  direct[is.na(direct)] <- parsed[is.na(direct)]
  direct
}

choose_day_column <- function(meta, override = NULL) {
  if (!is.null(override)) {
    if (!override %in% names(meta)) stop("Requested day column not found: ", override)
    return(override)
  }
  priority <- c("day", "Day", "timepoint", "Timepoint", "time", "Time", "orig.ident", "sample", "sample_id")
  candidates <- unique(c(priority[priority %in% names(meta)], names(meta)))
  target <- c(0L, 2L, 10L, 28L)
  valid <- candidates[vapply(candidates, function(name) {
    parsed <- parse_day(meta[[name]])
    all(target %in% unique(parsed[!is.na(parsed)])) && length(unique(parsed[!is.na(parsed)])) <= 8L
  }, logical(1L))]
  if (!length(valid)) stop("No metadata column encodes all frozen days 0, 2, 10 and 28")
  valid[[1L]]
}

choose_donor_column <- function(meta, day, override = NULL) {
  if (!is.null(override)) {
    if (!override %in% names(meta)) stop("Requested donor column not found: ", override)
    return(override)
  }
  priority <- c("donor_id", "donor", "Donor", "subject", "Subject", "participant", "patient")
  candidates <- unique(c(priority[priority %in% names(meta)], names(meta)))
  valid <- candidates[vapply(candidates, function(name) {
    value <- as.character(meta[[name]])
    keep <- !is.na(day) & day %in% c(0L, 2L, 10L, 28L) & !is.na(value) & nzchar(value)
    if (length(unique(value[keep])) != 6L) return(FALSE)
    coverage <- tapply(day[keep], value[keep], function(x) length(unique(x)))
    length(coverage) == 6L && all(coverage == 4L)
  }, logical(1L))]
  if (!length(valid)) stop("No metadata column identifies six donors with all four frozen days")
  valid[[1L]]
}

day_col <- choose_day_column(meta, day_override)
meta$day <- parse_day(meta[[day_col]])
donor_col <- choose_donor_column(meta, meta$day, donor_override)
meta$donor_id <- as.character(meta[[donor_col]])

gmt_lines <- readLines(gmt_path, warn = FALSE)
pathway_genes <- unique(unlist(lapply(strsplit(gmt_lines, "\t", fixed = TRUE), function(x) x[-c(1L, 2L)])))
forbidden <- unique(c(
  pathway_genes,
  unlist(config$population$forbidden_annotation_features, use.names = FALSE)
))
panels <- lapply(config$population$marker_panels, function(genes) {
  intersect(setdiff(unlist(genes, use.names = FALSE), forbidden), rownames(counts))
})
if (length(panels$CD14_like_monocyte) < 5L) stop("Fewer than five usable non-IFN CD14-like markers")
marker_genes <- sort(unique(unlist(panels, use.names = FALSE)))
library_size <- Matrix::colSums(counts)
detected_genes <- Matrix::colSums(counts > 0)
if (any(library_size <= 0)) stop("At least one cell has zero RNA library size")
marker_dense <- as.matrix(counts[marker_genes, , drop = FALSE])
marker_dense <- log1p(sweep(marker_dense, 2L, library_size / 10000, "/"))
marker_mean <- rowMeans(marker_dense)
marker_sd <- apply(marker_dense, 1L, sd)
marker_sd[!is.finite(marker_sd) | marker_sd <= .Machine$double.eps] <- 1
marker_z <- sweep(sweep(marker_dense, 1L, marker_mean, "-"), 1L, marker_sd, "/")
panel_scores <- vapply(panels, function(genes) colMeans(marker_z[genes, , drop = FALSE]), numeric(ncol(counts)))
if (is.null(dim(panel_scores))) panel_scores <- matrix(panel_scores, ncol = length(panels))
colnames(panel_scores) <- names(panels)
cd14_score <- panel_scores[, "CD14_like_monocyte"]
competitor <- apply(panel_scores[, setdiff(colnames(panel_scores), "CD14_like_monocyte"), drop = FALSE], 1L, max)
score_margin <- cd14_score - competitor
# CD14 belongs to a frozen pathway-family member and is therefore excluded from
# the multi-gene annotation score.  The protocol nevertheless declares a
# separate low CD14 lineage-inclusion threshold, evaluated here without adding
# CD14 back to that score.
if (!"CD14" %in% rownames(counts)) stop("Frozen CD14 lineage gate cannot be evaluated")
cd14_normalized <- log1p(as.numeric(counts["CD14", ]) / (library_size / 10000))

meta$total_rna_umi <- as.numeric(library_size)
meta$detected_rna_genes <- as.numeric(detected_genes)
target_meta <- meta[meta$day %in% c(0L, 2L, 10L, 28L) & !is.na(meta$donor_id), , drop = FALSE]
sample_qc <- as.data.table(target_meta)[, .(
  n_cells = .N,
  median_rna_umi = median(total_rna_umi),
  median_detected_genes = median(detected_rna_genes)
), by = .(donor_id, day)]
robust_abs_z <- function(x) {
  scale <- mad(x, center = median(x), constant = 1.4826)
  if (!is.finite(scale) || scale <= .Machine$double.eps) return(rep(0, length(x)))
  abs(x - median(x)) / scale
}
sample_qc[, median_rna_umi_abs_mad_z := robust_abs_z(median_rna_umi)]
sample_qc[, median_detected_genes_abs_mad_z := robust_abs_z(median_detected_genes)]
sample_qc[, blind_qc_pass := median_rna_umi_abs_mad_z <= 3 & median_detected_genes_abs_mad_z <= 3]
donor_qc <- sample_qc[, .(
  n_timepoints = uniqueN(day),
  all_sample_qc_pass = all(blind_qc_pass)
), by = donor_id]
qc_donors <- donor_qc[n_timepoints == 4L & all_sample_qc_pass == TRUE, donor_id]

selected <- score_margin >= as.numeric(config$population$minimum_score_margin) &
  cd14_normalized >= as.numeric(config$population$minimum_cd14_log_normalized) &
  meta$day %in% c(0L, 2L, 10L, 28L) & meta$donor_id %in% qc_donors
population_counts <- as.data.table(meta[selected, , drop = FALSE])[, .N, by = .(donor_id, day)]
setnames(population_counts, "N", "selected_cells")
donor_cells <- population_counts[, .(
  n_timepoints = uniqueN(day),
  all_cells_pass = all(selected_cells >= as.integer(config$population$minimum_cells_per_donor_time))
), by = donor_id]
evaluable_donors <- donor_cells[n_timepoints == 4L & all_cells_pass == TRUE, donor_id]
selected <- selected & meta$donor_id %in% evaluable_donors
if (length(evaluable_donors) < as.integer(config$design$minimum_evaluable_donors)) {
  stop("Only ", length(evaluable_donors), " donors pass blind QC and RNA-only population gates")
}

selected_meta <- meta[selected, c("cell_barcode", "donor_id", "day", "total_rna_umi", "detected_rna_genes"), drop = FALSE]
selected_meta$annotation_margin <- score_margin[selected]
selected_meta$cd14_log_normalized <- cd14_normalized[selected]
mt_genes <- grep("^MT-", rownames(counts), value = TRUE)
selected_meta$mitochondrial_fraction <- if (length(mt_genes)) {
  as.numeric(Matrix::colSums(counts[mt_genes, selected, drop = FALSE]) / library_size[selected])
} else rep(0, sum(selected))
selected_counts <- counts[, selected, drop = FALSE]
if (!identical(colnames(selected_counts), selected_meta$cell_barcode)) stop("Selected RNA counts and metadata are misaligned")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
fwrite(sample_qc, file.path(outdir, "sample_blind_qc.tsv"), sep = "\t")
fwrite(population_counts, file.path(outdir, "rna_only_population_counts.tsv"), sep = "\t")
fwrite(selected_meta, file.path(outdir, "selected_cell_metadata.tsv.gz"), sep = "\t", compress = "gzip")
writeLines(rownames(selected_counts), file.path(outdir, "rna_features.txt"), useBytes = TRUE)
mtx_path <- file.path(outdir, "selected_rna_counts.mtx")
writeMM(selected_counts, mtx_path)
R.utils::gzip(mtx_path, destname = paste0(mtx_path, ".gz"), overwrite = FALSE, remove = TRUE)

summary <- list(
  input = normalizePath(input, winslash = "/"),
  object_class = class(obj),
  rna_assay = rna_assay_name,
  rna_count_layers = count_layers,
  donor_metadata_column = donor_col,
  day_metadata_column = day_col,
  source_timepoint_mapping = c(Day0 = 0L, Day2 = 2L, Day11 = 10L, Day28 = 28L),
  frozen_days = c(0L, 2L, 10L, 28L),
  evaluable_donors = sort(evaluable_donors),
  n_evaluable_donors = length(evaluable_donors),
  selected_rna_only_cd14_like_cells = sum(selected),
  adt_assay_values_accessed = FALSE,
  annotation_excluded_pathway_and_outcome_genes = TRUE
)
write_json(summary, file.path(outdir, "rna_masked_export.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Created RNA-only masked export without requesting ADT values:", outdir, "\n")
