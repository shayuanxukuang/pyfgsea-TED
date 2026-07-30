#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else "scripts/extract_bnt162b2_flagship_adt.R"
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
input <- if (length(args) >= 1L) args[[1L]] else file.path(root, "data_external", "bnt162b2_cite_asap_2023", "source", "PBMC_vaccine_CITE_seuratV5.rds")
outdir <- if (length(args) >= 2L) args[[2L]] else file.path(root, "data_external", "bnt162b2_cite_asap_2023", "adt_unmasked_export_v1")
rna_freeze_path <- file.path(root, "results", "ted_bnt162b2_flagship", "rna_event_freeze_v1", "rna_event_status.json")
selected_meta_path <- file.path(root, "data_external", "bnt162b2_cite_asap_2023", "rna_masked_export_v1", "selected_cell_metadata.tsv.gz")

if (!file.exists(input)) stop("Input RDS not found: ", input)
if (!file.exists(rna_freeze_path)) stop("RNA event freeze is missing; ADT remains masked")
if (!file.exists(selected_meta_path)) stop("Frozen RNA-only selected-cell metadata is missing")
if (dir.exists(outdir)) stop("ADT export is create-only and already exists: ", outdir)
rna_freeze <- fromJSON(rna_freeze_path, simplifyVector = TRUE)
if (!isTRUE(rna_freeze$rna_event_freeze_complete) || !isTRUE(rna_freeze$adt_unmask_allowed)) {
  stop("RNA freeze does not authorize ADT unmasking")
}
if (!identical(rna_freeze$outcome_values_accessed, FALSE)) stop("RNA freeze outcome-mask audit is inconsistent")

selected_meta <- fread(selected_meta_path, colClasses = list(character = c("cell_barcode", "donor_id")))
obj <- readRDS(input)
assay_names <- Assays(obj)
is_target_assay <- vapply(assay_names, function(name) {
  features <- rownames(obj[[name]])
  # Select the protein assay by literal antibody labels. RNA/SCT assays contain
  # the synonymous genes FCGR1A and SIGLEC1 and must not be mistaken for ADT.
  has_cd64 <- any(grepl("(^|[_-])CD64($|[_-])", features, ignore.case = TRUE))
  has_cd169 <- any(grepl("(^|[_-])CD169($|[_-])", features, ignore.case = TRUE))
  has_cd64 && has_cd169
}, logical(1L))
adt_candidates <- setdiff(assay_names[is_target_assay], assay_names[toupper(assay_names) == "RNA"])
if (length(adt_candidates) != 1L) {
  stop("Could not identify exactly one non-RNA assay containing CD64 and CD169: ", paste(adt_candidates, collapse = ", "))
}
adt_assay_name <- adt_candidates[[1L]]
adt_assay <- obj[[adt_assay_name]]
layers <- Layers(adt_assay)
preferred <- c("data", "counts")
chosen <- preferred[preferred %in% layers]
if (length(chosen)) {
  chosen <- chosen[[1L]]
} else {
  chosen <- layers[grepl("^(data|counts)(\\.|$)", layers)]
  if (length(chosen) != 1L) {
    stop("ADT assay must expose one unambiguous data or counts layer; found: ", paste(layers, collapse = ", "))
  }
  chosen <- chosen[[1L]]
}
layer_name <- chosen
adt <- LayerData(adt_assay, layer = layer_name, fast = FALSE)
missing_cells <- setdiff(selected_meta$cell_barcode, colnames(adt))
if (length(missing_cells)) stop("ADT layer omits ", length(missing_cells), " frozen RNA-selected cells")
adt <- adt[, selected_meta$cell_barcode, drop = FALSE]
if (!identical(colnames(adt), selected_meta$cell_barcode)) stop("ADT and frozen selected-cell order differ")

features <- rownames(adt)
find_unique <- function(pattern, label) {
  hit <- features[grepl(pattern, features, ignore.case = TRUE)]
  if (length(hit) != 1L) stop("Expected exactly one ", label, " ADT feature; found: ", paste(hit, collapse = ", "))
  hit[[1L]]
}
cd64_feature <- find_unique("(^|[_-])CD64($|[_-])|FCGR1A", "CD64")
cd169_feature <- find_unique("(^|[_-])CD169($|[_-])|SIGLEC1", "CD169")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
fwrite(selected_meta[, .(cell_barcode, donor_id, day)], file.path(outdir, "adt_cell_metadata.tsv.gz"), sep = "\t", compress = "gzip")
writeLines(features, file.path(outdir, "adt_features.txt"), useBytes = TRUE)
mtx_path <- file.path(outdir, "selected_adt_values.mtx")
writeMM(Matrix::Matrix(adt, sparse = TRUE), mtx_path)
R.utils::gzip(mtx_path, destname = paste0(mtx_path, ".gz"), overwrite = FALSE, remove = TRUE)

summary <- list(
  input = normalizePath(input, winslash = "/"),
  rna_freeze = normalizePath(rna_freeze_path, winslash = "/"),
  adt_assay = adt_assay_name,
  adt_layer = layer_name,
  values_are_seurat_normalized = identical(layer_name, "data"),
  n_features = nrow(adt),
  n_cells = ncol(adt),
  CD64_feature = cd64_feature,
  CD169_feature = cd169_feature,
  unmasked_only_after_rna_event_freeze = TRUE
)
write_json(summary, file.path(outdir, "adt_unmask_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Created ADT export after verified RNA freeze:", outdir, "\n")
