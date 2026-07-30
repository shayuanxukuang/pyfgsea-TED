#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1L]]) else "scripts/inspect_bnt162b2_cite.R"
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
input <- if (length(args) >= 1L) args[[1L]] else file.path(root, "data_external", "bnt162b2_cite_asap_2023", "source", "PBMC_vaccine_CITE_seuratV5.rds")
outdir <- if (length(args) >= 2L) args[[2L]] else file.path(root, "data_external", "bnt162b2_cite_asap_2023", "structure_audit")

if (!file.exists(input)) stop("Input RDS not found: ", input)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(input)
meta <- obj[[]]
assay_names <- Assays(obj)

layer_rows <- list()
feature_rows <- list()
target_patterns <- c(
  "CD64", "FCGR1A", "CD169", "SIGLEC1", "CD14", "LYZ", "S100A8",
  "FCGR3A", "MS4A7", "ISG15", "IFI6", "IFITM3", "MX1", "STAT1"
)
for (assay_name in assay_names) {
  assay <- obj[[assay_name]]
  layers <- tryCatch(Layers(assay), error = function(e) character())
  if (!length(layers)) layers <- NA_character_
  for (layer_name in layers) {
    layer_rows[[length(layer_rows) + 1L]] <- data.frame(
      assay = assay_name,
      layer = layer_name,
      n_features = nrow(assay),
      n_cells = ncol(assay),
      matrix_class = "not_materialized_during_structure_audit",
      stringsAsFactors = FALSE
    )
  }
  features <- rownames(assay)
  matched <- features[Reduce(`|`, lapply(target_patterns, function(x) grepl(x, features, ignore.case = TRUE, fixed = TRUE)))]
  if (length(matched)) {
    feature_rows[[length(feature_rows) + 1L]] <- data.frame(
      assay = assay_name,
      feature = matched,
      stringsAsFactors = FALSE
    )
  }
}

meta_rows <- lapply(names(meta), function(name) {
  value <- meta[[name]]
  unique_values <- unique(as.character(value[!is.na(value)]))
  data.frame(
    column = name,
    storage_mode = typeof(value),
    class = paste(class(value), collapse = ";"),
    n_unique = length(unique_values),
    examples = paste(head(unique_values, 20L), collapse = " | "),
    stringsAsFactors = FALSE
  )
})

fwrite(rbindlist(meta_rows, fill = TRUE), file.path(outdir, "metadata_columns.tsv"), sep = "\t")
fwrite(rbindlist(layer_rows, fill = TRUE), file.path(outdir, "assay_layers.tsv"), sep = "\t")
fwrite(rbindlist(feature_rows, fill = TRUE), file.path(outdir, "target_feature_matches.tsv"), sep = "\t")

candidate_columns <- names(meta)[grepl(
  "donor|patient|subject|sample|orig.ident|time|day|celltype|cell_type|annot|batch|lane|ncount|nfeature|percent.mt|mito",
  names(meta), ignore.case = TRUE
)]
for (name in candidate_columns) {
  counts <- as.data.frame(table(value = as.character(meta[[name]]), useNA = "ifany"), stringsAsFactors = FALSE)
  counts$column <- name
  fwrite(counts[, c("column", "value", "Freq")], file.path(outdir, paste0("metadata_counts__", gsub("[^A-Za-z0-9_.-]", "_", name), ".tsv")), sep = "\t")
}

summary <- list(
  input = normalizePath(input, winslash = "/"),
  object_class = class(obj),
  n_cells = ncol(obj),
  assays = assay_names,
  default_assay = DefaultAssay(obj),
  metadata_columns = names(meta),
  candidate_design_columns = candidate_columns,
  inspected_without_accessing_assay_values = TRUE,
  note = "The object was deserialized, but no RNA or ADT layer matrix was requested or summarized."
)
write_json(summary, file.path(outdir, "structure_audit.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Wrote BNT162b2 CITE structure audit to", outdir, "\n")
