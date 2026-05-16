args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NA_character_) {
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) {
    return(default)
  }
  args[[hit + 1]]
}

outdir <- get_arg("--outdir", "data_external/ted_development_phase4_benchmark/direct_external_baseline")
expr_path <- get_arg("--expression")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(outdir, "gsva_aucell_direct_output.tsv")

expr <- read.delim(expr_path, check.names = FALSE)
genes <- expr[[1]]
mat <- as.matrix(expr[, -1, drop = FALSE])
rownames(mat) <- genes
gene_sets <- list(
  event_module = paste0("G", sprintf("%03d", 1:12)),
  negative_control = paste0("G", sprintf("%03d", 80:95))
)

rows <- list()

if (requireNamespace("GSVA", quietly = TRUE)) {
  gsva_result <- tryCatch(
    {
      param <- GSVA::gsvaParam(mat, gene_sets, kcdf = "Gaussian")
      score <- GSVA::gsva(param, verbose = FALSE)
      data.frame(
        method = "GSVA",
        package = "GSVA",
        status = "executed",
        package_version = as.character(utils::packageVersion("GSVA")),
        native_task = "pathway_activity_scoring",
        n_features_scored = nrow(score),
        event_module_mean = mean(score["event_module", ], na.rm = TRUE),
        output = out_path,
        stringsAsFactors = FALSE
      )
    },
    error = function(e) {
      data.frame(
        method = "GSVA",
        package = "GSVA",
        status = paste0("run_error:", conditionMessage(e)),
        package_version = as.character(utils::packageVersion("GSVA")),
        native_task = "pathway_activity_scoring",
        n_features_scored = 0,
        event_module_mean = NA_real_,
        output = "error",
        stringsAsFactors = FALSE
      )
    }
  )
  rows[["GSVA"]] <- gsva_result
} else {
  rows[["GSVA"]] <- data.frame(
    method = "GSVA",
    package = "GSVA",
    status = "not_run_missing_R_package",
    package_version = NA_character_,
    native_task = "pathway_activity_scoring",
    n_features_scored = 0,
    event_module_mean = NA_real_,
    output = "missing",
    stringsAsFactors = FALSE
  )
}

if (requireNamespace("AUCell", quietly = TRUE)) {
  auc_result <- tryCatch(
    {
      rankings <- AUCell::AUCell_buildRankings(mat, plotStats = FALSE, verbose = FALSE)
      auc <- AUCell::AUCell_calcAUC(gene_sets, rankings, verbose = FALSE)
      score <- as.matrix(SummarizedExperiment::assay(auc))
      data.frame(
        method = "AUCell",
        package = "AUCell",
        status = "executed",
        package_version = as.character(utils::packageVersion("AUCell")),
        native_task = "sparse_gene_set_activity",
        n_features_scored = nrow(score),
        event_module_mean = mean(score["event_module", ], na.rm = TRUE),
        output = out_path,
        stringsAsFactors = FALSE
      )
    },
    error = function(e) {
      data.frame(
        method = "AUCell",
        package = "AUCell",
        status = paste0("run_error:", conditionMessage(e)),
        package_version = as.character(utils::packageVersion("AUCell")),
        native_task = "sparse_gene_set_activity",
        n_features_scored = 0,
        event_module_mean = NA_real_,
        output = "error",
        stringsAsFactors = FALSE
      )
    }
  )
  rows[["AUCell"]] <- auc_result
} else {
  rows[["AUCell"]] <- data.frame(
    method = "AUCell",
    package = "AUCell",
    status = "not_run_missing_R_package",
    package_version = NA_character_,
    native_task = "sparse_gene_set_activity",
    n_features_scored = 0,
    event_module_mean = NA_real_,
    output = "missing",
    stringsAsFactors = FALSE
  )
}

write.table(
  do.call(rbind, rows),
  out_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
