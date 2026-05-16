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
meta_path <- get_arg("--metadata")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(outdir, "tradeseq_direct_output.tsv")

if (!requireNamespace("tradeSeq", quietly = TRUE)) {
  write.table(
    data.frame(
      method = "tradeSeq",
      package = "tradeSeq",
      status = "not_run_missing_R_package",
      package_version = NA_character_,
      native_task = "pseudotime_gene_dynamics",
      n_features_scored = 0,
      median_event_gene_stat = NA_real_,
      output = "missing",
      stringsAsFactors = FALSE
    ),
    out_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  quit(save = "no", status = 0)
}

expr <- read.delim(expr_path, check.names = FALSE)
meta <- read.delim(meta_path, check.names = FALSE)
genes <- expr[[1]]
mat <- as.matrix(expr[, -1, drop = FALSE])
rownames(mat) <- genes
counts <- round(pmax(mat, 0))
pseudotime <- matrix(meta$pseudotime, ncol = 1)
cell_weights <- matrix(1, nrow = nrow(meta), ncol = 1)
sce <- tradeSeq::fitGAM(
  counts = counts,
  pseudotime = pseudotime,
  cellWeights = cell_weights,
  nknots = 5,
  verbose = FALSE,
  parallel = FALSE
)
assoc <- tradeSeq::associationTest(sce)
assoc$gene <- rownames(assoc)
event_genes <- paste0("G", sprintf("%03d", 1:12))
event_stat <- median(assoc$waldStat[assoc$gene %in% event_genes], na.rm = TRUE)
write.table(
  data.frame(
    method = "tradeSeq",
    package = "tradeSeq",
    status = "executed",
    package_version = as.character(utils::packageVersion("tradeSeq")),
    native_task = "pseudotime_gene_dynamics",
    n_features_scored = nrow(assoc),
    median_event_gene_stat = event_stat,
    output = out_path,
    stringsAsFactors = FALSE
  ),
  out_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
