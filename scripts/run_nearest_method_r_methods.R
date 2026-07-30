suppressPackageStartupMessages({
  library(Matrix)
  library(monocle)
  library(tradeSeq)
})

# Monocle2 still uses the defunct igraph arguments `neimode` and `father`.
# Map them to their exact modern equivalents without changing the traversal.
extract_ddrtree_ordering_igraph_compat <- function(cds, root_cell, verbose = FALSE) {
  dp <- monocle::cellPairwiseDistances(cds)
  dp_mst <- monocle::minSpanningTree(cds)
  curr_state <- 1
  states <- rep(1, ncol(dp)); names(states) <- igraph::V(dp_mst)$name
  pseudotimes <- rep(0, ncol(dp)); names(pseudotimes) <- igraph::V(dp_mst)$name
  parents <- rep(NA_character_, ncol(dp)); names(parents) <- igraph::V(dp_mst)$name
  mst_traversal <- igraph::dfs(
    dp_mst, root = root_cell, mode = "all", unreachable = FALSE,
    parent = TRUE, order = TRUE
  )
  mst_traversal$father <- as.numeric(mst_traversal$parent)
  for (i in seq_along(mst_traversal$order)) {
    curr_node <- mst_traversal$order[i]
    curr_node_name <- igraph::V(dp_mst)[curr_node]$name
    if (!is.na(mst_traversal$father[curr_node])) {
      parent_node <- mst_traversal$father[curr_node]
      parent_node_name <- igraph::V(dp_mst)[parent_node]$name
      curr_node_pseudotime <- pseudotimes[parent_node_name] + dp[curr_node_name, parent_node_name]
      if (igraph::degree(dp_mst, v = parent_node_name) > 2) curr_state <- curr_state + 1
    } else {
      parent_node_name <- NA_character_
      curr_node_pseudotime <- 0
    }
    pseudotimes[curr_node_name] <- curr_node_pseudotime
    states[curr_node_name] <- curr_state
    parents[curr_node_name] <- parent_node_name
  }
  ordering_df <- data.frame(
    sample_name = names(states), cell_state = factor(states),
    pseudo_time = as.vector(pseudotimes), parent = parents
  )
  row.names(ordering_df) <- ordering_df$sample_name
  ordering_df
}
assignInNamespace(
  "extract_ddrtree_ordering", extract_ddrtree_ordering_igraph_compat,
  ns = "monocle"
)

# project2MST uses the other igraph 2.x defunct alias, nei(); .nei() is the
# documented drop-in replacement inside vertex-sequence indexing.
project2mst <- getFromNamespace("project2MST", "monocle")
project2mst_text <- paste(deparse(project2mst), collapse = "\n")
project2mst_text <- gsub("nei(", ".nei(", project2mst_text, fixed = TRUE)
project2mst_compat <- eval(parse(text = project2mst_text), envir = asNamespace("monocle"))
environment(project2mst_compat) <- asNamespace("monocle")
assignInNamespace("project2MST", project2mst_compat, ns = "monocle")

one_line <- function(x) gsub("[[:space:]]+", " ", as.character(x))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript run_nearest_method_r_methods.R INPUT_DIR")
input_dir <- normalizePath(args[[1]], mustWork = TRUE)

counts <- Matrix::readMM(file.path(input_dir, "raw_counts.mtx"))
counts <- as(counts, "dgCMatrix")
cells <- read.delim(file.path(input_dir, "cell_metadata.tsv"), check.names = FALSE)
membership <- read.delim(file.path(input_dir, "pathway_membership.tsv"), check.names = FALSE)
coord <- as.numeric(cells$ordered_coordinate)
gene_names <- sprintf("gene_%05d", seq_len(ncol(counts)))
cell_names <- as.character(cells$cell_id)
rownames(counts) <- cell_names
colnames(counts) <- gene_names

pathways <- split(
  gene_names[as.integer(membership$gene_index) + 1L],
  as.character(membership$pathway)
)

tips_one <- function(cds, genes, coordinate) {
  tryCatch({
    path_cds <- setOrderingFilter(cds, intersect(genes, rownames(cds)))
    path_cds <- reduceDimension(
      path_cds,
      reduction_method = "DDRTree",
      norm_method = "none",
      max_components = 2,
      verbose = FALSE
    )
    path_cds <- orderCells(path_cds)
    inferred <- as.numeric(pData(path_cds)$Pseudotime)
    rho <- suppressWarnings(cor(coordinate, inferred, method = "pearson", use = "complete.obs"))
    pval <- suppressWarnings(cor.test(coordinate, inferred, method = "pearson")$p.value)
    data.frame(
      native_score = abs(rho), signed_correlation = rho, p_value = pval,
      status = "ok", error = NA_character_, stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      native_score = NA_real_, signed_correlation = NA_real_, p_value = NA_real_,
      status = "error", error = one_line(conditionMessage(e)), stringsAsFactors = FALSE
    )
  })
}

run_tips <- function() {
  expression <- t(counts)
  phenotype <- new("AnnotatedDataFrame", data = data.frame(
    ordered_coordinate = coord,
    row.names = cell_names,
    stringsAsFactors = FALSE
  ))
  features <- new("AnnotatedDataFrame", data = data.frame(
    gene_short_name = gene_names,
    row.names = gene_names,
    stringsAsFactors = FALSE
  ))
  cds <- newCellDataSet(
    expression,
    phenoData = phenotype,
    featureData = features,
    lowerDetectionLimit = 0.1,
    expressionFamily = negbinomial.size()
  )
  cds <- estimateSizeFactors(cds)
  dispersion_status <- "ok"
  cds <- tryCatch(
    estimateDispersions(cds),
    error = function(e) {
      dispersion_status <<- paste0("skipped_incompatible_monocle2_dplyr: ", one_line(conditionMessage(e)))
      cds
    }
  )
  cds <- detectGenes(cds, min_expr = 0.1)
  rows <- lapply(names(pathways), function(pathway) {
    ans <- tips_one(cds, pathways[[pathway]], coord)
    ans$pathway <- pathway
    ans
  })
  out <- do.call(rbind, rows)
  out$q_value <- p.adjust(out$p_value, method = "BH")
  out$dispersion_status <- dispersion_status
  out <- out[, c("pathway", "native_score", "signed_correlation", "p_value", "q_value", "status", "error", "dispersion_status")]
  write.table(out, file.path(input_dir, "tips_native.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
}

run_tradeseq <- function() {
  y <- t(counts)
  pseudotime <- matrix(coord, ncol = 1)
  cell_weights <- matrix(1, nrow = length(coord), ncol = 1)
  out <- tryCatch({
    sce <- fitGAM(
      counts = y,
      pseudotime = pseudotime,
      cellWeights = cell_weights,
      nknots = 6,
      verbose = FALSE,
      parallel = FALSE
    )
    assoc <- associationTest(sce, global = TRUE)
    data.frame(
      gene = gene_names,
      wald_stat = as.numeric(assoc$waldStat),
      p_value = as.numeric(assoc$pvalue),
      mean_log_fc = as.numeric(assoc$meanLogFC),
      status = ifelse(is.finite(assoc$pvalue), "ok", "not_estimable"),
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      gene = gene_names,
      wald_stat = NA_real_, p_value = NA_real_, mean_log_fc = NA_real_,
      status = "error", error = one_line(conditionMessage(e)), stringsAsFactors = FALSE
    )
  })
  write.table(out, file.path(input_dir, "tradeseq_native_gene.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
}

started <- proc.time()[["elapsed"]]
run_tips()
tips_elapsed <- proc.time()[["elapsed"]] - started
started <- proc.time()[["elapsed"]]
run_tradeseq()
trade_elapsed <- proc.time()[["elapsed"]] - started
write.table(
  data.frame(method = c("TIPS", "tradeSeq"), elapsed_seconds = c(tips_elapsed, trade_elapsed)),
  file.path(input_dir, "r_method_runtime.tsv"), sep = "\t", quote = FALSE, row.names = FALSE
)
