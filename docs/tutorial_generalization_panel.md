# Tutorial: TED Generalization Panel

This tutorial describes how the generalization panel should be reproduced from processed outputs.

## Inputs

- GSE199308 stratified streamed TED-lite outputs.
- GSE123013 plant root fate-switch and stress-adjusted outputs.
- GSE127202 human endoderm CRISPRi adapter outputs.
- GSE157977 in vivo Perturb-seq guide-level adapter outputs.
- GSE292039 public functional-style alignment gate outputs.

## Main outputs

- `gse199308_ted_lite_main_figure.png`
- `gse123013_claim_discipline_panel.png`
- `ted_adapter_generalization_compact_panel.png`
- `final_claim_matrix.tsv`

## Claim boundaries

- GSE199308: embryo-block/pseudobulk event grammar only; no cell-type-specific fate loss.
- GSE123013: stress-sensitive fate-switch candidate only; no strong plant root mechanism.
- GSE127202: regulatory-gate adapter demonstration only; no causal CRISPR mechanism.
- GSE157977: guide-barcode adapter candidate only; no target-gene-specific neural fate claim.
- GSE292039: gene mapping complete, but public functional-style alignment not supported.
