# Tutorial: ZSCAPE Perturbation-TED

ZSCAPE is the perturbation-aware developmental benchmark module. Its main purpose is to distinguish true fate loss from delay, state accumulation, fate redirection, and composition artifact.

## Expected Inputs

- Embryo or sample block identifier
- Genetic perturbation
- Real developmental time
- Cell type or lineage annotation
- Event family/module scores

## Recommended Model

```text
event_score ~ genotype + time + genotype:time + cell_type + embryo_block + QC
```

The embryo block is essential. Cell-level permutation is not sufficient for this dataset because the major replication unit is the individually resolved embryo.

## Required Outputs

- `zscape_perturbation_event_model.tsv`
- `zscape_celltype_abundance_event.tsv`
- `zscape_delay_vs_loss_classifier.tsv`
- `zscape_mutant_specific_driver_axis.tsv`
- `zscape_embryo_block_permutation_fdr.tsv`
- `zscape_claim_ceiling.tsv`

## Claim Ceiling

Without functional rescue, ZSCAPE can support a perturbation-aware developmental mechanism candidate, but not a functional causal claim.
