# Claim Ceiling

TED treats claim strength as an algorithmic output, not a writing afterthought.

Formal rule:

```text
ClaimCeiling(row) = max_L { L : min_{g in RequiredGates(L)} Evidence_g(row) = 1 }
```

Interpretation:

- Level 1: descriptive trend
- Level 2: event-FDR supported
- Level 2.5: annotation/scaffold or matrix-level candidate
- Level 3: replicate, block, or time robust
- Level 3.5: perturbation, multiome, or lineage-supported mechanism candidate
- Level 4: functional or rescue supported
- Level 5: independently replicated causal mechanism

The key benchmark behavior is conservative failure. Under unidentifiable synthetic conditions, TED should not preserve a high claim level merely because a score changes. It should downgrade the claim ceiling and mark the missing evidence gate.

For the current TED-Development package, Phase 4.5 tests this behavior directly through low signal, dropout, block imbalance, missing timepoints, batch-time confounding, and rare-lineage sweeps.
