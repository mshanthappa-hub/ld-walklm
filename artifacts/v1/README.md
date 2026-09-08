# Migrated v1 provenance artifacts

These small artifacts were copied verbatim from the superseded experiment
packaging repository so `ld-walklm` no longer depends on that repository.

- `walklm_embedding.sha256`: checksum sidecar for the 8.5 GB frozen
  EmbeddingGemma walk-text tensor.
- `seed0_stage3_cs.txt`: preserved seed-0 stage-3 and v1 C&S output.

The large v1 prediction tensors and remaining logs are mirrored in a release
attached to this repository. They are exploratory, test-aware evidence rather
than the primary v2 submission result.

The stage-3 v2 tensors and complete v2 result evidence are in the
`ogbn-products-walklm-v2-10seed-20260904` release and `results/v2/`.
