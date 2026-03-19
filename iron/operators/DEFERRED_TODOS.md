# Deferred Operator Work

These operator areas are intentionally deferred for now:

- `encoder_pipeline_memtile`
  - Current implementation is stale.
  - It should be realigned to follow `encoder_pipeline`, while keeping the `LN1 -> FFN` path on-chip instead of routing through DDR.
