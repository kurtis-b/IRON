# Deferred Operator Work

These operator areas are intentionally deferred for now:

- `ffn`
  - Standalone `AIEFFN` still has a current-toolchain repeated-launch/runtime issue in the active down-projection path when `down_proj_depth > 1`.
- `encoder`
  - Depends on the standalone `ffn` path and should be revisited after `ffn` is fixed.
- `encoder_pipeline_memtile`
  - Current implementation is stale.
  - It should be realigned to follow `encoder_pipeline`, while keeping the `LN1 -> FFN` path on-chip instead of routing through DDR.
