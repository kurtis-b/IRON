# Deferred Operator Work

Deferred follow-up:

- `encoder_pipeline_memtile`
  - Non-sequence-parallel topologies are implemented and validated.
  - Sequence-parallel `2ps`, `2ps_2ph`, `2ps_2pffn`, and `2ps_2ph_2pffn` are
    implemented and validated.
  - Heavier mixed-head sequence-parallel topologies still need follow-up.
  - Multi-branch seq-par beyond `2ps_2ph_2pffn` still needs follow-up.
  - `4ps` remains unsupported for memtile mode.
  - A pure on-chip `4ps` path would need LN1 output to drive FFN-up, replay, and
    LN2 residual at once, and that exceeds the current channel budget without an
    extra worker tile or a DDR fallback.
