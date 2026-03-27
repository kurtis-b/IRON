# Design Pattern Considerations

This document records the engineering constraints and tradeoffs that matter
across the three `transformer_layer` design patterns. It is meant to sit next
to the workflow and methodology docs, not replace them.

## Shared Invariants

All compared executions are intentionally aligned on the same workload
boundary:

- one of two explicit input boundaries:
  - supplied post-projection `Q`, `K`, `V`, and residual `R`
  - `hidden_states` with derived residual `R`
- one encoder-style downstream workload
- batch size `1`
- `bfloat16`
- no attention mask
- synthetic-first inputs and weights

The fairness rule for this branch is:

- within one study axis, compare the same effective layer work even when the
  mapping across host and NPU is very different
- across study axes, record the input boundary explicitly rather than implying
  that post-projection and hidden-state runs are interchangeable

The checked-in study surface now includes:

- a long-sequence sweep through `seq_len=16384` on the retained
  `768/3072/12` and `1024/4096/16` families
- an embedding-scale sweep through `dense_8b_class`, with explicit
  unsupported rows instead of silent pruning
- a projection-inclusive follow-on axis where the same studies start from
  `hidden_states` instead of post-projection `Q/K/V/R`

Useful future extensions beyond that are:

- extend the embedding-scale sweep beyond `dense_8b_class` if the study needs
  a stronger scaling break point

## `encoder_pipeline` Considerations

`encoder_pipeline` is the most topology-aware design pattern in the study. Its
strength is that it is the clearest long-sequence-friendly path, but that comes
with higher placement and topology complexity.

Important considerations:

- support is constrained by topology, resource placement, and operator-specific
  parallelization choices
- bringup risk is often compile-time or placement-time rather than simple
  runtime correctness
- long-sequence capability is part of the value proposition of this pattern,
  not just an implementation detail
- the retained runtime surface now reaches `seq_len=16384` on both supported
  families, but the larger `1024/4096/16` family needed different topology
  defaults than the `768/3072/12` family, especially `emb_tile=128`

In the study, `encoder_pipeline` is the reference point for what a more
integrated NPU-native mapping can achieve when the topology is valid.

In the checked-in projection-inclusive follow-on axis, `encoder_pipeline` uses
host-side Q/K/V projection. That is not a flaw in the experiment; it is part
of what the comparison is intended to show about pipeline difficulty on a
resource-constrained device.

If the study adds a larger-embedding comparison later, `encoder_pipeline`
should also be treated as the hardest pattern to scale cleanly. That kind of
comparison is interesting precisely because it would show how quickly topology
and resource pressure become first-order constraints for a tightly integrated
pipeline design.

That is now reflected directly in the follow-on embedding-scale study design:
the study should record unsupported `encoder_pipeline` points as first-class
results rather than broadening the topology surface just to avoid empty cells.

## `gemm_only` Considerations

`gemm_only` is intentionally asymmetric:

- GEMMs are offloaded to the NPU
- non-GEMM work stays on the host

That makes it easier to reason about and easier to bring up than a more
integrated pattern, but it also means:

- host/device orchestration is part of steady-state cost
- stage boundaries are more explicit
- it is not expected to look like a fully NPU-native execution path

Its comparison value comes from keeping the operator set intentionally narrow
while showing what happens when the offload policy is GEMM-centric rather than
fully graph-centric.

The retained runtime surface now also reaches `seq_len=16384` on both
supported families. It gets there differently from the other two patterns:

- the long attention-score GEMM is partitioned in `N`
- the sequence is query-blocked so the host only materializes one query block
  of scores / probabilities / PV context at a time
- that keeps the design simple enough to extend, but it also makes the
  host-device orchestration cost more visible at long sequence lengths

In the checked-in projection-inclusive follow-on axis, `gemm_only` absorbs
Q/K/V projection directly on the NPU because that work is itself GEMM-heavy.

`gemm_only` is also a natural candidate for a larger-embedding comparison,
because increasing hidden size mostly expands GEMM surfaces that are already
the core of the design. That does not make scaling free, but it is a more
direct fit than asking `encoder_pipeline` to grow the same way.

That is why the follow-on embedding-scale study extends `gemm_only` through
`dense_8b_class` and treats it as one of the likely best-NPU candidates for
the final iGPU comparison.

## `operator_runlist` Considerations

`operator_runlist` uses a different tradeoff surface:

- it is built from separate NPU operators
- those operators are stitched into one `AIEEncoderRunlist` surface
- it aims to offload the downstream transformer graph without requiring the
  same integrated topology as `encoder_pipeline`
- the retained tested surface today is the non-pipelined path

Important current considerations:

- validation should continue to treat individual operators as first-class
  components, because stage-level failures are easier to isolate than full-layer
  failures
- long-sequence behavior is currently limited by large attention intermediates
  such as `attn_scores_output`, `attn_scaled_output`, and
  `attn_weights_output`
- the current runtime now query-blocks the long-sequence path so those
  intermediates are sized to one query block at a time instead of the full
  sequence
- the current runtime already reuses those large buffers within the query
  block by aliasing `attn_scaled_output` and `attn_weights_output` onto
  `attn_scores_output`
- the current runtime already has exact-size BO pooling and explicit aliasing,
  so any further BO work should keep using lifetime-based reuse of the large
  attention buffers rather than general suballocation from one large slab
- the current correctness-side mitigation is stage-level parity, where only the
  inputs and output of the operator under test are materialized instead of the
  full downstream layer
- the dedicated `operator_runlist` validator should remain the place where
  component-boundary parity and repeated completion checks are combined
- the retained short-surface full-layer parity issue has been fixed; the
  remaining long-sequence preference for component-boundary validation is now a
  deliberate methodology choice rather than a short-surface correctness bug

At the moment, full end-to-end `8192` and `16384` parity are still not the
preferred validation path for `operator_runlist`, because both the runtime path
and the host reference still become expensive if the entire layer is
materialized at once. The currently verified `operator_runlist` runtime surface
extends through `seq_len=16384` on the retained `768/3072/12` and
`1024/4096/16` families via query-blocked execution, and the long-sequence
correctness method is the dedicated component-boundary study rather than full
host materialization.

In the checked-in projection-inclusive follow-on axis, `operator_runlist`
absorbs Q/K/V projection as additional staged GEMM operators rather than as a
host-side preprocessing step.

`operator_runlist` is also a reasonable candidate for a larger-embedding
comparison. It still has real memory and runtime-pressure issues, but those are
easier to reason about stage by stage than the topology-wide constraints that
show up in `encoder_pipeline`.

## Cross-Pattern Implications For The Study

These considerations matter in different ways:

- fairness: all patterns must keep the same `Q/K/V/R` boundary and downstream
  workload definition
- feasibility: not every pattern has the same long-sequence runtime or memory
  story
- interpretation: results should distinguish between algorithmic capability and
  engineering cost

At the current retained support boundary, all three NPU patterns execute
through `seq_len=16384` on the `768/3072/12` and `1024/4096/16` families, but
they do so by very different mechanisms:

- `encoder_pipeline` depends on family-specific topology choices
- `gemm_only` depends on partitioned attention-score GEMMs and query blocking
- `operator_runlist` depends on query-blocked execution and staged validation

The follow-on embedding-scale study answers a different question: how far each
design can scale hidden size before the practical best pattern changes. That is
why the study needs explicit unsupported rows and a best-NPU-vs-iGPU compare
rather than assuming all three NPU patterns remain feasible at every point.

The projection-inclusive axis intentionally relaxes the post-projection
`Q/K/V/R` boundary and answers a different question: how much harder is it to
carry projection work inside each design pattern, especially for
`encoder_pipeline` on a resource-constrained NPU?

A larger-embedding comparison would answer a different question too: which
design patterns keep scaling pressure mostly local to individual operators, and
which ones become difficult to support because the integrated topology itself
gets harder to place and compile?

This means the methodology should not imply that every pattern must have the
same internal execution structure. Instead, the study should make clear which
constraints are part of the pattern itself and which are temporary engineering
limitations of the current branch.

## Optional Next Extensions

The next engineering steps implied by these considerations are:

1. measure whether more BO reuse / aliasing is needed beyond the current
   query-blocked score/scale/softmax reuse
2. if desired, extend the embedding-scale study beyond `dense_8b_class`
3. if desired, add a thesis-facing support-matrix figure that makes the
   `encoder_pipeline` feasibility boundary explicit next to the performance
   plots
4. if desired, extend the sensitivity or final thesis sweeps beyond the
   currently retained `64..2048` single-family surface now that the follow-on
   long-sequence study runs through `16384`

Those steps belong to implementation work, not to the benchmark methodology
itself.
