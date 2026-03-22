# encoder_pipeline History And Archive Guide

This directory used to have several top-level logs and experiment plans. They
were useful during bring-up, but they made the active operator harder to read.

The active docs are now:

- [README.md](/home/agi-demo/iron/operators/encoder_pipeline_archive/README.md)
- [current_status.md](/home/agi-demo/iron/operators/encoder_pipeline_archive/current_status.md)
- [optimization_plan.md](/home/agi-demo/iron/operators/encoder_pipeline_archive/optimization_plan.md)
- [readability_cleanup_plan.md](/home/agi-demo/iron/operators/encoder_pipeline_archive/readability_cleanup_plan.md)

Archived docs live in:

- `operators/encoder_pipeline_archive/docs/archive/`

Frozen compiler repros now live in:

- `operators/encoder_pipeline_archive/docs/repros/`

Archive contents:

- `debug_findings_2026-03-04.md`
  - detailed debugging history and resolved issues
- `resource_utilization_2026-03-04.md`
  - hardware limits and past resource pressure notes
- `design_optimization_plan.md`
  - older optimization roadmap before the current simplified plan
- `ffn_latency_optimization_log_2026-03-08.md`
  - older FFN/LN optimization notes

Frozen repro contents:

- `docs/repros/shim_drain_bd_repro/`
  - frozen shim-drain BD allocator repro artifacts

Use the archive only when you need historical reasoning or exact older
measurements. It is not intended to be the starting point for normal design
work.
