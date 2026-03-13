# encoder_pipeline Readability Cleanup Plan

## Goal

Make `operators/encoder_pipeline` easier to read before doing more
optimization work.

The priority is:

1. remove lines
2. remove files
3. collapse duplicated logic
4. only then split code into new files if that clearly reduces total surface
   area

This is a deletion-first cleanup plan, not a wrapper-creation plan.

## Rules

1. No behavior change per cleanup pass.
2. Prefer deleting obsolete code over moving it.
3. Prefer collapsing duplicated branches over adding another abstraction layer.
4. Do not create a new file unless it removes more complexity than it adds.
5. Keep the regression set fixed across all cleanup passes.

## Current readability problems

### 1. Too many live docs

The top-level directory had too many overlapping "current" docs. Pass 1 already
reduced that, but the remaining code still carries a lot of historical context
inline.

### 2. `design.py` carries too much old experiment residue

[design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py) is large
not just because the operator is complex, but because it still contains:

- old tuning knobs
- topology-specific workaround branches
- long explanatory comments from already-resolved experiments
- duplicated policy checks
- one-off runtime and placement exceptions

The first cleanup target is to delete or collapse those, not to split the file
immediately.

### 3. Too many optional knobs

The operator now has many env overrides and tuning switches. Some are still
valuable, but many are only useful for experiments that are already settled.

Those should be removed or demoted before introducing more structure.

### 4. Mode-specific logic still leaks into shared code

This is real, but the first fix should be:

- remove dead mode branches
- collapse duplicated memtile/ddr decisions

not immediately create more hook layers.

## Fixed regression set

Every cleanup pass must keep these passing:

- `lnstage_memtile-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg`
- `lnstage_ddr-encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_128embtile_4pheads_6pffn_6pacc_2opg`
- `lnstage_memtile-encoder_64seq_64hdim_12heads_3072ffn_64qseqtile_64kvtile_48embtile_6pheads_1pffn_16pacc_2opg`

## Cleanup order

### Pass 1. Documentation triage

Status:

- done

Result:

- live docs reduced to:
  - [README.md](/home/agi-demo/iron/operators/encoder_pipeline/README.md)
  - [current_status.md](/home/agi-demo/iron/operators/encoder_pipeline/current_status.md)
  - [optimization_plan.md](/home/agi-demo/iron/operators/encoder_pipeline/optimization_plan.md)
  - [readability_cleanup_plan.md](/home/agi-demo/iron/operators/encoder_pipeline/readability_cleanup_plan.md)
- older docs moved under:
  - [docs/history.md](/home/agi-demo/iron/operators/encoder_pipeline/docs/history.md)
  - `docs/archive/`

### Pass 2. Delete dead documentation and historical clutter

Target:

- reduce top-level file count further
- remove stale documents that do not need to stay near the operator code

Work:

1. Decide whether [readability_cleanup_plan.md](/home/agi-demo/iron/operators/encoder_pipeline/readability_cleanup_plan.md)
   remains live or is temporary.
2. Move frozen repro material out of the top-level reading path:
   - `row_store_repro/`
   - `shim_drain_bd_repro/`
3. Remove top-level result artifacts if they are not required as checked-in
   reference material:
   - `encoder_pipeline_results.csv`

Expected outcome:

- top-level directory becomes mostly code plus 3 active docs

### Pass 3. Delete dead code paths and stale knobs

Target:

- shrink `design.py` without splitting it yet

Work:

1. Remove env overrides that are only useful for closed experiments.
2. Remove row-store tuning knobs that have already been rejected as defaults
   and no longer guide live work.
3. Remove stale comments that explain paths no longer present.
4. Remove old fallback branches that are no longer reachable because:
   - uneven FFN partitioning now errors out
   - certain row-store scopes are now fixed
   - some topology families are intentionally unsupported

Expected outcome:

- `design.py` loses lines before it gains helper files

### Pass 4. Collapse duplicated policy logic

Target:

- reduce repeated decision logic across shared and mode-specific code

Work:

1. Identify repeated memtile/ddr policy checks and replace them with one shared
   decision point.
2. Collapse duplicated row-store eligibility conditions.
3. Collapse duplicated runtime wait/fill policy checks.

Important:

- this pass should still avoid creating new modules unless the duplicate logic
  cannot be removed in place.

Expected outcome:

- fewer condition clusters
- smaller mode files
- smaller shared file

### Pass 5. Simplify tests by deleting matrix noise

Target:

- keep the useful comparisons, remove dead or redundant test generation logic

Work:

1. Remove test cases that are now structurally unsupported.
2. Collapse repetitive case helpers where the output names can be derived more
   simply.
3. Keep one path for regular tests and one path for stage-profile tests.

Expected outcome:

- [test.py](/home/agi-demo/iron/operators/encoder_pipeline/test.py) gets
  shorter without adding a test-framework layer

### Pass 6. Only then consider extraction

Target:

- split only the pieces that remain large after deletion

Allowed extractions:

1. `core_fn_*` bodies into one `core_fns.py` file if that removes a large block
   from [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
   without creating a web of wrappers
2. env/config parsing into one `config.py` file if the top of
   [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
   remains noisy after stale knob deletion

Not allowed:

- splitting into many small files just to redistribute the same complexity
- adding more hook layers or indirection unless they remove duplicated code

## What should be removed first

These are the highest-value deletion targets:

1. obsolete experiment comments in [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
2. stale env overrides in [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
3. top-level frozen repro directories from the main reading path
4. result/log files that are not active documentation
5. duplicated policy branches between:
   - [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)
   - [design_ln1_ddr.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_ddr.py)
   - [design_ln1_memtile.py](/home/agi-demo/iron/operators/encoder_pipeline/design_ln1_memtile.py)

## Success criteria

Cleanup is successful when:

1. the top-level directory mostly contains code plus a small doc set
2. [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py) is
   materially shorter because dead logic is gone
3. a reader can find:
   - current status
   - optimization priorities
   - run commands
   without opening historical notes
4. no extra helper files were created unless they clearly reduced net
   complexity

## Immediate next pass

The next cleanup pass should be:

- delete or relocate frozen repro material and result artifacts
- then do a dead-knob / dead-comment sweep in
  [design.py](/home/agi-demo/iron/operators/encoder_pipeline/design.py)

That is the shortest path to a smaller, more readable directory.
