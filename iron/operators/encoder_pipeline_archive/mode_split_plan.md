# encoder_pipeline Shared Directory Retirement Plan

Historical plan. The split is complete and `operators/encoder_pipeline/` has
already been removed. Remaining references to that directory below describe the
old state during the transition.

## Goal

Remove `operators/encoder_pipeline` as an implementation dependency so the two
mode directories are fully independent:

- `operators/encoder_pipeline_ddr`
- `operators/encoder_pipeline_memtile`

Duplicated code is acceptable. The primary goal is to minimize the amount of
text that must be read when working on one mode.

## Current blockers

The mode directories are not isolated yet. They still depend on the shared
package for core implementation pieces:

- `operators/encoder_pipeline_ddr/op.py`
  - subclasses `operators.encoder_pipeline.op.AIEEncoderPipeline`
- `operators/encoder_pipeline_memtile/op.py`
  - subclasses `operators.encoder_pipeline.op.AIEEncoderPipeline`
- `operators/encoder_pipeline_ddr/cases.py`
  - imports `operators.encoder_pipeline.reference.generate_golden_reference`
- `operators/encoder_pipeline_memtile/cases.py`
  - imports `operators.encoder_pipeline.reference.generate_golden_reference`
- `operators/encoder_pipeline_ddr/hooks.py`
  - imports `operators.encoder_pipeline.design.fused_mha`
- `operators/encoder_pipeline_memtile/hooks.py`
  - imports `operators.encoder_pipeline.design.fused_mha`
- both local `design.py` files still import shared helpers:
  - `operators.encoder_pipeline.debug_modes`
  - `operators.encoder_pipeline.mapping_validation`

That means the shared directory is still required for:
- operator construction
- reference generation
- debug-mode handling
- mapping validation
- mode entry hooks

## Blocker-by-blocker solution

### Blocker 1

- current:
  - `encoder_pipeline_ddr/op.py`
  - `encoder_pipeline_memtile/op.py`
  - both subclass `operators.encoder_pipeline.op.AIEEncoderPipeline`

- solution:
  - copy the shared operator implementation into each mode directory
  - make each local `op.py` own the full operator class directly
  - hardcode:
    - local `design.py`
    - local staging mode
    - local design import path

- result:
  - no mode-local operator depends on `operators/encoder_pipeline/op.py`

### Blocker 2

- current:
  - `encoder_pipeline_ddr/cases.py`
  - `encoder_pipeline_memtile/cases.py`
  - both import shared `reference.generate_golden_reference`

- solution:
  - copy `reference.py` into each mode directory
  - update each local `cases.py` to import only its local `reference.py`
  - keep `run_test` shared under `operators/common`, since that is not part of
    the encoder-pipeline split problem

- result:
  - mode-local tests no longer depend on `operators/encoder_pipeline/reference.py`

### Blocker 3

- current:
  - `encoder_pipeline_ddr/hooks.py`
  - `encoder_pipeline_memtile/hooks.py`
  - both import shared `design.fused_mha`

- solution:
  - delete `hooks.py` entirely
  - merge the hook contents directly into each local `design.py`
  - rename the local top-level entrypoint consistently inside each mode-local
    design file

- result:
  - no mode-local file imports `operators/encoder_pipeline/design.py`

### Blocker 4

- current:
  - both local `design.py` files import:
    - `operators.encoder_pipeline.debug_modes`
    - `operators.encoder_pipeline.mapping_validation`

- solution:
  - copy these shared helper files into each mode directory
  - change imports to use local modules only
  - duplicate freely; do not create another shared helper package

- result:
  - each local `design.py` becomes self-contained for design helpers

### Blocker 5

- current:
  - the shared directory still owns the mixed-mode entrypoints:
    - `operators/encoder_pipeline/op.py`
    - `operators/encoder_pipeline/test.py`
    - `operators/encoder_pipeline/design.py`

- solution:
  - once the first four blockers are removed, delete these files instead of
    trying to preserve them
  - keep only one of these two end states:
    - delete `operators/encoder_pipeline/` entirely
    - or reduce it to a tiny compatibility package with a short README

- result:
  - no ambiguity about which directory owns real implementation

### Blocker 6

- current:
  - `operators/__init__.py` still exports the shared `AIEEncoderPipeline`

- solution:
  - remove the shared export, or alias it explicitly to one chosen mode
  - update any callers that still import the shared symbol

- result:
  - import surface matches the real split architecture

## Target state

Each mode directory should contain everything needed to iterate on that mode:

- `design.py`
- `op.py`
- `reference.py`
- `cases.py`
- `debug_modes.py`
- `mapping_validation.py`
- `README.md`
- optional small local helpers only if they reduce reading cost

After cutover:
- `encoder_pipeline_ddr` reads only `encoder_pipeline_ddr/*`
- `encoder_pipeline_memtile` reads only `encoder_pipeline_memtile/*`
- `operators/encoder_pipeline` is removed or reduced to a compatibility stub
- `operators/__init__.py` exports only the split operators, or aliases the old
  shared name intentionally

## Constraints

- Prefer copying over introducing new wrapper layers.
- Prefer deleting the shared directory over preserving backward compatibility.
- Keep implementation steps mechanical and low-risk.
- Validate after each pass with one DDR smoke case and one memtile smoke case.

## Phase plan

### Phase 1: eliminate shared runtime/operator dependencies

1. Copy shared `op.py` into each mode directory.
2. In each local `op.py`:
   - remove subclassing of `operators.encoder_pipeline.op.AIEEncoderPipeline`
   - make the local operator class inherit directly from `AIEOperator`
   - hardcode the local `design.py` import path
   - hardcode the local staging mode
3. Copy shared `reference.py` into each mode directory.
4. Update local `cases.py` files to import only local modules.

Result:
- mode-local operator construction and test generation no longer depend on the
  shared package.

### Phase 2: eliminate shared design helper dependencies

1. Copy these shared helper files into each mode directory:
   - `debug_modes.py`
   - `mapping_validation.py`
2. Update local `design.py` imports to use only local helpers.
3. Remove any shared-package imports from local `hooks.py`, or inline the hook
   behavior directly into each local `design.py`.

Result:
- each local `design.py` becomes self-contained for its mode.
- no local mode file imports from `operators.encoder_pipeline.*`.

### Phase 3: retire the shared package

1. Remove `operators/encoder_pipeline/test.py`.
2. Remove `operators/encoder_pipeline/op.py`.
3. Remove `operators/encoder_pipeline/design.py`.
4. Remove shared helper files from `operators/encoder_pipeline`.
5. Decide one of two end states:
   - preferred: delete `operators/encoder_pipeline/` entirely
   - fallback: keep only a tiny `__init__.py` / README compatibility stub

Result:
- there is no implementation ambiguity.
- users choose DDR or memtile explicitly.

### Phase 4: clean import surface

1. Update `operators/__init__.py`.
2. Remove `AIEEncoderPipeline` shared export, or alias it explicitly to one
   mode with a comment.
3. Update any application/test imports still using the shared path.

Result:
- the public import surface matches the real architecture split.

## Validation plan

After each phase, run:

### Memtile smoke

```bash
rm -rf ./build
source /opt/xilinx/xrt/setup.sh
source /home/agi-demo/iron/ironenv/bin/activate
pytest -q operators/encoder_pipeline_memtile/cases.py -k 'encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg' -s -x
```

### DDR smoke

```bash
rm -rf ./build
source /opt/xilinx/xrt/setup.sh
source /home/agi-demo/iron/ironenv/bin/activate
pytest -q operators/encoder_pipeline_ddr/cases.py -k 'encoder_512seq_64hdim_12heads_3072ffn_32qseqtile_64kvtile_96embtile_4pheads_4pffn_8pacc_4opg' -s -x
```

## Recommended implementation order

1. Phase 1
2. Validate
3. Phase 2
4. Validate
5. Phase 3 and Phase 4 together
6. Run both mode-local suites again

## What not to do

- Do not keep a thin shared `op.py` base class.
- Do not keep shared hook files.
- Do not introduce another common helper package just to avoid duplication.
- Do not try to preserve the current mixed-mode directory as the main entry
  point.

## Expected outcome

This will increase total repository duplication, but it will reduce per-mode
reading cost sharply:
- working on DDR will require reading only `encoder_pipeline_ddr/*`
- working on memtile will require reading only `encoder_pipeline_memtile/*`

That is the right tradeoff for focused experimentation.

## Current status

Implemented:
- Phase 1
- Phase 2, with local `hooks.py` retained as mode-local files

Resolved blockers:
- mode-local `op.py` no longer subclasses `operators.encoder_pipeline.op`
- mode-local `cases.py` no longer imports shared `reference.py`
- mode-local `reference.py`, `debug_modes.py`, and
  `mapping_validation.py` now exist in each mode directory
- mode-local `design.py` imports only local helper modules
- local `hooks.py` no longer imports from the shared package

Still left before deleting `operators/encoder_pipeline/`:
- retire the shared mixed-mode entrypoints:
  - `design.py`
  - `op.py`
  - `test.py`
  - `profile_debug_modes.py`
  - shared helper files
- update `operators/__init__.py` and any callers that still import the shared
  `AIEEncoderPipeline`
- either delete `operators/encoder_pipeline/` entirely or reduce it to a tiny
  compatibility stub temporarily

## Remaining blockers and solutions

### Blocker A: shared export in `operators/__init__.py`

- current:
  - `operators/__init__.py` still exports `AIEEncoderPipeline`

- solution:
  - remove the shared export entirely
  - keep only:
    - `AIEEncoderPipelineDDR`
    - `AIEEncoderPipelineMemtile`

- reason:
  - keeping `AIEEncoderPipeline` preserves the illusion that the shared package
    is still the canonical implementation

### Blocker B: external callers still use the shared operator

- current:
  - `applications/bert/src/block/transformer.py` imports `AIEEncoderPipeline`

- solution:
  - update each caller to import an explicit mode:
    - `AIEEncoderPipelineDDR` if it wants the old default behavior
    - `AIEEncoderPipelineMemtile` if it is intentionally using memtile mode
  - for BERT specifically, use `AIEEncoderPipelineDDR` as the compatibility
    replacement because the shared operator defaulted to DDR staging

- reason:
  - the caller must own the mode choice once the shared package is removed

### Blocker C: shared mixed-mode implementation files still exist

- current:
  - `operators/encoder_pipeline/op.py`
  - `operators/encoder_pipeline/design.py`
  - `operators/encoder_pipeline/reference.py`
  - `operators/encoder_pipeline/debug_modes.py`
  - `operators/encoder_pipeline/mapping_validation.py`

- solution:
  - delete them after `Blocker A` and `Blocker B` are resolved
  - do not keep compatibility wrappers here

- reason:
  - a compatibility wrapper would keep the shared package alive as an
    implementation dependency and defeat the split

### Blocker D: shared mixed-mode tests still exist

- current:
  - `operators/encoder_pipeline/test.py`
  - `operators/encoder_pipeline/profile_debug_modes.py`

- solution:
  - retire the shared test/profile entrypoints
  - keep only:
    - `operators/encoder_pipeline_ddr/cases.py`
    - `operators/encoder_pipeline_memtile/cases.py`
  - if stage-profile functionality is still needed, duplicate it into
    mode-local scripts under each mode directory

- reason:
  - the shared mixed-mode test file forces the shared package to remain real

### Blocker E: docs and repros still live under the shared directory

- current:
  - shared docs live under `operators/encoder_pipeline/`
  - repro READMEs still reference the old shared import paths

- solution:
  - move historical docs/repros out of the operator package path entirely
  - preferred target:
    - `operators/encoder_pipeline_archive/`
  - update repro READMEs to use explicit mode-local imports if they are kept

- reason:
  - docs do not need to block deletion of the shared implementation directory
  - an archive directory is a better home for historical material than a live
    operator package

### Blocker F: local `hooks.py` still exists

- current:
  - `encoder_pipeline_ddr/hooks.py`
  - `encoder_pipeline_memtile/hooks.py`

- solution:
  - optional for full deletion of `operators/encoder_pipeline/`
  - not a blocker anymore, because they are mode-local and no longer import the
    shared package
  - inline later only if reducing per-mode reading cost matters more than
    keeping the mode-local policy surface factored

- reason:
  - they no longer prevent deletion of the shared directory

## Final deletion sequence

1. Update `applications/bert/src/block/transformer.py` to import an explicit
   mode operator.
2. Update `operators/__init__.py` to drop `AIEEncoderPipeline`.
3. Move shared repro/docs material to `operators/encoder_pipeline_archive/`.
4. Delete:
   - `operators/encoder_pipeline/test.py`
   - `operators/encoder_pipeline/profile_debug_modes.py`
5. Delete the shared implementation files:
   - `design.py`
   - `op.py`
   - `reference.py`
   - `debug_modes.py`
   - `mapping_validation.py`
6. Delete `operators/encoder_pipeline/` entirely, or leave only a stub README
   if a temporary transition marker is needed.
