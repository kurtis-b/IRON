#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fnmatch import fnmatch
import json
import logging
from pathlib import Path
import re
import tempfile
from typing import Iterable, Literal

import torch

from iron.common import AIEContext
from iron.operators.addnorm.op import AIEAddAndNorm
from iron.operators.elementwise_add.op import AIEElementwiseAdd
from iron.operators.elementwise_mul.op import AIEElementwiseMul
from iron.operators.ffn.op import AIEFFN
from iron.operators.gelu.op import AIEGELU
from iron.operators.gemm.op import AIEGEMM
from iron.operators.layer_norm.op import AIELayerNorm
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.softmax.op import AIESoftmax
from iron.operators.transpose.op import AIETranspose

from iron.applications.transformer_layer_new.pattern.runlist.op import (
    AIETransformerRunlist,
)
from iron.applications.transformer_layer_new.study.block.cases import BlockWorkload
from iron.applications.transformer_layer_new.study.block.run import (
    BLOCK_CONFIG_COLUMNS,
    operator_kwargs,
)
from iron.applications.transformer_layer_new.study.end_to_end.cases import (
    EndToEndWorkload,
    MODE_OPERATORS,
)
from iron.applications.transformer_layer_new.study.end_to_end.select import (
    load_result_rows,
    select_result_rows,
)
from iron.applications.transformer_layer_new.study.end_to_end.modes import (
    resolve_runlist_operator_config,
)
from iron.applications.transformer_layer_new.study.resource_usage.analysis import (
    parse_input_physical_mlir,
)

LOGGER = logging.getLogger(__name__)

Scope = Literal["all", "dataflow_blocks", "runlist_ops"]

RUNLIST_OPERATOR_ORDER = MODE_OPERATORS["runlist"]
APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parents[2]

DATAFLOW_BLOCK_FIELDNAMES = (
    "family_id",
    "family_label",
    "seq_len",
    "hidden_size",
    "head_dim",
    "num_heads",
    "ffn_dim",
    "block_kind",
    "artifact_search_mode",
    "artifact_search_name",
    "preferred_scope_dir",
    "artifact_group_dir",
    "artifact_prj_dir",
    "artifact_input_physical_mlir",
    "artifact_missing",
    "artifact_missing_note",
    "run_status",
    "error_message",
    "source_row_count",
    "source_cases_json",
    "source_candidate_indices_json",
    *(column for columns in BLOCK_CONFIG_COLUMNS.values() for column in columns),
    "compute_tiles_used",
    "aie_tiles_with_buffers",
    "aie_tile_allocated_bytes",
    "aie_tile_memory_utilization",
    "mem_tiles_with_buffers",
    "mem_tile_allocated_bytes",
    "mem_tile_memory_utilization",
    "shim_tiles_with_s2mm",
    "shim_s2mm_channels_used",
    "shim_s2mm_channel_utilization",
    "shim_tiles_with_mm2s",
    "shim_mm2s_channels_used",
    "shim_mm2s_channel_utilization",
)

RUNLIST_FIELDNAMES = (
    "study_case_id",
    "study_case_label",
    "family_id",
    "seq_len",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "attention_head_size",
    "logical_operator",
    "selected_candidate_id",
    "operator_config_json",
    "artifact_search_mode",
    "artifact_search_name",
    "preferred_scope_dir",
    "artifact_group_dir",
    "artifact_prj_dir",
    "artifact_input_physical_mlir",
    "artifact_missing",
    "artifact_missing_note",
    "run_status",
    "error_message",
    "compute_tiles_used",
    "aie_tiles_with_buffers",
    "aie_tile_allocated_bytes",
    "aie_tile_memory_utilization",
    "mem_tiles_with_buffers",
    "mem_tile_allocated_bytes",
    "mem_tile_memory_utilization",
    "shim_tiles_with_s2mm",
    "shim_s2mm_channels_used",
    "shim_s2mm_channel_utilization",
    "shim_tiles_with_mm2s",
    "shim_mm2s_channels_used",
    "shim_mm2s_channel_utilization",
)

_LEADING_PREFIX_RE = re.compile(r"(?P<prefix>\d+)_")
_LAYER_NORM_HASH_RE = re.compile(r"_[0-9a-f]{12}$")


@dataclass(frozen=True)
class ArtifactSpec:
    search_mode: Literal["exact", "glob"]
    search_name: str


@dataclass(frozen=True)
class BuildArtifactIndex:
    build_root: Path
    prj_dirs: tuple[Path, ...]
    by_name: dict[str, tuple[Path, ...]]


@dataclass(frozen=True)
class ArtifactMatch:
    preferred_scope_dir: Path | None
    artifact_group_dir: Path | None
    artifact_prj_dir: Path | None
    artifact_input_physical_mlir: Path | None
    artifact_missing: bool
    artifact_missing_note: str
    run_status: str
    error_message: str


def default_output_dir() -> Path:
    return APP_ROOT / "results" / "resource_usage"


def default_build_root() -> Path:
    return REPO_ROOT / "build" / "transformer_layer_new_end_to_end"


def _snapshot_result_roots(app_root: Path = APP_ROOT) -> tuple[Path, ...]:
    return tuple(sorted(app_root.glob("results_commit*"), reverse=True))


def default_block_results_path(app_root: Path = APP_ROOT) -> Path:
    canonical = app_root / "results" / "block" / "results.csv"
    if canonical.exists():
        return canonical
    for snapshot_root in _snapshot_result_roots(app_root):
        candidate = snapshot_root / "block" / "results.csv"
        if candidate.exists():
            return candidate
    return canonical


def default_end_to_end_results_path(app_root: Path = APP_ROOT) -> Path:
    canonical_dir = app_root / "results" / "end_to_end"
    preferred = (
        canonical_dir / "results_all_power.csv",
        canonical_dir / "results.csv",
    )
    for path in preferred:
        if path.exists():
            return path
    for snapshot_root in _snapshot_result_roots(app_root):
        snapshot_dir = snapshot_root / "end_to_end"
        for name in ("results_all_power.csv", "results.csv"):
            candidate = snapshot_dir / name
            if candidate.exists():
                return candidate
    return preferred[0]


def _leading_numeric_prefix(path: Path) -> int:
    match = _LEADING_PREFIX_RE.match(path.name)
    if match is None:
        return -1
    return int(match.group("prefix"))


def _artifact_sort_key(path: Path, *, prefer_isolated: bool) -> tuple[int, int, str]:
    parent_name = path.parent.name
    if prefer_isolated:
        if re.match(r"\d+_dataflow_", parent_name):
            rank = 0
        elif re.match(r"\d+_mode_dataflow_", parent_name):
            rank = 2
        elif parent_name.startswith("mode_dataflow_"):
            rank = 1
        else:
            rank = 3
    else:
        if parent_name.startswith("mode_runlist_"):
            rank = 0
        elif re.match(r"\d+_mode_runlist_", parent_name):
            rank = 1
        elif re.match(r"\d+_runlist_", parent_name):
            rank = 2
        else:
            rank = 3
    return (rank, -_leading_numeric_prefix(path.parent), str(path))


def build_artifact_index(build_root: str | Path) -> BuildArtifactIndex:
    root = Path(build_root)
    if not root.exists():
        return BuildArtifactIndex(build_root=root, prj_dirs=(), by_name={})
    prj_dirs = tuple(sorted(path.parent for path in root.rglob("input_physical.mlir")))
    by_name: dict[str, list[Path]] = {}
    for prj_dir in prj_dirs:
        by_name.setdefault(prj_dir.name, []).append(prj_dir)
    return BuildArtifactIndex(
        build_root=root,
        prj_dirs=prj_dirs,
        by_name={name: tuple(paths) for name, paths in by_name.items()},
    )


def _value_from_csv(value: str) -> object:
    stripped = value.strip()
    if stripped == "":
        return None
    lower = stripped.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return stripped


def _bool_from_csv(value: str | object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _dataflow_block_candidate(row: dict[str, str]) -> tuple[object, ...]:
    block_kind = str(row["block_kind"])
    return tuple(
        _value_from_csv(str(row.get(column, "")))
        for column in BLOCK_CONFIG_COLUMNS[block_kind]
    )


def _empty_block_config_row() -> dict[str, object]:
    return {
        column: "" for columns in BLOCK_CONFIG_COLUMNS.values() for column in columns
    }


def _block_config_row(
    block_kind: str,
    candidate: tuple[object, ...],
) -> dict[str, object]:
    row = _empty_block_config_row()
    for column, value in zip(BLOCK_CONFIG_COLUMNS[block_kind], candidate):
        row[column] = value
    return row


def _workload_from_block_row(row: dict[str, str]) -> BlockWorkload:
    return BlockWorkload(
        seq_len=int(row["seq_len"]),
        head_dim=int(row["head_dim"]),
        num_heads=int(row["num_heads"]),
        ffn_dim=int(row["ffn_dim"]),
    )


def _make_temporary_context(tempdir: str | Path) -> AIEContext:
    context = AIEContext()
    context.build_dir = Path(tempdir)
    context.build_dir.mkdir(parents=True, exist_ok=True)
    return context


def _zero_vector(size: int) -> torch.Tensor:
    return torch.zeros((size,), dtype=torch.bfloat16)


def _dummy_weight_kwargs(
    hidden_size: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    return {
        "ln1_weight": _zero_vector(hidden_size),
        "ln2_weight": _zero_vector(hidden_size),
        "q_weight": torch.zeros((hidden_size, hidden_size), dtype=torch.bfloat16),
        "k_weight": torch.zeros((hidden_size, hidden_size), dtype=torch.bfloat16),
        "v_weight": torch.zeros((hidden_size, hidden_size), dtype=torch.bfloat16),
        "attn_output_weight": torch.zeros(
            (hidden_size, hidden_size), dtype=torch.bfloat16
        ),
        "ffn_up_weight": torch.zeros(
            (hidden_size, intermediate_size), dtype=torch.bfloat16
        ),
        "ffn_down_weight": torch.zeros(
            (intermediate_size, hidden_size), dtype=torch.bfloat16
        ),
    }


def _dataflow_block_artifact_specs_from_row(
    row: dict[str, str],
) -> tuple[ArtifactSpec, ...]:
    block_kind = str(row["block_kind"])
    workload = _workload_from_block_row(row)
    candidate = _dataflow_block_candidate(row)
    kwargs = operator_kwargs(workload, block_kind, candidate)
    with tempfile.TemporaryDirectory(prefix="tl_resource_usage_") as tempdir:
        context = _make_temporary_context(tempdir)

        if block_kind == "qkv_proj":
            operator = AIEQKVProj(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts()
            stem = xclbin_artifact.path.stem
            suffix = stem.removeprefix("qkv_proj_")
            return (
                ArtifactSpec("exact", f"{stem}.mlir.prj"),
                ArtifactSpec("exact", f"encoder_dataflow_qkvo_proj_{suffix}.mlir.prj"),
            )

        if block_kind == "mha_out_proj":
            operator = AIEMHAOutProj(context=context, skip_add_to_list=True, **kwargs)
            operator.set_up_artifacts()
            return (
                ArtifactSpec("exact", f"{operator.xclbin_artifact.path.stem}.mlir.prj"),
            )

        if block_kind == "addnorm":
            operator = AIEAddAndNorm(
                context=context,
                skip_add_to_list=True,
                weights=_zero_vector(workload.hidden_size),
                **kwargs,
            )
            xclbin_artifact, _ = operator.get_artifacts()
            suffix = xclbin_artifact.path.stem.removeprefix("weighted_layer_norm_")
            return (
                ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),
                ArtifactSpec("exact", f"encoder_dataflow_add_norm1_{suffix}.mlir.prj"),
                ArtifactSpec("exact", f"encoder_dataflow_add_norm2_{suffix}.mlir.prj"),
            )

        if block_kind == "ffn":
            operator = AIEFFN(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts()
            suffix = xclbin_artifact.path.stem.removeprefix("ffn_")
            return (
                ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),
                ArtifactSpec("exact", f"encoder_dataflow_ffn_{suffix}.mlir.prj"),
            )

    raise ValueError(f"Unsupported block kind: {block_kind}")


def _runlist_operator_artifact_specs(
    workload: EndToEndWorkload,
    logical_operator: str,
    operator_config: dict[str, object],
) -> tuple[ArtifactSpec, ...]:
    kwargs = resolve_runlist_operator_config(
        workload.seq_len,
        workload.hidden_size,
        workload.intermediate_size,
        workload.num_attention_heads,
        operator_config={logical_operator: operator_config},
    )[logical_operator]
    prefix = f"encoder_runlist_{logical_operator}_"
    with tempfile.TemporaryDirectory(prefix="tl_resource_usage_") as tempdir:
        context = _make_temporary_context(tempdir)

        if logical_operator in {
            "qkvo_proj",
            "attn_scores",
            "attn_output",
            "up_proj",
            "down_proj",
        }:
            if workload.seq_len >= 16384 and logical_operator in {
                "attn_scores",
                "attn_output",
            }:
                weights = _dummy_weight_kwargs(
                    workload.hidden_size,
                    workload.intermediate_size,
                )
                operator = AIETransformerRunlist(
                    seq_len=workload.seq_len,
                    hidden_size=workload.hidden_size,
                    intermediate_size=workload.intermediate_size,
                    num_heads=workload.num_attention_heads,
                    ln1_weight=weights["ln1_weight"],
                    ln2_weight=weights["ln2_weight"],
                    operator_config={logical_operator: operator_config},
                    context=context,
                )
                operator.set_up_artifacts()
                xclbin_artifact = getattr(operator, f"{logical_operator}_xclbin")
                exact_name = f"{xclbin_artifact.path.stem}.mlir.prj"
                suffix = exact_name.removeprefix(f"encoder_runlist_{logical_operator}_")
                return (
                    ArtifactSpec("exact", exact_name),
                    ArtifactSpec(
                        "glob",
                        f"encoder_runlist_{logical_operator}_block_*_{suffix}",
                    ),
                )
            operator = AIEGEMM(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            exact_name = f"{xclbin_artifact.path.stem}.mlir.prj"
            if logical_operator in {"attn_scores", "attn_output"}:
                suffix = exact_name.removeprefix(f"encoder_runlist_{logical_operator}_")
                return (
                    ArtifactSpec("exact", exact_name),
                    ArtifactSpec(
                        "glob",
                        f"encoder_runlist_{logical_operator}_block_*_{suffix}",
                    ),
                )
            return (ArtifactSpec("exact", exact_name),)

        if logical_operator == "k_transpose":
            operator = AIETranspose(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            return (ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),)

        if logical_operator == "attn_scale":
            operator = AIEElementwiseMul(
                context=context, skip_add_to_list=True, **kwargs
            )
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            return (ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),)

        if logical_operator == "attn_softmax":
            operator = AIESoftmax(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            return (ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),)

        if logical_operator == "add":
            operator = AIEElementwiseAdd(
                context=context, skip_add_to_list=True, **kwargs
            )
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            return (ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),)

        if logical_operator in {"ln1", "ln2"}:
            operator = AIELayerNorm(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            base_name = _LAYER_NORM_HASH_RE.sub("", xclbin_artifact.path.stem)
            return (ArtifactSpec("glob", f"{base_name}_*.mlir.prj"),)

        if logical_operator == "gelu":
            operator = AIEGELU(context=context, skip_add_to_list=True, **kwargs)
            xclbin_artifact, _ = operator.get_artifacts(prefix=prefix)
            return (ArtifactSpec("exact", f"{xclbin_artifact.path.stem}.mlir.prj"),)

    raise ValueError(f"Unsupported runlist logical operator: {logical_operator}")


def _find_mode_scope_dir(
    build_root: Path,
    *,
    execution_mode: str,
    hidden_size: int,
    seq_len: int,
) -> Path | None:
    stable = build_root / f"mode_{execution_mode}_{hidden_size}_{seq_len}"
    if stable.exists():
        return stable
    prefixed = sorted(
        build_root.glob(f"*_mode_{execution_mode}_{hidden_size}_{seq_len}"),
        key=_leading_numeric_prefix,
        reverse=True,
    )
    return prefixed[0] if prefixed else None


def _glob_paths(root: Path, pattern: str) -> list[Path]:
    return sorted(
        [path for path in root.glob(pattern) if (path / "input_physical.mlir").exists()]
    )


def _global_glob_paths(index: BuildArtifactIndex, pattern: str) -> list[Path]:
    return sorted([path for path in index.prj_dirs if fnmatch(path.name, pattern)])


def _locate_artifact(
    index: BuildArtifactIndex,
    *,
    specs: tuple[ArtifactSpec, ...],
    preferred_scope_dir: Path | None,
    prefer_isolated: bool,
) -> ArtifactMatch:
    notes: list[str] = []
    if not index.build_root.exists():
        return ArtifactMatch(
            preferred_scope_dir=preferred_scope_dir,
            artifact_group_dir=None,
            artifact_prj_dir=None,
            artifact_input_physical_mlir=None,
            artifact_missing=True,
            artifact_missing_note=f"build root does not exist: {index.build_root}",
            run_status="missing_artifact",
            error_message="",
        )

    if preferred_scope_dir is not None and not preferred_scope_dir.exists():
        notes.append(f"preferred scope missing: {preferred_scope_dir}")

    for spec in specs:
        if preferred_scope_dir is not None and preferred_scope_dir.exists():
            scoped_matches = (
                [preferred_scope_dir / spec.search_name]
                if spec.search_mode == "exact"
                else _glob_paths(preferred_scope_dir, spec.search_name)
            )
            scoped_matches = [
                path
                for path in scoped_matches
                if (path / "input_physical.mlir").exists()
            ]
            if scoped_matches:
                prj_dir = sorted(scoped_matches)[0]
                return ArtifactMatch(
                    preferred_scope_dir=preferred_scope_dir,
                    artifact_group_dir=prj_dir.parent,
                    artifact_prj_dir=prj_dir,
                    artifact_input_physical_mlir=prj_dir / "input_physical.mlir",
                    artifact_missing=False,
                    artifact_missing_note="; ".join(notes),
                    run_status="passed",
                    error_message="",
                )

        global_matches = (
            list(index.by_name.get(spec.search_name, ()))
            if spec.search_mode == "exact"
            else _global_glob_paths(index, spec.search_name)
        )
        if not global_matches:
            continue
        chosen = sorted(
            global_matches,
            key=lambda path: _artifact_sort_key(path, prefer_isolated=prefer_isolated),
        )[0]
        if preferred_scope_dir is not None and chosen.parent != preferred_scope_dir:
            notes.append(
                f"used fallback artifact outside preferred scope: {chosen.parent}"
            )
        return ArtifactMatch(
            preferred_scope_dir=preferred_scope_dir,
            artifact_group_dir=chosen.parent,
            artifact_prj_dir=chosen,
            artifact_input_physical_mlir=chosen / "input_physical.mlir",
            artifact_missing=False,
            artifact_missing_note="; ".join(notes),
            run_status="passed",
            error_message="",
        )

    missing_note = "; ".join(
        note
        for note in [
            *notes,
            f"no matching artifact found for {[spec.search_name for spec in specs]}",
        ]
        if note
    )
    return ArtifactMatch(
        preferred_scope_dir=preferred_scope_dir,
        artifact_group_dir=None,
        artifact_prj_dir=None,
        artifact_input_physical_mlir=None,
        artifact_missing=True,
        artifact_missing_note=missing_note,
        run_status="missing_artifact",
        error_message="",
    )


def _resource_row(match: ArtifactMatch) -> dict[str, object]:
    if match.artifact_input_physical_mlir is None:
        return {
            "preferred_scope_dir": str(match.preferred_scope_dir or ""),
            "artifact_group_dir": "",
            "artifact_prj_dir": "",
            "artifact_input_physical_mlir": "",
            "artifact_missing": True,
            "artifact_missing_note": match.artifact_missing_note,
            "run_status": match.run_status,
            "error_message": match.error_message,
            "compute_tiles_used": "",
            "aie_tiles_with_buffers": "",
            "aie_tile_allocated_bytes": "",
            "aie_tile_memory_utilization": "",
            "mem_tiles_with_buffers": "",
            "mem_tile_allocated_bytes": "",
            "mem_tile_memory_utilization": "",
            "shim_tiles_with_s2mm": "",
            "shim_s2mm_channels_used": "",
            "shim_s2mm_channel_utilization": "",
            "shim_tiles_with_mm2s": "",
            "shim_mm2s_channels_used": "",
            "shim_mm2s_channel_utilization": "",
        }

    usage = parse_input_physical_mlir(match.artifact_input_physical_mlir)
    return {
        "preferred_scope_dir": str(match.preferred_scope_dir or ""),
        "artifact_group_dir": str(match.artifact_group_dir or ""),
        "artifact_prj_dir": str(match.artifact_prj_dir or ""),
        "artifact_input_physical_mlir": str(match.artifact_input_physical_mlir),
        "artifact_missing": False,
        "artifact_missing_note": match.artifact_missing_note,
        "run_status": match.run_status,
        "error_message": match.error_message,
        **usage.as_row(),
    }


def _group_block_rows(
    rows: list[dict[str, str]],
    *,
    family_filter: str,
    seq_len_filter: str,
) -> list[tuple[dict[str, str], list[dict[str, str]]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("run_status") != "passed":
            continue
        if not _bool_from_csv(row.get("is_best", "")):
            continue
        if family_filter != "all" and row.get("family_id") != family_filter:
            continue
        if seq_len_filter != "all" and int(row.get("seq_len", "0")) != int(
            seq_len_filter
        ):
            continue
        block_kind = str(row["block_kind"])
        try:
            specs = _dataflow_block_artifact_specs_from_row(row)
        except Exception:
            specs = ()
        search_name = (
            specs[0].search_name
            if specs
            else json.dumps(
                [row.get(column, "") for column in BLOCK_CONFIG_COLUMNS[block_kind]]
            )
        )
        grouped.setdefault((block_kind, search_name), []).append(row)

    result: list[tuple[dict[str, str], list[dict[str, str]]]] = []
    for rows_for_key in grouped.values():
        representative = sorted(
            rows_for_key,
            key=lambda row: (int(row["hidden_size"]), int(row["seq_len"])),
        )[0]
        result.append((representative, rows_for_key))
    return sorted(
        result,
        key=lambda item: (
            item[0]["block_kind"],
            int(item[0]["hidden_size"]),
            int(item[0]["seq_len"]),
        ),
    )


def export_dataflow_block_best_configs(
    *,
    block_results_input: str | Path,
    family_filter: str,
    seq_len_filter: str,
    build_index: BuildArtifactIndex,
) -> list[dict[str, object]]:
    rows = load_result_rows(block_results_input)
    export_rows: list[dict[str, object]] = []
    for representative, grouped_rows in _group_block_rows(
        rows,
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
    ):
        block_kind = str(representative["block_kind"])
        candidate = _dataflow_block_candidate(representative)
        workload = _workload_from_block_row(representative)
        try:
            specs = _dataflow_block_artifact_specs_from_row(representative)
            match = _locate_artifact(
                build_index,
                specs=specs,
                preferred_scope_dir=None,
                prefer_isolated=True,
            )
            resource_row = _resource_row(match)
            search_mode = specs[0].search_mode
            search_name = specs[0].search_name
        except Exception as exc:
            resource_row = _resource_row(
                ArtifactMatch(
                    preferred_scope_dir=None,
                    artifact_group_dir=None,
                    artifact_prj_dir=None,
                    artifact_input_physical_mlir=None,
                    artifact_missing=True,
                    artifact_missing_note="artifact resolution failed",
                    run_status="failed_exception",
                    error_message=str(exc),
                )
            )
            search_mode = ""
            search_name = ""

        source_cases = [
            {
                "family_id": row["family_id"],
                "seq_len": int(row["seq_len"]),
            }
            for row in sorted(
                grouped_rows, key=lambda row: (row["family_id"], int(row["seq_len"]))
            )
        ]
        source_candidate_indices = [
            int(row["candidate_index"])
            for row in sorted(
                grouped_rows, key=lambda row: (row["family_id"], int(row["seq_len"]))
            )
        ]
        export_rows.append(
            {
                "family_id": representative["family_id"],
                "family_label": representative["family_label"],
                "seq_len": int(representative["seq_len"]),
                "hidden_size": int(representative["hidden_size"]),
                "head_dim": int(representative["head_dim"]),
                "num_heads": int(representative["num_heads"]),
                "ffn_dim": int(representative["ffn_dim"]),
                "block_kind": block_kind,
                "artifact_search_mode": search_mode,
                "artifact_search_name": search_name,
                "source_row_count": len(grouped_rows),
                "source_cases_json": json.dumps(source_cases, sort_keys=True),
                "source_candidate_indices_json": json.dumps(source_candidate_indices),
                **_block_config_row(block_kind, candidate),
                **resource_row,
            }
        )
    return export_rows


def export_runlist_selected_ops(
    *,
    end_to_end_results_input: str | Path,
    family_filter: str,
    seq_len_filter: str,
    build_index: BuildArtifactIndex,
) -> list[dict[str, object]]:
    selected_rows = select_result_rows(
        load_result_rows(end_to_end_results_input),
        family_filter=family_filter,
        seq_len_filter=seq_len_filter,
        mode_filter="runlist",
    )
    export_rows: list[dict[str, object]] = []
    for selected_row in selected_rows:
        workload = EndToEndWorkload(
            seq_len=selected_row.seq_len,
            hidden_size=selected_row.hidden_size,
            intermediate_size=selected_row.intermediate_size,
            num_attention_heads=selected_row.num_attention_heads,
        )
        preferred_scope_dir = _find_mode_scope_dir(
            build_index.build_root,
            execution_mode="runlist",
            hidden_size=selected_row.hidden_size,
            seq_len=selected_row.seq_len,
        )
        for logical_operator in RUNLIST_OPERATOR_ORDER:
            if logical_operator not in selected_row.selected_config:
                continue
            try:
                specs = _runlist_operator_artifact_specs(
                    workload,
                    logical_operator,
                    selected_row.selected_config[logical_operator],
                )
                match = _locate_artifact(
                    build_index,
                    specs=specs,
                    preferred_scope_dir=preferred_scope_dir,
                    prefer_isolated=False,
                )
                resource_row = _resource_row(match)
                search_mode = specs[0].search_mode
                search_name = specs[0].search_name
            except Exception as exc:
                resource_row = _resource_row(
                    ArtifactMatch(
                        preferred_scope_dir=preferred_scope_dir,
                        artifact_group_dir=None,
                        artifact_prj_dir=None,
                        artifact_input_physical_mlir=None,
                        artifact_missing=True,
                        artifact_missing_note="artifact resolution failed",
                        run_status="failed_exception",
                        error_message=str(exc),
                    )
                )
                search_mode = ""
                search_name = ""

            export_rows.append(
                {
                    "study_case_id": selected_row.study_case_id,
                    "study_case_label": selected_row.study_case_label,
                    "family_id": selected_row.study_case_id,
                    "seq_len": selected_row.seq_len,
                    "hidden_size": selected_row.hidden_size,
                    "intermediate_size": selected_row.intermediate_size,
                    "num_attention_heads": selected_row.num_attention_heads,
                    "attention_head_size": selected_row.attention_head_size,
                    "logical_operator": logical_operator,
                    "selected_candidate_id": selected_row.selected_candidate_ids.get(
                        logical_operator, ""
                    ),
                    "operator_config_json": json.dumps(
                        selected_row.selected_config[logical_operator],
                        sort_keys=True,
                    ),
                    "artifact_search_mode": search_mode,
                    "artifact_search_name": search_name,
                    **resource_row,
                }
            )
    return export_rows


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export resource-usage summaries from end-to-end study compilation "
            "artifacts without recompiling."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("all", "dataflow_blocks", "runlist_ops"),
        default="all",
    )
    parser.add_argument(
        "--family", choices=("all", "baseline_768", "baseline_1024"), default="all"
    )
    parser.add_argument(
        "--seq-len",
        default="all",
        choices=(
            "all",
            "64",
            "128",
            "256",
            "512",
            "1024",
            "2048",
            "4096",
            "8192",
            "16384",
        ),
    )
    parser.add_argument(
        "--block-results-input",
        type=Path,
        default=default_block_results_path(),
    )
    parser.add_argument(
        "--end-to-end-results-input",
        type=Path,
        default=default_end_to_end_results_path(),
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=default_build_root(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(message)s"
    )

    build_index = build_artifact_index(args.build_root)
    if not build_index.build_root.exists():
        LOGGER.warning("Build root does not exist: %s", build_index.build_root)

    if args.scope in ("all", "dataflow_blocks"):
        dataflow_rows = export_dataflow_block_best_configs(
            block_results_input=args.block_results_input,
            family_filter=args.family,
            seq_len_filter=args.seq_len,
            build_index=build_index,
        )
        dataflow_output = args.output_dir / "dataflow_block_best_configs.csv"
        _write_csv(dataflow_output, DATAFLOW_BLOCK_FIELDNAMES, dataflow_rows)
        LOGGER.info(
            "Wrote %d dataflow block rows to %s", len(dataflow_rows), dataflow_output
        )

    if args.scope in ("all", "runlist_ops"):
        runlist_rows = export_runlist_selected_ops(
            end_to_end_results_input=args.end_to_end_results_input,
            family_filter=args.family,
            seq_len_filter=args.seq_len,
            build_index=build_index,
        )
        runlist_output = args.output_dir / "runlist_selected_ops.csv"
        _write_csv(runlist_output, RUNLIST_FIELDNAMES, runlist_rows)
        LOGGER.info("Wrote %d runlist rows to %s", len(runlist_rows), runlist_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
