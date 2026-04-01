#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from ml_dtypes import bfloat16

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iron.common import AIEContext
from iron.common.utils import torch_to_numpy
from iron.operators.addnorm_ffn_addnorm.op import AIEAddNormFFNAddNorm
from iron.operators.addnorm_ffn_addnorm.reference import (
    generate_golden_reference as generate_block3_reference,
)
from iron.operators.addnorm_ffn_addnorm.topology import addnorm_ffn_addnorm_topologies
from iron.operators.mha_out_proj.op import AIEMHAOutProj
from iron.operators.mha_out_proj.reference import (
    generate_golden_reference as generate_block2_reference,
)
from iron.operators.mha_out_proj.topology import (
    mha_out_proj_practical_topologies,
    mha_out_proj_topologies,
)
from iron.operators.qkv_proj.op import AIEQKVProj
from iron.operators.qkv_proj.reference import (
    generate_golden_reference as generate_block1_reference,
)
from iron.operators.qkv_proj.topology import qkv_proj_topologies


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _count_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rel_tol: float,
    abs_tol: float,
) -> int:
    return int((~torch.isclose(actual, expected, rtol=rel_tol, atol=abs_tol)).sum())


def _flatten_head_major(tensor: torch.Tensor) -> torch.Tensor:
    num_heads, seq_len, head_dim = tensor.shape
    return tensor.permute(1, 0, 2).contiguous().view(seq_len, num_heads * head_dim)


@dataclass
class CaseSpec:
    block: str
    case_id: str
    source_tests: list[str]
    workload: dict[str, int]
    topology_id: str
    topology_family: str
    case_kind: str


def _merge_case_specs(case_specs: list[CaseSpec]) -> list[CaseSpec]:
    merged: dict[tuple[str, str], CaseSpec] = {}
    for case in case_specs:
        key = (case.block, case.case_id)
        if key not in merged:
            merged[key] = case
            continue
        existing = merged[key]
        existing.source_tests = sorted(
            set(existing.source_tests).union(case.source_tests)
        )
    return list(merged.values())


def collect_block1_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for seq_len, num_heads, head_dim in (
        (64, 12, 64),
        (512, 12, 64),
        (64, 16, 64),
        (512, 16, 64),
    ):
        hidden_size = num_heads * head_dim
        for topology in qkv_proj_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
        ):
            topology_id = str(topology["topology_id"])
            cases.append(
                CaseSpec(
                    block="block1",
                    case_id=f"block1_{seq_len}_{num_heads}_{head_dim}_{topology_id}",
                    source_tests=["test_qkv_proj"],
                    workload={
                        "seq_len": seq_len,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
                        "hidden_size": hidden_size,
                    },
                    topology_id=topology_id,
                    topology_family=str(topology["topology_family"]),
                    case_kind="runtime_matrix",
                )
            )

    for seq_len, num_heads, head_dim in ((64, 24, 64), (64, 12, 80)):
        hidden_size = num_heads * head_dim
        topology = qkv_proj_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
        )[0]
        topology_id = str(topology["topology_id"])
        cases.append(
            CaseSpec(
                block="block1",
                case_id=(
                    f"block1_generalized_{seq_len}_{num_heads}_{head_dim}_{topology_id}"
                ),
                source_tests=["test_qkv_proj_generalized_runtime_workloads"],
                workload={
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "hidden_size": hidden_size,
                },
                topology_id=topology_id,
                topology_family=str(topology["topology_family"]),
                case_kind="generalized_spot_check",
            )
        )

    cases.append(
        CaseSpec(
            block="block1",
            case_id="block1_known_runtime_k24_n32_topology",
            source_tests=[
                "test_qkv_proj_known_runtime_k24_n32_topology_compiles_and_runs"
            ],
            workload={
                "seq_len": 64,
                "num_heads": 24,
                "head_dim": 64,
                "hidden_size": 24 * 64,
            },
            topology_id="m16_k24_n32_c8_ps4_pe8",
            topology_family="shared_runtime_qkv_proj",
            case_kind="known_runtime_case",
        )
    )
    return _merge_case_specs(cases)


def collect_block2_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for seq_len, head_dim, num_heads in (
        (64, 64, 1),
        (512, 64, 1),
        (64, 64, 12),
        (512, 64, 12),
        (64, 64, 16),
        (512, 64, 16),
        (1984, 64, 16),
        (2048, 64, 16),
    ):
        supported_ids = {
            str(topology["topology_id"])
            for topology in mha_out_proj_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        }
        supported_topology_map = {
            str(topology["topology_id"]): topology
            for topology in mha_out_proj_topologies(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        }
        for topology in mha_out_proj_practical_topologies(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ):
            topology_id = str(topology["topology_id"])
            if topology_id not in supported_ids:
                continue
            runtime_topology = supported_topology_map[topology_id]
            cases.append(
                CaseSpec(
                    block="block2",
                    case_id=(
                        f"mha_out_proj_{num_heads}heads_{seq_len}seq_"
                        f"{head_dim}hdim_{topology_id}"
                    ),
                    source_tests=["test_mha_out_proj"],
                    workload={
                        "seq_len": seq_len,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
                        "hidden_size": num_heads * head_dim,
                    },
                    topology_id=topology_id,
                    topology_family=str(runtime_topology["topology_family"]),
                    case_kind="runtime_matrix",
                )
            )

    topology = mha_out_proj_topologies(seq_len=64, num_heads=24, head_dim=64)[0]
    cases.append(
        CaseSpec(
            block="block2",
            case_id=f"block2_generalized_64_24_64_{topology['topology_id']}",
            source_tests=["test_generalized_block2_runtime_topology_runs_numerically"],
            workload={
                "seq_len": 64,
                "num_heads": 24,
                "head_dim": 64,
                "hidden_size": 24 * 64,
            },
            topology_id=str(topology["topology_id"]),
            topology_family=str(topology["topology_family"]),
            case_kind="generalized_spot_check",
        )
    )
    return _merge_case_specs(cases)


def collect_block3_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for seq_len, hidden_size, intermediate_size in (
        (64, 768, 3072),
        (512, 768, 3072),
        (64, 1024, 4096),
        (512, 1024, 4096),
        (64, 2048, 8192),
        (512, 2048, 8192),
    ):
        for topology in addnorm_ffn_addnorm_topologies(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ):
            topology_id = str(topology["topology_id"])
            cases.append(
                CaseSpec(
                    block="block3",
                    case_id=(
                        f"block3_{seq_len}x{hidden_size}x{intermediate_size}_"
                        f"{topology_id}"
                    ),
                    source_tests=["test_addnorm_ffn_addnorm"],
                    workload={
                        "seq_len": seq_len,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                    },
                    topology_id=topology_id,
                    topology_family=str(topology["topology_family"]),
                    case_kind="runtime_matrix",
                )
            )

    topology = addnorm_ffn_addnorm_topologies(
        seq_len=64,
        hidden_size=1536,
        intermediate_size=6144,
    )[0]
    cases.append(
        CaseSpec(
            block="block3",
            case_id=(f"block3_generalized_64x1536x6144_{topology['topology_id']}"),
            source_tests=["test_generalized_block3_runtime_topology_runs_numerically"],
            workload={
                "seq_len": 64,
                "hidden_size": 1536,
                "intermediate_size": 6144,
            },
            topology_id=str(topology["topology_id"]),
            topology_family=str(topology["topology_family"]),
            case_kind="generalized_spot_check",
        )
    )
    return _merge_case_specs(cases)


def _measure_block1(
    case: CaseSpec, *, warmup_iters: int, timed_iters: int
) -> dict[str, Any]:
    seq_len = case.workload["seq_len"]
    num_heads = case.workload["num_heads"]
    head_dim = case.workload["head_dim"]
    hidden_size = case.workload["hidden_size"]
    context = AIEContext()
    try:
        golden_ref = generate_block1_reference(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            seed=7 if case.case_kind == "runtime_matrix" else 11,
        )
        if case.case_kind == "known_runtime_case":
            golden_ref = generate_block1_reference(
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                seed=17,
            )

        operator = AIEQKVProj(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            topology_id=case.topology_id,
            context=context,
        )
        operator.q_proj.weight = golden_ref["q_proj_weight"].contiguous()
        operator.k_proj.weight = golden_ref["k_proj_weight"].contiguous()
        operator.v_proj.weight = golden_ref["v_proj_weight"].contiguous()
        context.compile_all()
        context.prepare_runtime()

        for _ in range(warmup_iters):
            operator.run_runlist()

        for buf_name in ("Q", "K", "V"):
            operator.write_buffer(
                buf_name, np.zeros(operator.buffers[buf_name], dtype=np.uint8)
            )
        operator.write_buffer(
            "A",
            operator._pad_hidden_states(torch_to_numpy(golden_ref["hidden_states"])),
        )

        latencies_us: list[float] = []
        total_bytes = (
            operator.buffers["A"]
            + operator.buffers["Q"]
            + operator.buffers["K"]
            + operator.buffers["V"]
        )
        bandwidths_gbps: list[float] = []
        for _ in range(timed_iters):
            elapsed = operator.run_runlist()
            latency_us = elapsed * 1e6
            latencies_us.append(latency_us)
            bandwidths_gbps.append(total_bytes / elapsed / 1e9)

        q = operator._read_projection_output("Q", seq_len)
        k = operator._read_projection_output("K", seq_len)
        v = operator._read_projection_output("V", seq_len)
        q_ref = _flatten_head_major(golden_ref["q"])
        k_ref = _flatten_head_major(golden_ref["k"])
        v_ref = _flatten_head_major(golden_ref["v"])
        rel_tol = 4.0e-2
        abs_tol = 1.5e-1
        error_threshold = 0.005
        max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)
        q_errors = _count_errors(q, q_ref, rel_tol=rel_tol, abs_tol=abs_tol)
        k_errors = _count_errors(k, k_ref, rel_tol=rel_tol, abs_tol=abs_tol)
        v_errors = _count_errors(v, v_ref, rel_tol=rel_tol, abs_tol=abs_tol)
        return {
            "latency_us": _stats(latencies_us),
            "bandwidth_gbps": _stats(bandwidths_gbps),
            "error_counts": {
                "Q": q_errors,
                "K": k_errors,
                "V": v_errors,
            },
            "max_acceptable_errors": max_acceptable_errors,
            "status": (
                "passed"
                if q_errors <= max_acceptable_errors
                and k_errors <= max_acceptable_errors
                and v_errors <= max_acceptable_errors
                else "failed"
            ),
            "timed_iterations": timed_iters,
            "warmup_iterations": warmup_iters,
            "dynamic_io_bytes": total_bytes,
        }
    finally:
        context.reset_runtime()


def _measure_block2(
    case: CaseSpec, *, warmup_iters: int, timed_iters: int
) -> dict[str, Any]:
    seq_len = case.workload["seq_len"]
    num_heads = case.workload["num_heads"]
    head_dim = case.workload["head_dim"]
    context = AIEContext()
    try:
        golden_ref = generate_block2_reference(
            seq_len=seq_len,
            d=head_dim,
            heads=num_heads,
            debug=0,
        )
        operator = AIEMHAOutProj(
            num_heads=num_heads,
            seq_len=seq_len,
            d=head_dim,
            topology_id=case.topology_id,
            debug=0,
            context=context,
        )
        context.compile_all()
        context.prepare_runtime()

        for _ in range(warmup_iters):
            operator.run_runlist()

        operator.write_buffer("O", np.zeros(operator.buffers["O"], dtype=np.uint8))
        operator.write_buffer("Q", golden_ref["Q"].flatten())
        operator.write_buffer("K", golden_ref["K"].flatten())
        operator.write_buffer("V", golden_ref["V"].flatten())
        operator.write_buffer("W_O", golden_ref["W_O"].flatten())

        latencies_us: list[float] = []
        total_bytes = (
            operator.buffers["Q"]
            + operator.buffers["K"]
            + operator.buffers["V"]
            + operator.buffers["W_O"]
            + operator.buffers["O"]
        )
        bandwidths_gbps: list[float] = []
        for _ in range(timed_iters):
            elapsed = operator.run_runlist()
            latency_us = elapsed * 1e6
            latencies_us.append(latency_us)
            bandwidths_gbps.append(total_bytes / elapsed / 1e9)

        output = operator.read_buffer(
            "O",
            shape=(seq_len, num_heads * head_dim),
            copy=True,
            dtype=bfloat16,
        ).astype(np.float32)
        expected = (
            torch_to_numpy(golden_ref["O"])
            .reshape(seq_len, num_heads * head_dim)
            .astype(np.float32)
        )
        errors = int((~np.isclose(output, expected, rtol=4.0e-2, atol=1.5e-1)).sum())
        error_threshold = 0.005
        max_acceptable_errors = int(seq_len * head_dim * num_heads * error_threshold)
        return {
            "latency_us": _stats(latencies_us),
            "bandwidth_gbps": _stats(bandwidths_gbps),
            "error_counts": {"O": errors},
            "max_acceptable_errors": max_acceptable_errors,
            "status": "passed" if errors <= max_acceptable_errors else "failed",
            "timed_iterations": timed_iters,
            "warmup_iterations": warmup_iters,
            "dynamic_io_bytes": total_bytes,
        }
    finally:
        context.reset_runtime()


def _measure_block3(
    case: CaseSpec, *, warmup_iters: int, timed_iters: int
) -> dict[str, Any]:
    seq_len = case.workload["seq_len"]
    hidden_size = case.workload["hidden_size"]
    intermediate_size = case.workload["intermediate_size"]
    context = AIEContext()
    try:
        seed = 7
        if case.case_kind == "generalized_spot_check":
            seed = 19
        golden = generate_block3_reference(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            seed=seed,
        )
        operator = AIEAddNormFFNAddNorm(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            topology_id=case.topology_id,
            context=context,
        )
        operator.weight_up_proj = golden["ffn_up_weight"].contiguous().T
        operator.weight_down_proj = golden["ffn_down_weight"].contiguous().T
        operator.ln1_weight = golden["ln1_weight"].contiguous()
        operator.ln2_weight = golden["ln2_weight"].contiguous()
        context.compile_all()
        context.prepare_runtime()

        for _ in range(warmup_iters):
            operator.run_runlist()

        operator.write_buffer("C", np.zeros(operator.buffers["C"], dtype=np.uint8))
        operator.write_buffer("A", torch_to_numpy(golden["hidden_states"]).reshape(-1))
        operator.write_buffer("R", torch_to_numpy(golden["residual"]).reshape(-1))

        latencies_us: list[float] = []
        total_bytes = (
            operator.buffers["A"] + operator.buffers["R"] + operator.buffers["C"]
        )
        bandwidths_gbps: list[float] = []
        for _ in range(timed_iters):
            elapsed = operator.run_runlist()
            latency_us = elapsed * 1e6
            latencies_us.append(latency_us)
            bandwidths_gbps.append(total_bytes / elapsed / 1e9)

        output = operator.read_buffer_as_torch(
            "C",
            shape=(seq_len, hidden_size),
            dtype=bfloat16,
        )
        rel_tol = 4.0e-2
        abs_tol = 1.5e-1
        error_threshold = 0.005
        output_errors = _count_errors(
            output,
            golden["output"],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        max_acceptable_errors = int(seq_len * hidden_size * error_threshold)
        return {
            "latency_us": _stats(latencies_us),
            "bandwidth_gbps": _stats(bandwidths_gbps),
            "error_counts": {"C": output_errors},
            "max_acceptable_errors": max_acceptable_errors,
            "status": "passed" if output_errors <= max_acceptable_errors else "failed",
            "timed_iterations": timed_iters,
            "warmup_iterations": warmup_iters,
            "dynamic_io_bytes": total_bytes,
        }
    finally:
        context.reset_runtime()


def measure_case(
    case: CaseSpec, *, warmup_iters: int, timed_iters: int
) -> dict[str, Any]:
    if case.block == "block1":
        performance = _measure_block1(
            case, warmup_iters=warmup_iters, timed_iters=timed_iters
        )
    elif case.block == "block2":
        performance = _measure_block2(
            case, warmup_iters=warmup_iters, timed_iters=timed_iters
        )
    elif case.block == "block3":
        performance = _measure_block3(
            case, warmup_iters=warmup_iters, timed_iters=timed_iters
        )
    else:
        raise ValueError(f"Unknown block {case.block!r}")

    return {
        "block": case.block,
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "topology_id": case.topology_id,
        "topology_family": case.topology_family,
        "workload": case.workload,
        "source_tests": case.source_tests,
        **performance,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append(
            {
                "block": row["block"],
                "case_id": row["case_id"],
                "case_kind": row["case_kind"],
                "topology_id": row["topology_id"],
                "topology_family": row["topology_family"],
                "seq_len": row["workload"].get("seq_len"),
                "hidden_size": row["workload"].get("hidden_size"),
                "num_heads": row["workload"].get("num_heads"),
                "head_dim": row["workload"].get("head_dim"),
                "intermediate_size": row["workload"].get("intermediate_size"),
                "source_tests": ";".join(row["source_tests"]),
                "status": row["status"],
                "dynamic_io_bytes": row["dynamic_io_bytes"],
                "timed_iterations": row["timed_iterations"],
                "warmup_iterations": row["warmup_iterations"],
                "latency_us_mean": row["latency_us"]["mean"],
                "latency_us_median": row["latency_us"]["median"],
                "latency_us_min": row["latency_us"]["min"],
                "latency_us_max": row["latency_us"]["max"],
                "latency_us_stddev": row["latency_us"]["stddev"],
                "bandwidth_gbps_mean": row["bandwidth_gbps"]["mean"],
                "bandwidth_gbps_median": row["bandwidth_gbps"]["median"],
                "bandwidth_gbps_min": row["bandwidth_gbps"]["min"],
                "bandwidth_gbps_max": row["bandwidth_gbps"]["max"],
                "bandwidth_gbps_stddev": row["bandwidth_gbps"]["stddev"],
                "max_acceptable_errors": row["max_acceptable_errors"],
                "error_counts_json": json.dumps(row["error_counts"], sort_keys=True),
            }
        )

    fieldnames = list(flat_rows[0].keys()) if flat_rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def _build_payload(
    *,
    timed_iters: int,
    warmup_iters: int,
    measured_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = defaultdict(dict)
    for block in ("block1", "block2", "block3"):
        block_rows = [row for row in measured_rows if row["block"] == block]
        summary[block] = {
            "case_count": len(block_rows),
            "failed_case_count": sum(row["status"] != "passed" for row in block_rows),
            "source_tests": sorted(
                {test_name for row in block_rows for test_name in row["source_tests"]}
            ),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "measurement_method": {
            "warmup_iterations": warmup_iters,
            "timed_iterations": timed_iters,
            "latency_source": "operator.run_runlist()",
            "bandwidth_source": "dynamic host-visible input+output bytes divided by measured runlist latency",
        },
        "summary": summary,
        "cases": measured_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-output",
        type=Path,
        default=REPO_ROOT
        / "iron/applications/transformer_layer/study/block_operator_test_case_performance.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=REPO_ROOT
        / "iron/applications/transformer_layer/study/block_operator_test_case_performance.csv",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=5)
    args = parser.parse_args()

    cases = collect_block1_cases() + collect_block2_cases() + collect_block3_cases()
    measured_rows: list[dict[str, Any]] = []
    if args.json_output.exists():
        existing_payload = json.loads(args.json_output.read_text())
        measured_rows = list(existing_payload.get("cases", []))

    measured_by_case_id = {row["case_id"]: row for row in measured_rows}
    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        if case.case_id in measured_by_case_id:
            print(
                f"[{index}/{total_cases}] skipping {case.case_id} (already saved)",
                flush=True,
            )
            continue
        print(f"[{index}/{total_cases}] measuring {case.case_id}", flush=True)
        measured_row = measure_case(
            case,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
        )
        measured_rows.append(measured_row)
        measured_by_case_id[case.case_id] = measured_row
        payload = _build_payload(
            timed_iters=args.timed_iters,
            warmup_iters=args.warmup_iters,
            measured_rows=measured_rows,
        )
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2))
        _write_csv(args.csv_output, measured_rows)

    payload = _build_payload(
        timed_iters=args.timed_iters,
        warmup_iters=args.warmup_iters,
        measured_rows=measured_rows,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2))
    _write_csv(args.csv_output, measured_rows)

    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
