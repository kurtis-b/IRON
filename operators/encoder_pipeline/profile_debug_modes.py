#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

# Make repo root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operators.common.aie_context import AIEContext
from operators.common.aie_device_manager import AIEDeviceManager
from operators.encoder_pipeline.test import (
    DEFAULT_TEST_TIMED_ITERS,
    DEFAULT_TEST_WARMUP_ITERS,
    _case_name,
    _run_encoder_pipeline_case,
    generate_test_params,
)


@dataclass(frozen=True)
class DebugModeSpec:
    name: str
    debug_mode: int
    contributes_to_bottleneck: bool


DEBUG_MODE_SPECS: tuple[DebugModeSpec, ...] = (
    DebugModeSpec("full", -1, False),
    DebugModeSpec("self_attn", 0, False),
    DebugModeSpec("mha_input", 1, False),
    DebugModeSpec("residual", 2, False),
    DebugModeSpec("ffn_up", 3, True),
    DebugModeSpec("ffn_down", 4, True),
    DebugModeSpec("addnorm2", 5, True),
    DebugModeSpec("mha", 6, True),
    DebugModeSpec("addnorm1", 7, True),
)


def _total_error_count(errors: dict[str, list[int]]) -> int:
    return sum(len(v) for v in errors.values())


def _parse_case_filter(raw: str | None, num_cases: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(range(num_cases))
    idxs: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        idx = int(tok)
        if idx < 0 or idx >= num_cases:
            raise ValueError(
                f"--case-index out of range: {idx} (valid: 0..{num_cases - 1})"
            )
        idxs.append(idx)
    if not idxs:
        raise ValueError("--case-index resolved to an empty selection")
    return sorted(set(idxs))


def _run_single_mode(
    case: tuple[int, ...],
    mode: DebugModeSpec,
    warmup_iters: int,
    timed_iters: int,
    ln1_staging_design: str,
) -> dict:
    ctx = None
    try:
        ctx = AIEContext()
        errors, latency_us, bandwidth_gbps = _run_encoder_pipeline_case(
            case,
            debug_mode=mode.debug_mode,
            aie_context=ctx,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            ln1_staging_design=ln1_staging_design,
        )
        return {
            "mode": mode.name,
            "debug_mode": mode.debug_mode,
            "status": "pass",
            "latency_us": float(latency_us),
            "bandwidth_gbps": float(bandwidth_gbps),
            "error_count": _total_error_count(errors),
            "error_buffers": sorted(errors.keys()),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "mode": mode.name,
            "debug_mode": mode.debug_mode,
            "status": "fail",
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }
    finally:
        # Explicitly reset singleton-backed device state between runs so each
        # mode/case probe is independent.
        try:
            AIEDeviceManager().reset()
        except Exception:  # noqa: BLE001
            pass
        del ctx


def _find_bottleneck(mode_results: list[dict]) -> tuple[str, float] | None:
    stage_results = [
        r
        for r in mode_results
        if r.get("status") == "pass"
        and any(
            spec.name == r["mode"] and spec.contributes_to_bottleneck
            for spec in DEBUG_MODE_SPECS
        )
    ]
    if not stage_results:
        return None
    worst = max(stage_results, key=lambda r: r["latency_us"])
    return (worst["mode"], worst["latency_us"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run encoder_pipeline across all debug modes for each test case and "
            "report bottleneck stages."
        )
    )
    parser.add_argument(
        "--design",
        choices=("ddr", "memtile"),
        default="ddr",
        help="LN1 staging design to use.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=DEFAULT_TEST_WARMUP_ITERS,
        help=f"Warmup iterations per mode (default: {DEFAULT_TEST_WARMUP_ITERS}).",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=DEFAULT_TEST_TIMED_ITERS,
        help=f"Timed iterations per mode (default: {DEFAULT_TEST_TIMED_ITERS}).",
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Use extensive test parameter set from test.py.",
    )
    parser.add_argument(
        "--case-index",
        type=str,
        default=None,
        help="Comma-separated case indices from generated params (example: 0,2,4).",
    )
    parser.add_argument(
        "--clean-build",
        action="store_true",
        help="Delete ./build before running.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write detailed JSON results.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop immediately when any mode run fails.",
    )
    args = parser.parse_args()

    if args.warmup_iters < 0 or args.timed_iters <= 0:
        raise ValueError("--warmup-iters must be >= 0 and --timed-iters must be > 0")

    if args.clean_build:
        shutil.rmtree(Path.cwd() / "build", ignore_errors=True)

    cases, names = generate_test_params(extensive=args.extensive)
    selected_idxs = _parse_case_filter(args.case_index, len(cases))

    summary: dict[str, object] = {
        "design": args.design,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "cases": [],
    }
    any_failures = False

    for case_idx in selected_idxs:
        case = cases[case_idx]
        case_name = names[case_idx] if case_idx < len(names) else _case_name(*case)
        print(f"\n=== case[{case_idx}] {case_name} ===")
        mode_results = []
        for spec in DEBUG_MODE_SPECS:
            print(f"  running mode={spec.name} debug={spec.debug_mode} ...", flush=True)
            result = _run_single_mode(
                case,
                spec,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                ln1_staging_design=args.design,
            )
            mode_results.append(result)
            if result["status"] == "pass":
                print(
                    "    pass "
                    f"latency_us={result['latency_us']:.2f} "
                    f"bandwidth_gbps={result['bandwidth_gbps']:.4f} "
                    f"errors={result['error_count']}"
                )
            else:
                any_failures = True
                print(f"    fail {result['exception']}")
                if args.stop_on_fail:
                    break

        bottleneck = _find_bottleneck(mode_results)
        if bottleneck is None:
            print("  bottleneck_stage: unavailable (no successful stage-mode runs)")
        else:
            print(f"  bottleneck_stage: {bottleneck[0]} ({bottleneck[1]:.2f} us)")

        summary["cases"].append(
            {
                "case_index": case_idx,
                "case_name": case_name,
                "params": case,
                "modes": mode_results,
                "bottleneck_stage": (
                    None
                    if bottleneck is None
                    else {"stage": bottleneck[0], "latency_us": bottleneck[1]}
                ),
            }
        )
        if args.stop_on_fail and any_failures:
            break

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote detailed results to {args.output_json}")

    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
