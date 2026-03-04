#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from operators.encoder_pipeline.mapping_validation import (
        find_ffn_layout,
        validate_mlir_mapping_constraints,
        validate_layout_constraints,
    )
except ModuleNotFoundError:
    from mapping_validation import (
        find_ffn_layout,
        validate_mlir_mapping_constraints,
        validate_layout_constraints,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validate_mapping",
        description="Validate encoder_pipeline core placement constraints.",
    )
    parser.add_argument(
        "--parallel-heads",
        type=int,
        default=None,
        help="Parallel MHA heads (required for pre-build layout validation).",
    )
    parser.add_argument(
        "--mlir",
        type=Path,
        default=None,
        help="Validate realized mapping from generated MLIR instead of planned layout.",
    )
    parser.add_argument(
        "--mha-output-row",
        type=int,
        default=5,
        help=(
            "Compute row of the MHA producer feeding LN1 "
            "(default uses current o-proj row: 5)."
        ),
    )
    parser.add_argument(
        "--max-non-neighbor-ln2-streams",
        type=int,
        default=2,
        help="Maximum non-neighbor down->LN2 streams allowed.",
    )
    parser.add_argument(
        "--require-mha-ln-neighbor",
        action="store_true",
        help=("Require MHA output tile and LN1 to be direct neighbors."),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of human-readable text.",
    )
    args = parser.parse_args()

    if args.mlir is not None:
        report = validate_mlir_mapping_constraints(
            mlir_path=args.mlir,
            max_non_neighbor_down_to_ln2=args.max_non_neighbor_ln2_streams,
            require_mha_ln_neighbor=args.require_mha_ln_neighbor,
        )
        report["valid"] = len(report["errors"]) == 0

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"mlir={report['mlir_path']}")
            print(f"mha_output_tile={report['mha_output_tile']}")
            print(f"ln1={report['ln1_tile']}")
            print(f"up={report['up_tiles']}")
            print(f"down={report['down_tiles']}")
            print(f"ln2={report['ln2_tile']}")
            print(f"up_to_down_edges={report['up_to_down_edges']}")
            print(f"reduction_edges={report['reduction_edges']}")
            print(f"direct_down_to_ln2={report['direct_down_to_ln2']}")
            print(
                "non_neighbor_direct_down_to_ln2="
                f"{report['non_neighbor_direct_down_to_ln2']}"
            )
            if report["errors"]:
                print("VALIDATION: FAILED")
                for err in report["errors"]:
                    print(f"- {err}")
            else:
                print("VALIDATION: PASSED")
        return 1 if report["errors"] else 0

    if args.parallel_heads is None:
        parser.error("`--parallel-heads` is required unless `--mlir` is provided.")

    mha_output_tile = (args.parallel_heads - 1, args.mha_output_row)
    require_mha_ln_neighbor = args.require_mha_ln_neighbor

    try:
        layout = find_ffn_layout(
            parallel_heads=args.parallel_heads,
            max_non_neighbor_down_to_ln2=args.max_non_neighbor_ln2_streams,
            mha_output_tile=mha_output_tile,
            require_mha_ln_neighbor=require_mha_ln_neighbor,
        )
    except ValueError as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "parallel_heads": args.parallel_heads,
                        "mha_output_tile": mha_output_tile,
                        "valid": False,
                        "errors": [str(exc)],
                    },
                    indent=2,
                )
            )
        else:
            print(f"parallel_heads={args.parallel_heads}")
            print(f"mha_output_tile={mha_output_tile}")
            print("VALIDATION: FAILED")
            print(f"- {exc}")
        return 1

    errors = validate_layout_constraints(
        parallel_heads=args.parallel_heads,
        layout=layout,
        max_non_neighbor_down_to_ln2=args.max_non_neighbor_ln2_streams,
        mha_output_tile=mha_output_tile,
        require_mha_ln_neighbor=require_mha_ln_neighbor,
    )

    report = {
        "parallel_heads": args.parallel_heads,
        "mha_output_tile": mha_output_tile,
        "layout": layout,
        "valid": len(errors) == 0,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"parallel_heads={args.parallel_heads}")
        print(f"mha_output_tile={mha_output_tile}")
        print(f"ln1={layout['ln1_tile']}")
        print(f"up={layout['up_tiles']}")
        print(f"down={layout['down_tiles']}")
        print(f"ln2={layout['ln2_tile']}")
        print(f"down_root={layout['down_root_tile']}")
        print(f"down_reduction_edges={layout['down_reduction_edges']}")
        print(f"down_to_ln2={layout['down_to_ln2_tiles']}")
        print(f"non_neighbor_down_to_ln2={layout['non_neighbor_down_to_ln2']}")
        if errors:
            print("VALIDATION: FAILED")
            for err in errors:
                print(f"- {err}")
        else:
            print("VALIDATION: PASSED")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
