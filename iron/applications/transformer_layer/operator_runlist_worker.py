#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from iron.applications.transformer_layer.src.pipeline.operator_runlist_worker import (
    benchmark_operator_runlist_request,
    parity_operator_runlist_request,
    run_worker,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run runlist benchmark in an isolated child process."
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        request_path = Path(args.request_json)
        response_path = Path(args.response_json)
        request = json.loads(request_path.read_text())
        row = run_worker(request)
        response_path.write_text(json.dumps(row))
        os._exit(0)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)


__all__ = [
    "benchmark_operator_runlist_request",
    "parity_operator_runlist_request",
]


if __name__ == "__main__":
    main()
