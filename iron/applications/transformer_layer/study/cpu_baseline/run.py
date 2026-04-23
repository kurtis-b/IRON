#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from ..host_comparison.run import main as host_comparison_main


def main(argv: list[str] | None = None) -> int:
    delegated_argv = list(argv or [])
    if "--host-backends" not in delegated_argv:
        delegated_argv.extend(["--host-backends", "cpu"])
    return host_comparison_main(delegated_argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
