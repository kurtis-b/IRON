#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ..artifact_integrity import ARTIFACT_PROFILES
from .. import unattended_reboot


def _default_env_check_script() -> Path:
    return Path(__file__).resolve().parents[5] / "scripts" / "check_transformer_layer_env.sh"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the transformer-layer reviewer quickcheck by validating the local "
            "environment and delegating to the existing execution smoke flow."
        )
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--source-results-root", type=Path, default=None)
    parser.add_argument("--run-user", default=None)
    parser.add_argument("--reboot-command", default="true")
    parser.add_argument("--artifact-profile", choices=ARTIFACT_PROFILES, default="p0")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--env-check-script", type=Path, default=_default_env_check_script())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_env_check:
        subprocess.run([str(args.env_check_script.expanduser())], check=True)

    delegated_argv: list[str] = ["execution-smoke-test"]
    if args.run_id is not None:
        delegated_argv.extend(["--run-id", str(args.run_id)])
    if args.state is not None:
        delegated_argv.extend(["--state", str(args.state.expanduser())])
    if args.results_root is not None:
        delegated_argv.extend(["--results-root", str(args.results_root.expanduser())])
    if args.source_results_root is not None:
        delegated_argv.extend(
            ["--source-results-root", str(args.source_results_root.expanduser())]
        )
    if args.run_user is not None:
        delegated_argv.extend(["--run-user", str(args.run_user)])
    delegated_argv.extend(
        [
            "--reboot-command",
            str(args.reboot_command),
            "--artifact-profile",
            str(args.artifact_profile),
            "--log-level",
            str(args.log_level),
        ]
    )
    return unattended_reboot.main(delegated_argv)


if __name__ == "__main__":
    raise SystemExit(main())
