#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path


def iter_powercap_zone_dirs(powercap_root):
    root = Path(powercap_root)
    for entry in sorted(root.iterdir()):
        if (entry / "name").exists():
            yield entry


def discover_energy_paths(powercap_root=Path("/sys/class/powercap")):
    energy_paths = []
    for zone_dir in sorted(iter_powercap_zone_dirs(Path(powercap_root))):
        energy_path = zone_dir / "energy_uj"
        max_energy_range_path = zone_dir / "max_energy_range_uj"
        if energy_path.exists() and max_energy_range_path.exists():
            energy_paths.append(energy_path)
    return energy_paths


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Read Linux powercap RAPL energy counters and emit JSON."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the target energy counters are readable without emitting JSON.",
    )
    parser.add_argument("paths", nargs="*")
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)
    paths = (
        [Path(raw_path) for raw_path in args.paths]
        if args.paths
        else discover_energy_paths()
    )
    payload = {}
    for path in paths:
        payload[str(path)] = Path(path).read_text(encoding="utf-8").strip()
    if not args.check:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
