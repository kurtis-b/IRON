#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from .artifact_integrity import ARTIFACT_PROFILES, validate_results_root
from .results_manifest import write_results_root_manifest

LOGGER = logging.getLogger(__name__)

PUBLISHABLE_SUBDIRS: tuple[str, ...] = (
    "analysis",
    "block",
    "end_to_end",
    "memory_tile_staging",
    "host_comparison",
    "memcpy_bandwidth",
    "resource_usage",
    "roofline",
)


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_target_results_root() -> Path:
    return app_root() / "results"


def _copy_publishable_tree(source_root: Path, target_root: Path) -> None:
    for subdir in PUBLISHABLE_SUBDIRS:
        source_path = source_root / subdir
        if not source_path.exists():
            continue
        target_path = target_root / subdir
        shutil.copytree(source_path, target_path)


def publish_results_root(
    *,
    source_root: str | Path,
    target_root: str | Path,
    force: bool = False,
    artifact_profile: str = "p0",
) -> Path:
    source = Path(source_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    if source == target:
        raise ValueError("source_root and target_root must be different paths")

    write_results_root_manifest(source)
    validate_results_root(source, artifact_profile=artifact_profile)

    if target.exists():
        if not force:
            raise FileExistsError(
                f"Target results root already exists: {target}. Pass --force to replace it."
            )
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    _copy_publishable_tree(source, target)
    write_results_root_manifest(
        target,
        source_results_root=source,
        evaluation_profile=("paper" if artifact_profile == "paper" else None),
    )
    validate_results_root(target, artifact_profile=artifact_profile)
    LOGGER.info("Published validated results root from %s to %s", source, target)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a validated transformer-layer results root into the canonical "
            "paper-facing results/ tree."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--target-root", type=Path, default=default_target_results_root()
    )
    parser.add_argument("--artifact-profile", choices=ARTIFACT_PROFILES, default="p0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    publish_results_root(
        source_root=args.source_root,
        target_root=args.target_root,
        force=bool(args.force),
        artifact_profile=str(args.artifact_profile),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
