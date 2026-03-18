#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
from pathlib import Path

from model_support import (
    HF_CONFIG_FILENAME,
    MODEL_FILENAME,
    DEFAULT_STUDY_MANIFEST,
    load_study_manifest,
    resolve_study_paths,
)

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download encoder model weights/config from Hugging Face into the local "
            "iron/applications/bert/models layout."
        )
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default=None,
        help="Comma-separated study ids to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every study entry from the manifest.",
    )
    parser.add_argument(
        "--study-manifest",
        type=str,
        default=str(DEFAULT_STUDY_MANIFEST),
        help="Path to the study manifest JSON.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Root directory for downloaded model artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the resolved download plan and exit without downloading.",
    )
    return parser.parse_args()


def parse_requested_study_ids(args):
    if args.all and args.study_id:
        raise ValueError("Use either --all or --study-id, not both")
    manifest_path, entries = load_study_manifest(args.study_manifest)
    if args.all:
        return manifest_path, [entry["study_id"] for entry in entries]
    if args.study_id:
        requested = []
        for token in args.study_id.split(","):
            token = token.strip()
            if token and token not in requested:
                requested.append(token)
        if requested:
            return manifest_path, requested
    raise ValueError("Provide --study-id or --all")


def resolve_hf_token():
    for env_var in HF_TOKEN_ENV_VARS:
        token = os.environ.get(env_var)
        if token:
            return token
    return None


def download_file(*, repo_id, filename, destination, token, force):
    if destination.exists() and not force:
        return destination, False

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, destination)
    return destination, True


def main():
    args = parse_args()
    manifest_path, study_ids = parse_requested_study_ids(args)
    token = resolve_hf_token()

    for study_id in study_ids:
        resolved = resolve_study_paths(
            study_id,
            manifest_path=manifest_path,
            models_root=args.models_root,
        )
        model_dir = Path(resolved["model_dir"])
        weights_path = Path(resolved["weights_file_path"])
        config_path = Path(resolved["hf_config_path"])
        print(
            f"study_id={study_id} repo={resolved['hf_model_id']} family={resolved['family']} "
            f"model_dir={model_dir}",
            flush=True,
        )
        print(f"  weights -> {weights_path}", flush=True)
        print(f"  config  -> {config_path}", flush=True)
        if args.print_plan:
            continue

        for filename, destination in (
            (MODEL_FILENAME, weights_path),
            (HF_CONFIG_FILENAME, config_path),
        ):
            _, downloaded = download_file(
                repo_id=resolved["hf_model_id"],
                filename=filename,
                destination=destination,
                token=token,
                force=args.force,
            )
            action = "downloaded" if downloaded else "cached"
            print(f"  {action}: {destination}", flush=True)


if __name__ == "__main__":
    main()
