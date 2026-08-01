#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

IMAGE_NAME="iron-public-dev-github-runner"
GITHUB_OWNER="amd"
GITHUB_REPO="IRON"

load_github_pat() {
  if [ -n "${GITHUB_PAT:-}" ]; then
    printf '%s\n' "${GITHUB_PAT}"
    return 0
  fi

  if [ -z "${GITHUB_PAT_FILE:-}" ]; then
    echo "Set GITHUB_PAT or GITHUB_PAT_FILE before running this script" >&2
    exit 1
  fi

  if [ ! -f "${GITHUB_PAT_FILE}" ]; then
    echo "GITHUB_PAT_FILE does not exist: ${GITHUB_PAT_FILE}" >&2
    exit 1
  fi

  cat "${GITHUB_PAT_FILE}"
}

GITHUB_PAT="$(load_github_pat)"

DATE=$(printf '%(%Y_%m_%d_%H_%M_%S)T')
NAME="ci-run-${DATE}"
docker run \
    --rm \
    --name "${NAME}" \
    --device-cgroup-rule 'c 261:* rmw' \
    --ulimit memlock=-1:-1 \
    -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro \
    -v /dev/accel/accel0:/dev/accel/accel0 \
    -v /srv:/srv:ro \
    -e GITHUB_PAT="${GITHUB_PAT}" \
    -e GITHUB_OWNER="${GITHUB_OWNER}" \
    -e GITHUB_REPO="${GITHUB_REPO}" \
    -it \
    ${IMAGE_NAME} \
    bash
