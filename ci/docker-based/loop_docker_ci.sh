#!/bin/bash

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x 

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
SCOPE="repo"

get_registration_token() {
  if [ "${SCOPE}" = "repo" ]; then
    URL="https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/runners/registration-token"
  elif [ "${SCOPE}" = "org" ]; then
    URL="https://api.github.com/orgs/${GITHUB_OWNER}/actions/runners/registration-token"
  else
    echo "Unknown scope: ${SCOPE}"
    exit 1
  fi

  TOKEN=$(curl -sX POST -H "Authorization: token ${GITHUB_PAT}" \
    -H "Accept: application/vnd.github+json" "${URL}" \
    | jq -r .token)

  if [ "${TOKEN}" = "null" ]; then
    echo "Failed to get runner registration token"
    exit 1
  fi
  echo "${TOKEN}"
}

while true; do
    DATE=$(printf '%(%Y_%m_%d_%H_%M_%S)T')
    NAME="ci-run-${DATE}"
    if ! TOKEN="$(get_registration_token)"; then
        echo "Failed to get runner registration token" >&2
        exit 1
    fi
    echo "Got token for runner registration: ${TOKEN}"
    echo "Running new container: ${NAME}"
    docker run \
        --rm \
        --name "${NAME}" \
        --device-cgroup-rule 'c 261:* rmw' \
        --ulimit memlock=-1:-1 \
        -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro \
        -v /dev/accel/accel0:/dev/accel/accel0 \
        -v /srv:/srv:ro \
        -e GITHUB_RUNNER_TOKEN="${TOKEN}" \
        -e GITHUB_OWNER="${GITHUB_OWNER}" \
        -e GITHUB_REPO="${GITHUB_REPO}" \
        ${IMAGE_NAME}
    echo "Container ${NAME} exited. Restarting in 2 seconds..."
    sleep 2
done
