#!/usr/bin/env bash
set -euo pipefail

# Minimal bootstrapper that installs the pinned toolchain using uv.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.12.11"
UV_INSTALLER_URL="https://astral.sh/uv/install.sh"
CHECKPOINT_URL="https://huggingface.co/datasets/medarc/path-fm-dinov3/resolve/main/dinov3_vith16plus_saved_teacher.pth"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing from ${UV_INSTALLER_URL}..."
  curl -LsSf "${UV_INSTALLER_URL}" | sh
  hash -r
fi

cd "${PROJECT_ROOT}"

if ! uv python list --only-installed | grep -q "${PYTHON_VERSION}"; then
  echo "Installing Python ${PYTHON_VERSION} via uv..."
  uv python install "${PYTHON_VERSION}"
fi

uv venv
uv pip install -e . --torch-backend=auto -p .venv/bin/python

CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints"
CHECKPOINT_FILE="${CHECKPOINTS_DIR}/dinov3_vith16plus_saved_teacher.pth"

mkdir -p "${CHECKPOINTS_DIR}"
if [ ! -f "${CHECKPOINT_FILE}" ]; then
  echo "Downloading teacher checkpoint to ${CHECKPOINT_FILE}..."
  wget -O "${CHECKPOINT_FILE}" "${CHECKPOINT_URL}"
else
  echo "Checkpoint already exists at ${CHECKPOINT_FILE}; skipping download."
fi

echo "Environment ready. Activate it with 'source .venv/bin/activate' and log into wandb before training."
