# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dockerfile using uv environment.

ARG TARGETPLATFORM
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

FROM ${BASE_IMAGE}

# Set the DEBIAN_FRONTEND environment variable to avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        tree \
        wget

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

WORKDIR /workspace

# Install the project's dependencies using the lockfile and settings.
# Python 3.13 matches upstream cosmos-framework's .python-version. The cu128
# dependency group ships cp313-only wheels (flash-attn 2.7.4.post1+cu128.torch210
# from nvidia-cosmos.github.io/cosmos-dependencies), so 3.10 fails at uv sync.
ARG PYTHON_VERSION=3.13
ARG CUDA_NAME=cu128
ENV CUDA_NAME=${CUDA_NAME}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages,target=packages \
    echo "${PYTHON_VERSION}" > .python-version && \
    uv sync --locked --no-install-project --group=${CUDA_NAME}

# Copy the code into the container if in standalone mode. Otherwise, just install the dependencies at runtime.
# We mount the source code to /tmp and copy it to /workspace if in standalone mode.
# Direct uv sync (not `just install ${CUDA_NAME}`) — cosmos-framework's install
# recipe forwards its args as `--reinstall <arg>` to uv, which uv rejects. This
# call installs the project itself plus all extras + the cuda group (matches
# upstream cosmos-framework's Dockerfile pattern, minus vllm).
ARG STANDALONE
RUN --mount=type=bind,source=.,target=/tmp/workspace \
   if [ "$STANDALONE" = "true" ] ; then cp -r /tmp/workspace/* /workspace && echo "${PYTHON_VERSION}" > /workspace/.python-version && uv sync --locked --no-editable --all-extras --group=${CUDA_NAME} && rm -rf /workspace/.git ; else echo "Run uv sync to install all the dependencies at runtime" ; fi

# Place executables in the environment at the front of the path
ENV PATH="/workspace/.venv/bin:$PATH"

# Install Ray for Lilypad workload orchestration. cosmos-framework's uv.lock
# already brings ray==2.46.0, but Applied's Lilypad protocol needs 2.50.x.
# Applied's index only publishes cp310-cp312 wheels for ray 2.50.1.7, so we
# also allow public pypi via --index-strategy unsafe-best-match. Public pypi
# has cp313 wheels for 2.50.x, and we still prefer Applied's index when it
# has a compatible version.
RUN uv pip install "ray[default]==2.50.1.7" \
    --extra-index-url https://ursa.pypi.applied.dev/simple \
    --index-strategy unsafe-best-match

# click 8.3.x _Sentinel deepcopy bug is Python-3.10-only; on 3.13 no pin needed.

# Install Lilypad SDK for cross-region boto caching utilities.
RUN uv pip install "lilypad-py==2.27.0" \
    --extra-index-url https://ursa.pypi.applied.dev/simple \
    --index-strategy unsafe-best-match

# Skip uv sync at container start — all deps are already installed above (STANDALONE=true bakes
# everything in at build time). This prevents uv sync from downgrading click and breaking Ray.
ENV SKIP_UV_SYNC=true

ENTRYPOINT ["/workspace/bin/entrypoint.sh"]

CMD ["/bin/bash"]
