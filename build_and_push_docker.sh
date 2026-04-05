#!/bin/bash
set -e

# Settings
# NB: run this script from this branch with: chmod +x ./build_and_push_docker.sh && docker login -u aiinfos && ./build_and_push_docker.sh 
# it will default to current branch name: v0.19.1rc0.x or from another branch: e.g. "./build_and_push_docker.sh v0.17.1rc0.x"
IMAGE_NAME="aiinfos/vllm-gfx906-mobydick"
IMAGE_TAG="${1:-v0.19.1rc0.x}"

echo "Creating Dockerfile..."

cat << 'EOF' > Dockerfile
FROM rocm/pytorch:rocm6.3.4_ubuntu24.04_py3.12_pytorch_release_2.4.0

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install required build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    sudo \
    build-essential \
    ffmpeg \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# The base image has an incompatible version of PyTorch, TorchVision and Triton for gfx906.
# We uninstall them before rebuilding our own for gfx906.
RUN pip uninstall -y torch torchvision torchaudio triton

# Global environment variables
ENV USE_ROCM=1
ENV PYTORCH_ROCM_ARCH=gfx906

WORKDIR /workspace

# Install PyTorch 2.10.0 from source targeting gfx906
RUN git clone --branch v2.10.0 --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    pip install -r requirements.txt && \
    pip install mkl-static mkl-include && \
    python tools/amd_build/build_amd.py && \
    MAX_JOBS=$(nproc) pip wheel --no-build-isolation -v -w dist -e . && \
    pip install ./dist/torch*.whl && \
    cd .. && \
    rm -rf pytorch

# Install Torchvision 0.25.0
RUN git clone --branch v0.25.0 https://github.com/pytorch/vision.git && \
    cd vision && \
    FORCE_CUDA=1 python setup.py install && \
    cd .. && \
    rm -rf vision

# Install Torchaudio 2.10.0
RUN git clone --branch v2.10.0 https://github.com/pytorch/audio.git && \
    cd audio && \
    python setup.py install && \
    cd .. && \
    rm -rf audio

# Install Triton-gfx906 3.6.0
RUN git clone --branch v3.6.0+gfx906 https://github.com/ai-infos/triton-gfx906.git && \
    cd triton-gfx906 && \
    pip install -r python/requirements.txt && \
    TRITON_CODEGEN_BACKENDS="amd" pip wheel --no-build-isolation -w dist . && \
    pip install ./dist/triton-*.whl && \
    cd .. && \
    rm -rf triton-gfx906

# Install Flash-Attention for gfx906
RUN git clone https://github.com/ai-infos/flash-attention-gfx906.git && \
    cd flash-attention-gfx906 && \
    FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install && \
    cd .. && \
    rm -rf flash-attention-gfx906

# Install vLLM-gfx906-mobydick
RUN git clone https://github.com/ai-infos/vllm-gfx906-mobydick.git && \
    cd vllm-gfx906-mobydick && \
    pip install 'cmake>=3.26.1,<4' 'packaging>=24.2' 'setuptools>=77.0.3,<80.0.0' 'setuptools-scm>=8' 'jinja2>=3.1.6' 'amdsmi>=6.3,<6.4' 'timm>=1.0.17' && \
    pip install -r requirements/rocm.txt && \
    pip wheel --no-build-isolation -v -w dist . && \
    pip install ./dist/vllm-*.whl && \
    pip install transformers==5.5.0 "numpy<2"

# Ensure user remains in the repo directory during interactive shells
WORKDIR /workspace/vllm-gfx906-mobydick

# Set recommended runtime environment variables
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV VLLM_LOGGING_LEVEL=INFO

CMD ["/bin/bash"]
EOF

echo "Dockerfile has been created."

echo "Building the Docker image: ${IMAGE_NAME}:${IMAGE_TAG} ..."
# Provide robust error tracking and multi-platform compatibility
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest .

echo "We need to authenticate with Docker Hub before pushing."
echo "If you haven't logged in yet, it will prompt for your Docker Hub credentials."
docker login -u aiinfos

echo "Pushing the Docker image to Docker Hub at ${IMAGE_NAME}:${IMAGE_TAG} and ${IMAGE_NAME}:latest ..."
docker push ${IMAGE_NAME}:${IMAGE_TAG}
docker push ${IMAGE_NAME}:latest

echo "Process completed successfully! The images are pushed to Docker Hub."
