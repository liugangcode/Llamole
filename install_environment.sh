#!/bin/bash

# Set non-interactive frontend
export DEBIAN_FRONTEND=noninteractive

# Activate the llama_factory environment
# Uncomment the following line if you need to create the environment
# conda create --name llamole python=3.11 -y
# conda activate llamole

# Function to get the current conda environment name
get_current_conda_env() {
    echo "current: $CONDA_DEFAULT_ENV"
}

# Get the current conda environment
current_env=$(basename "$CONDA_PREFIX")

# Check if the current environment is 'llamole'
if [ "$current_env" != "llama_factory" ] && [ "$current_env" != "llamole_test" ]; then
    echo "current: $CONDA_DEFAULT_ENV"
    echo "Current conda environment is neither 'llama_factory' nor 'llamole'."
    echo "Please activate one of these environments before running this script."
    echo "You can activate an environment using one of these commands:"
    echo "conda activate llama_factory"
    echo "conda activate llamole"
    exit 1
fi

echo "Running in conda environment: $current_env"

# "pandas>=2.0.0" \
# Install packages using pip
pip install --no-cache-dir \
    pyarrow \
    "pandas>=1.5.3" \
    "rdkit==2023.9.6" \
    pyyaml \
    ipykernel \
    packaging \
    gdown \
    "fcd_torch==1.0.7" \
    "omegaconf==2.3.0" \
    "imageio==2.26.0" \
    wandb \
    pandarallel \
    scipy \
    einops \
    sentencepiece \
    tiktoken \
    protobuf \
    uvicorn \
    pydantic \
    fastapi \
    sse-starlette \
    "matplotlib>=3.7.0" \
    fire \
    "numpy<2.0.0" \
    gradio

pip install --no-cache-dir hydra-core --upgrade

# Install PyTorch
pip install --no-cache-dir torch

# Install PyTorch Geometric and related packages
pip install --no-cache-dir torch_geometric

# for retro reaction
pip install rdchiral
pip install nltk

# Install transformers and related packages
pip install --no-cache-dir \
    "transformers>=4.41.3" \
    "datasets>=2.16.0" \
    "accelerate>=0.30.1" \
    "peft>=0.11.1" \
    "trl>=0.8.6" \
    "gradio>=4.0.0"

# Install mini-moses from GitHub
pip install --no-cache-dir git+https://github.com/igor-krawczuk/mini-moses

echo "Installation complete!"