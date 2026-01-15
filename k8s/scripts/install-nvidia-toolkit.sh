#!/bin/bash
# This script installs the Nvidia Container Toolkit on Ubuntu/Debian systems
# Run with sudo for seamless installation

set -e

echo "=== Installing Nvidia Container Toolkit ==="

# Update package list and install prerequisites
echo "[1/5] Updating package list and installing prerequisites..."
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    curl \
    gnupg2

# Add Nvidia GPG key
echo "[2/5] Adding Nvidia GPG key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add Nvidia repository
echo "[3/5] Adding Nvidia repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
echo "[4/5] Updating package list..."
sudo apt-get update -y

# Install Nvidia Container Toolkit
echo "[5/5] Installing Nvidia Container Toolkit..."
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-cuda-toolkit

# Configure containerd runtime
echo "=== Configuring containerd runtime ==="
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd

echo "=== Nvidia Container Toolkit installation complete ==="
echo "Please verify GPU availability with: nvidia-smi"
