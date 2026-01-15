#!/bin/bash
# This script sets up a local minikube cluster with GPU support
# WARNING: This will delete any existing minikube cluster

set -e

INSTALL_DIR="${TMPDIR:-/tmp}/k8s-install-$$"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "=== Setting up Minikube cluster with GPU support ==="

# Clone vLLM production stack repository if not already done
if [ ! -d "production-stack" ]; then
    git clone -b vllm-stack-0.1.8 https://github.com/vllm-project/production-stack.git
fi

cd production-stack/utils

# Remove existing minikube installation
echo "[1/3] Removing existing minikube installation..."
sudo minikube delete || true
sudo apt remove minikube -y || true

# Install minikube cluster
echo "[2/3] Installing minikube cluster..."
bash install-minikube-cluster.sh

# Verify GPU availability
echo "[3/3] Verifying GPU availability in cluster..."
minikube status

# Check if GPU is detected
echo "Checking GPU availability in nodes..."
kubectl describe nodes | grep -i gpu || echo "Warning: GPU not detected in nodes"

echo "=== Minikube cluster setup complete ==="
echo "You can test GPU with:"
echo "  kubectl run gpu-test --image=nvidia/cuda:12.2.0-runtime-ubuntu22.04 --restart=Never -- nvidia-smi"
echo "  kubectl logs gpu-test"

# Cleanup
cd /
rm -rf "$INSTALL_DIR"
