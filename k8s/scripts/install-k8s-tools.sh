#!/bin/bash
# This script installs k8s tools: kubectl, helm, and minikube
# Based on vLLM production stack utilities

set -e

INSTALL_DIR="${TMPDIR:-/tmp}/k8s-install-$$"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "=== Installing k8s tools for vLLM ==="

# Clone vLLM production stack repository
echo "[1/3] Cloning vLLM production stack..."
git clone -b vllm-stack-0.1.8 https://github.com/vllm-project/production-stack.git
cd production-stack/utils

# Install kubectl
echo "[2/3] Installing kubectl..."
bash install-kubectl.sh

# Install helm
echo "[3/3] Installing helm..."
bash install-helm.sh

echo "=== kubectl and helm installation complete ==="

# Cleanup
cd /
rm -rf "$INSTALL_DIR"

echo "You can now install minikube using install-minikube-cluster.sh"
