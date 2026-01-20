#!/bin/bash
# This script sets up a multi-node Kubernetes cluster for vLLM with KubeRay
# Supports both single-node (k3s) and multi-node (kubeadm) setups

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/utils.sh" 2>/dev/null || true

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Configuration
K8S_TYPE="${K8S_TYPE:-k3s}"  # Options: k3s (single-node), kubeadm (multi-node)
CLUSTER_NAME="${CLUSTER_NAME:-vllm-k8s}"

echo "======================================"
echo "Multi-Node Kubernetes Setup for vLLM"
echo "======================================"
echo ""
print_step "Cluster type: $K8S_TYPE"
echo ""

# Detect GPU
print_step "Detecting NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi
echo ""

# Install k3s for single-node setup
if [ "$K8S_TYPE" = "k3s" ]; then
    print_step "Installing k3s (single-node cluster)..."

    # Check if k3s is already installed
    if command -v k3s &> /dev/null; then
        print_warning "k3s is already installed"
        k3s --version
    else
        # Install k3s with GPU support
        curl -sfL https://get.k3s.io | sh -s - \
            --write-kubeconfig-mode 644 \
            --disable traefik \
            --disable servicelb \
            --disable cloud-controller-manager \
            --disable local-storage \
            --kube-proxy-arg=proxy-mode=ipvs

        # Wait for k3s to be ready
        echo "Waiting for k3s to start..."
        sleep 10
    fi

    # Configure kubectl
    mkdir -p ~/.kube
    sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config 2>/dev/null || true
    sudo chown $USER:$USER ~/.kube/config 2>/dev/null || true

    # Verify cluster
    print_step "Verifying k3s cluster..."
    kubectl get nodes
    kubectl get pods -A

    echo ""
    echo "=== k3s installation complete ==="

# Install kubeadm for multi-node setup (following production-stack tutorial)
elif [ "$K8S_TYPE" = "kubeadm" ]; then
    print_step "Installing multi-node cluster using production-stack utilities..."

    INSTALL_DIR="${TMPDIR:-/tmp}/k8s-install-$$"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # Clone vLLM production stack repository
    print_step "Cloning vLLM production stack (multi-node k8s tutorial)..."
    if [ ! -d "production-stack" ]; then
        git clone -b vllm-stack-0.1.8 https://github.com/vllm-project/production-stack.git
    fi

    cd production-stack/tutorials

    # Run the multi-node k8s installation script
    print_step "Running multi-node k8s installation..."
    if [ -f "00-a-install-multinode-kubernetes-env.sh" ]; then
        bash 00-a-install-multinode-kubernetes-env.sh
    else
        echo "Error: Multi-node installation script not found"
        echo "Please follow the manual setup at:"
        echo "https://github.com/vllm-project/production-stack/blob/main/tutorials/00-a-install-multinode-kubernetes-env.md"
        exit 1
    fi

    # Cleanup
    cd /
    rm -rf "$INSTALL_DIR"

    echo ""
    echo "=== Multi-node k8s installation complete ==="

else
    echo "Error: Invalid K8S_TYPE. Use 'k3s' or 'kubeadm'"
    exit 1
fi

# Verify GPU support in cluster
echo ""
print_step "Verifying GPU support in Kubernetes..."
kubectl describe nodes 2>/dev/null | grep -i gpu || print_warning "GPU not detected in nodes yet (will be available after nvidia-device-plugin installation)"

echo ""
echo "======================================"
echo "Kubernetes Cluster Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Install KubeRay operator: ./scripts/install-kuberay-operator.sh"
echo "  2. Deploy vLLM with Ray:     ./scripts/deploy-vllm-ray.sh"
echo ""
