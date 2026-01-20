#!/bin/bash
# This script installs the KubeRay operator on Kubernetes
# Required for running Ray clusters with vLLM

set -e

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

echo "======================================"
echo "KubeRay Operator Installation"
echo "======================================"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found. Please install Kubernetes first."
    echo "Run: ./scripts/install-k8s-cluster.sh"
    exit 1
fi

# Check cluster connection
print_step "Checking Kubernetes connection..."
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    exit 1
fi
kubectl cluster-info
echo ""

# Check if KubeRay operator is already installed
print_step "Checking for existing KubeRay installation..."
if kubectl get deployment kuberay-operator -n kuberay-system &> /dev/null 2>&1; then
    print_warning "KubeRay operator is already installed"
    echo ""
    read -p "Do you want to reinstall? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation"
        exit 0
    fi
    print_step "Removing existing KubeRay operator..."
    kubectl delete namespace kuberay-system --ignore-not-found=true
fi

# Install KubeRay operator
print_step "Installing KubeRay operator (v1.2.1)..."

# Method 1: Using the production-stack utility (if available)
INSTALL_DIR="${TMPDIR:-/tmp}/k8s-install-$$"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -d "production-stack" ]; then
    git clone -b vllm-stack-0.1.8 https://github.com/vllm-project/production-stack.git
fi

cd production-stack/tutorials

# Run the KubeRay installation script
if [ -f "00-b-install-kuberay-operator.sh" ]; then
    print_step "Using production-stack KubeRay installer..."
    bash 00-b-install-kuberay-operator.sh
else
    # Method 2: Direct installation via Helm
    print_step "Installing KubeRay via Helm..."

    # Add KubeRay Helm repository
    helm repo add kuberay https://ray-project.github.io/kuberay-helm
    helm repo update

    # Install KubeRay operator
    helm install kuberay-operator kuberay/kuberay-operator \
        --namespace kuberay-system \
        --create-namespace \
        --version 1.2.1 \
        --set image.repository=rayproject/kuberay-operator \
        --set image.tag=v1.2.1
fi

# Wait for operator to be ready
print_step "Waiting for KubeRay operator to be ready..."
kubectl wait --for=condition=available --timeout=120s \
    deployment/kuberay-operator -n kuberay-system

# Cleanup
cd /
rm -rf "$INSTALL_DIR"

echo ""
print_step "Verifying KubeRay operator installation..."
kubectl get pods -n kuberay-system
kubectl get crd | grep ray.io || print_warning "Ray CRDs not found yet"

echo ""
echo "======================================"
echo "KubeRay Operator Installed!"
echo "======================================"
echo ""
echo "KubeRay operator is running in namespace: kuberay-system"
echo ""
echo "Next step: Deploy vLLM with Ray cluster"
echo "  ./scripts/deploy-vllm-ray.sh"
echo ""
