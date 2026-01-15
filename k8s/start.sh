#!/bin/bash
# Main deployment script for vLLM on k8s
# This script orchestrates the entire setup process

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Parse command line arguments
SKIP_NVIDIA=false
SKIP_K8S=false
SKIP_DEPLOY=false
ONLY_DEPLOY=false
ONLY_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-nvidia)
            SKIP_NVIDIA=true
            shift
            ;;
        --skip-k8s)
            SKIP_K8S=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --only-deploy)
            ONLY_DEPLOY=true
            shift
            ;;
        --only-test)
            ONLY_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-nvidia    Skip Nvidia toolkit installation"
            echo "  --skip-k8s       Skip k8s tools installation"
            echo "  --skip-deploy    Skip vLLM deployment"
            echo "  --only-deploy    Only run vLLM deployment (assume k8s is ready)"
            echo "  --only-test      Only run tests (assume vLLM is deployed)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "======================================"
echo "vLLM on Kubernetes Deployment Script"
echo "======================================"
echo ""

# Test mode
if [ "$ONLY_TEST" = true ]; then
    print_step "Running vLLM tests only..."
    bash ./scripts/test-vllm.sh
    exit 0
fi

# Step 1: Install Nvidia toolkit
if [ "$ONLY_DEPLOY" = false ] && [ "$SKIP_NVIDIA" = false ]; then
    print_step "Step 1: Installing Nvidia Container Toolkit..."
    if [ -f "./scripts/install-nvidia-toolkit.sh" ]; then
        bash ./scripts/install-nvidia-toolkit.sh
    else
        print_error "Nvidia toolkit script not found"
        exit 1
    fi
    echo ""
elif [ "$SKIP_NVIDIA" = true ]; then
    print_warning "Skipping Nvidia toolkit installation"
fi

# Step 2: Install k8s tools
if [ "$ONLY_DEPLOY" = false ] && [ "$SKIP_K8S" = false ]; then
    print_step "Step 2: Installing k8s tools (kubectl, helm)..."
    if [ -f "./scripts/install-k8s-tools.sh" ]; then
        bash ./scripts/install-k8s-tools.sh
    else
        print_error "k8s tools script not found"
        exit 1
    fi
    echo ""
elif [ "$SKIP_K8S" = true ]; then
    print_warning "Skipping k8s tools installation"
fi

# Step 3: Setup minikube
if [ "$ONLY_DEPLOY" = false ] && [ "$SKIP_K8S" = false ]; then
    print_step "Step 3: Setting up Minikube with GPU support..."
    if [ -f "./scripts/install-minikube.sh" ]; then
        bash ./scripts/install-minikube.sh
    else
        print_error "Minikube script not found"
        exit 1
    fi
    echo ""
elif [ "$SKIP_K8S" = true ]; then
    print_warning "Skipping Minikube setup"
fi

# Step 4: Deploy vLLM
if [ "$SKIP_DEPLOY" = false ]; then
    print_step "Step 4: Deploying vLLM on Kubernetes..."
    if [ -f "./scripts/deploy-vllm.sh" ]; then
        bash ./scripts/deploy-vllm.sh
    else
        print_error "vLLM deployment script not found"
        exit 1
    fi
    echo ""
else
    print_warning "Skipping vLLM deployment"
fi

# Step 5: Run tests
if [ "$SKIP_DEPLOY" = false ]; then
    print_step "Step 5: Running vLLM API tests..."
    if [ -f "./scripts/test-vllm.sh" ]; then
        # Give services time to fully start
        sleep 5
        bash ./scripts/test-vllm.sh
    else
        print_warning "Test script not found, skipping tests"
    fi
fi

echo ""
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Quick commands:"
echo "  Check pods:        kubectl get pods"
echo "  Check services:    kubectl get svc"
echo "  View logs:         kubectl logs -l app=vllm"
echo "  Port forward:      kubectl port-forward svc/vllm-router-service 30080:80"
echo "  Run tests:         ./scripts/test-vllm.sh"
echo ""
