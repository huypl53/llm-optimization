#!/bin/bash
# Main deployment script for vLLM on k8s
# This script orchestrates the entire setup process
# Supports two deployment modes:
#   - minikube: Local development with Minikube
#   - kuberay:  Distributed inference with KubeRay (for remote/multi-node)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}Info:${NC} $1"
}

# Parse command line arguments
SKIP_NVIDIA=false
SKIP_K8S=false
SKIP_DEPLOY=false
ONLY_DEPLOY=false
ONLY_TEST=false
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-minikube}"  # Options: minikube, kuberay
K8S_TYPE="${K8S_TYPE:-k3s}"  # For kuberay mode: k3s, kubeadm

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --k8s-type)
            K8S_TYPE="$2"
            shift 2
            ;;
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
            echo "Deployment Modes:"
            echo "  --mode minikube    Local development with Minikube (default)"
            echo "  --mode kuberay     Distributed inference with KubeRay"
            echo ""
            echo "KubeRay Options:"
            echo "  --k8s-type k3s     Single-node cluster (default for kuberay)"
            echo "  --k8s-type kubeadm  Multi-node cluster"
            echo ""
            echo "Other Options:"
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
print_info "Deployment mode: $DEPLOYMENT_MODE"
if [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
    print_info "K8s type: $K8S_TYPE"
fi
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

# Step 3: Setup Kubernetes cluster
if [ "$ONLY_DEPLOY" = false ] && [ "$SKIP_K8S" = false ]; then
    if [ "$DEPLOYMENT_MODE" = "minikube" ]; then
        print_step "Step 3: Setting up Minikube with GPU support..."
        if [ -f "./scripts/install-minikube.sh" ]; then
            bash ./scripts/install-minikube.sh
        else
            print_error "Minikube script not found"
            exit 1
        fi
        echo ""
    elif [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
        print_step "Step 3: Setting up Kubernetes cluster for KubeRay ($K8S_TYPE)..."
        if [ -f "./scripts/install-k8s-cluster.sh" ]; then
            K8S_TYPE="$K8S_TYPE" bash ./scripts/install-k8s-cluster.sh
        else
            print_error "K8s cluster script not found"
            exit 1
        fi
        echo ""
    fi
elif [ "$SKIP_K8S" = true ]; then
    print_warning "Skipping Kubernetes cluster setup"
fi

# Step 3.5: Install KubeRay operator (only for kuberay mode)
if [ "$ONLY_DEPLOY" = false ] && [ "$SKIP_K8S" = false ] && [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
    print_step "Step 3.5: Installing KubeRay operator..."
    if [ -f "./scripts/install-kuberay-operator.sh" ]; then
        bash ./scripts/install-kuberay-operator.sh
    else
        print_error "KubeRay operator script not found"
        exit 1
    fi
    echo ""
fi

# Step 4: Deploy vLLM
if [ "$SKIP_DEPLOY" = false ]; then
    if [ "$DEPLOYMENT_MODE" = "minikube" ]; then
        print_step "Step 4: Deploying vLLM on Kubernetes (standard mode)..."
        if [ -f "./scripts/deploy-vllm.sh" ]; then
            bash ./scripts/deploy-vllm.sh
        else
            print_error "vLLM deployment script not found"
            exit 1
        fi
        echo ""
    elif [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
        print_step "Step 4: Deploying vLLM on Kubernetes (KubeRay mode)..."
        if [ -f "./scripts/deploy-vllm-ray.sh" ]; then
            bash ./scripts/deploy-vllm-ray.sh
        else
            print_error "vLLM KubeRay deployment script not found"
            exit 1
        fi
        echo ""
    fi
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
if [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
    echo "KubeRay Deployment:"
    echo "  Ray head:        kubectl get pods -l ray.io/node-type=head"
    echo "  Ray workers:     kubectl get pods -l ray.io/node-type=worker"
    echo "  Ray cluster:     kubectl get raycluster"
    echo ""
fi
echo "Quick commands:"
echo "  Check pods:        kubectl get pods"
echo "  Check services:    kubectl get svc"
if [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
    echo "  View logs:         kubectl logs -l app.kubernetes.io/instance=vllm"
else
    echo "  View logs:         kubectl logs -l app=vllm"
fi
echo "  Port forward:      kubectl port-forward svc/vllm-router-service 30080:80"
echo "  Run tests:         ./scripts/test-vllm.sh"
echo ""
if [ "$DEPLOYMENT_MODE" = "kuberay" ]; then
    echo "KubeRay Monitoring:"
    echo "  GPU usage (head):  kubectl exec -it <ray-head-pod> -- nvidia-smi"
    echo "  GPU usage (worker): kubectl exec -it <ray-worker-pod> -- nvidia-smi"
    echo ""
fi
