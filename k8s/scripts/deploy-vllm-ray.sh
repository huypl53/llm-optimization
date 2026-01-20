#!/bin/bash
# This script deploys vLLM using KubeRay for distributed inference
# Supports pipeline and tensor parallelism across multiple GPUs/nodes

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Configuration
HELM_REPO_NAME="${HELM_REPO_NAME:-vllm}"
HELM_REPO_URL="${HELM_REPO_URL:-https://vllm-project.github.io/production-stack}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-vllm}"
HELM_CHART="${HELM_CHART:-vllm/vllm-stack}"
VALUES_FILE="${VALUES_FILE:-./config/vllm-ray-cluster-values.yaml}"
SERVICE_PORT="${SERVICE_PORT:-30080}"
SERVICE_NAME="${SERVICE_NAME:-vllm-router-service}"

echo "======================================"
echo "vLLM Deployment with KubeRay"
echo "======================================"
echo ""

# Check prerequisites
print_step "Checking prerequisites..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found"
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    print_error "helm not found"
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi
echo "Kubernetes cluster: connected"

# Check if KubeRay operator is installed
if ! kubectl get deployment kuberay-operator -n kuberay-system &> /dev/null 2>&1; then
    print_warning "KubeRay operator not found"
    echo ""
    echo "Please install KubeRay operator first:"
    echo "  ./scripts/install-kuberay-operator.sh"
    exit 1
fi
echo "KubeRay operator: installed"

# Check GPU availability
GPU_COUNT=$(kubectl describe nodes 2>/dev/null | grep -c "nvidia.com/gpu" || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    print_warning "No GPUs detected in cluster"
    echo "GPU operator may still be initializing..."
fi
echo ""

# Check if values file exists
if [ ! -f "$VALUES_FILE" ]; then
    print_error "Values file not found: $VALUES_FILE"
    exit 1
fi

# Check for HF_TOKEN
if [ -z "${HF_TOKEN}" ]; then
    print_warning "HF_TOKEN environment variable not set"
    print_warning "You may need it for gated models"
    echo ""
    read -p "Continue without HF_TOKEN? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Set HF_TOKEN and try again:"
        echo "  export HF_TOKEN=your_token_here"
        exit 1
    fi
fi
echo ""

# Add Helm repository
print_step "[1/6] Adding Helm repository..."
helm repo add "$HELM_REPO_NAME" "$HELM_REPO_URL" 2>/dev/null || true
helm repo update
echo ""

# Check for existing deployment
print_step "[2/6] Checking for existing deployment..."
if helm list -A | grep -q "$HELM_RELEASE_NAME"; then
    print_warning "Helm release '$HELM_RELEASE_NAME' already exists"
    echo ""
    read -p "Upgrade existing deployment? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Upgrading existing deployment..."
        helm upgrade "$HELM_RELEASE_NAME" "$HELM_CHART" -f "$VALUES_FILE"
    else
        echo "Deployment skipped"
        exit 0
    fi
else
    # Deploy the Helm Chart
    print_step "Deploying vLLM Helm chart with KubeRay..."
    helm install "$HELM_RELEASE_NAME" "$HELM_CHART" -f "$VALUES_FILE"
fi
echo ""

# Wait for Ray cluster to be ready
print_step "[3/6] Waiting for Ray cluster pods to be ready..."
echo "This may take several minutes as the model downloads..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance="$HELM_RELEASE_NAME" \
    --timeout=600s || true

echo ""
print_step "[4/6] Checking pod status..."
kubectl get pods -l app.kubernetes.io/instance="$HELM_RELEASE_NAME"
echo ""

# Show Ray cluster status
print_step "[5/6] Ray cluster status..."
kubectl get raycluster -o wide 2>/dev/null || true
echo ""

# Setup port forwarding (run in background)
print_step "[6/6] Setting up port forwarding..."
# Kill existing port-forward if any
pkill -f "port-forward.*$SERVICE_NAME" || true

echo "Starting port forward for $SERVICE_NAME on port $SERVICE_PORT..."
kubectl port-forward "svc/$SERVICE_NAME" "$SERVICE_PORT:80" &
PF_PID=$!
echo "Port forward started with PID: $PF_PID"
echo ""

# Wait a bit for services to be ready
sleep 5

echo "======================================"
echo "vLLM with KubeRay Deployed!"
echo "======================================"
echo ""
echo "Ray Cluster Information:"
echo "  Head node:     kubectl get pods -l ray.io/node-type=head"
echo "  Worker nodes:  kubectl get pods -l ray.io/node-type=worker"
echo ""
echo "API Access:"
echo "  Local:         http://localhost:$SERVICE_PORT"
echo ""
echo "Test Commands:"
echo "  List models:   curl http://localhost:$SERVICE_PORT/v1/models"
echo ""
echo "  Test completion:"
echo '    curl -X POST http://localhost:'$SERVICE_PORT'/v1/completions \'
echo '      -H "Content-Type: application/json" \'
echo '      -d '"'"'{"model": "huypl53/Qwen3-VL-4B-Instruct-AWQ-INT4", "prompt": "Hello, world!", "max_tokens": 20}'"'"
echo ""
echo "Monitoring:"
echo "  Logs:         kubectl logs -l app.kubernetes.io/instance='$HELM_RELEASE_NAME' -f"
echo "  GPU usage:    kubectl exec -it <ray-head-pod> -- nvidia-smi"
echo ""
echo "To stop port forwarding: kill $PF_PID"
echo ""

# Run quick health check
echo "Running health check..."
sleep 3
if curl -s "http://localhost:$SERVICE_PORT/v1/models" > /dev/null 2>&1; then
    print_step "API is responding!"
    curl -s "http://localhost:$SERVICE_PORT/v1/models" | head -20
else
    print_warning "API not ready yet, check pod logs:"
    echo "  kubectl logs -l app.kubernetes.io/instance=$HELM_RELEASE_NAME"
fi
echo ""
