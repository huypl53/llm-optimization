#!/bin/bash
# This script deploys vLLM using Helm chart with custom configuration

set -e

# Configuration
HELM_REPO_NAME="${HELM_REPO_NAME:-vllm}"
HELM_REPO_URL="${HELM_REPO_URL:-https://vllm-project.github.io/production-stack}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-vllm}"
HELM_CHART="${HELM_CHART:-vllm/vllm-stack}"
VALUES_FILE="${VALUES_FILE:-./config/vllm-values.yaml}"
SERVICE_PORT="${SERVICE_PORT:-30080}"
SERVICE_NAME="${SERVICE_NAME:-vllm-router-service}"

echo "=== Deploying vLLM on k8s ==="

# Check if values file exists
if [ ! -f "$VALUES_FILE" ]; then
    echo "Error: Values file not found: $VALUES_FILE"
    echo "Please create a values file or specify VALUES_FILE environment variable"
    exit 1
fi

# Add Helm repository
echo "[1/5] Adding Helm repository..."
helm repo add "$HELM_REPO_NAME" "$HELM_REPO_URL"
helm repo update

# Deploy the Helm Chart
echo "[2/5] Deploying vLLM Helm chart..."
helm install "$HELM_RELEASE_NAME" "$HELM_CHART" -f "$VALUES_FILE"

# Wait for deployment to be ready
echo "[3/5] Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/"$HELM_RELEASE_NAME"-vllm 2>/dev/null || true

# Setup port forwarding (run in background)
echo "[4/5] Setting up port forwarding..."
echo "Starting port forward for $SERVICE_NAME on port $SERVICE_PORT..."
kubectl port-forward "svc/$SERVICE_NAME" "$SERVICE_PORT:80" &
PF_PID=$!
echo "Port forward started with PID: $PF_PID"

# Wait a bit for services to start
echo "[5/5] Waiting for services to initialize..."
sleep 10

echo "=== vLLM deployment complete ==="
echo ""
echo "API is available at: http://localhost:$SERVICE_PORT"
echo ""
echo "Test commands:"
echo "  curl http://localhost:$SERVICE_PORT/v1/models"
echo ""
echo "To stop port forwarding: kill $PF_PID"
echo ""
echo "To test completion:"
echo '  curl -X POST http://localhost:'$SERVICE_PORT'/v1/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model": "facebook/opt-125m", "prompt": "Once upon a time,", "max_tokens": 10}'"'"
