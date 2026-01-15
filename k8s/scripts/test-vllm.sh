#!/bin/bash
# Test script for vLLM deployment

set -e

SERVICE_PORT="${SERVICE_PORT:-30080}"
BASE_URL="${BASE_URL:-http://localhost:$SERVICE_PORT}"

echo "=== Testing vLLM Deployment ==="

# Test 1: List models
echo "[1/3] Testing models endpoint..."
echo "GET $BASE_URL/v1/models"
curl -s "$BASE_URL/v1/models" | jq . || echo "Response received (may need jq for pretty output)"
echo ""

# Test 2: Completion (with default model)
echo "[2/3] Testing completion endpoint..."
echo "POST $BASE_URL/v1/completions"
curl -s -X POST "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "Once upon a time,",
        "max_tokens": 10
    }' | jq . || echo "Response received (may need jq for pretty output)"
echo ""

# Test 3: Health check
echo "[3/3] Testing health endpoint..."
echo "GET $BASE_URL/health"
curl -s "$BASE_URL/health" || echo "Health endpoint not available"
echo ""

echo "=== Tests complete ==="
