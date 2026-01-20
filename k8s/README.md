# vLLM Kubernetes Deployment

This repository contains scripts for deploying vLLM (OpenAI-compatible API server for LLM inference) on Kubernetes with GPU support.

**Two Deployment Modes:**
- **Minikube Mode** - Local development and testing
- **KubeRay Mode** - Distributed inference with pipeline/tensor parallelism (for remote servers)

## Prerequisites

- Ubuntu/Debian Linux system
- NVIDIA GPU with drivers installed
- sudo privileges (for system package installation)
- At least 16GB RAM and 20GB disk space
- HuggingFace token (for gated models)

`★ Insight ─────────────────────────────────────`
**Which mode to choose?**
- **Minikube**: Quick local development, single-node, limited scale
- **KubeRay**: Production-grade distributed inference, multi-node, supports:
  - Pipeline parallelism (model stages across nodes)
  - Tensor parallelism (model shards across GPUs)
  - Better resource utilization for large models
`─────────────────────────────────────────────────`

## Quick Start

### Minikube Mode (Local Development)

```bash
./start.sh --mode minikube
```

### KubeRay Mode (Distributed Inference)

```bash
# For single-server with multiple GPUs (uses k3s)
./start.sh --mode kuberay --k8s-type k3s

# For multi-node cluster (uses kubeadm)
./start.sh --mode kuberay --k8s-type kubeadm
```

## Deployment Modes Comparison

| Feature | Minikube | KubeRay |
|---------|----------|---------|
| Use Case | Local dev/testing | Production/remote servers |
| Architecture | Single node | Head + worker nodes |
| Parallelism | Basic replicas | Pipeline + tensor parallelism |
| GPU Support | Single node | Multi-node GPU clusters |
| Scalability | Limited | Horizontal scaling |
| Setup Complexity | Simple | Moderate |

## KubeRay Configuration

### Understanding Parallelism

KubeRay enables two types of parallelism:

1. **Tensor Parallelism** - Model shards across GPUs within a node
2. **Pipeline Parallelism** - Model stages distributed across nodes

**Formula:**
```
Total GPUs = pipelineParallelSize × tensorParallelSize
```

### Configuration Examples

Edit `config/vllm-ray-cluster-values.yaml`:

```yaml
# Example 1: Single GPU (head only)
replicaCount: 0          # No workers
requestGPU: 1
tensorParallelSize: 1
pipelineParallelSize: 1
# Total: 1 GPU

# Example 2: 2 GPUs on one node (tensor parallelism)
replicaCount: 0          # No workers
requestGPU: 2
tensorParallelSize: 2
pipelineParallelSize: 1
# Total: 2 GPUs

# Example 3: 2 nodes with 1 GPU each (pipeline parallelism)
replicaCount: 1          # 1 worker
requestGPU: 1
tensorParallelSize: 1
pipelineParallelSize: 2  # head + 1 worker
# Total: 2 GPUs

# Example 4: 2 nodes with 2 GPUs each (both parallelisms)
replicaCount: 1          # 1 worker
requestGPU: 2
tensorParallelSize: 2
pipelineParallelSize: 2  # head + 1 worker
# Total: 4 GPUs
```

### Environment Variables

```bash
# HuggingFace token (for gated models)
export HF_TOKEN=your_token_here

# Service configuration
export SERVICE_PORT=30080
export HELM_RELEASE_NAME=vllm
export VALUES_FILE=./config/vllm-ray-cluster-values.yaml
```

## CLI Options

```bash
# Minikube mode (default)
./start.sh --mode minikube

# KubeRay mode with different k8s types
./start.sh --mode kuberay --k8s-type k3s      # Single-node cluster
./start.sh --mode kuberay --k8s-type kubeadm   # Multi-node cluster

# Skip steps (if already installed)
./start.sh --skip-nvidia
./start.sh --skip-k8s

# Only deploy (assumes k8s is ready)
./start.sh --mode kuberay --only-deploy

# Only run tests
./start.sh --only-test

# Show help
./start.sh --help
```

## Step-by-Step (KubeRay)

```bash
# 1. Install Nvidia Container Toolkit
./scripts/install-nvidia-toolkit.sh

# 2. Install k8s tools (kubectl, helm)
./scripts/install-k8s-tools.sh

# 3. Setup Kubernetes cluster (k3s or kubeadm)
K8S_TYPE=k3s ./scripts/install-k8s-cluster.sh

# 4. Install KubeRay operator
./scripts/install-kuberay-operator.sh

# 5. Deploy vLLM with Ray cluster
export HF_TOKEN=your_token
./scripts/deploy-vllm-ray.sh

# 6. Test the deployment
./scripts/test-vllm.sh
```

## Testing

After deployment, test the API:

```bash
# List available models
curl http://localhost:30080/v1/models

# Generate completion
curl -X POST http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "huypl53/Qwen3-VL-4B-Instruct-AWQ-INT4",
    "prompt": "Once upon a time,",
    "max_tokens": 20
  }'
```

Or run the test script:

```bash
./scripts/test-vllm.sh
```

## KubeRay Monitoring

```bash
# Check Ray cluster status
kubectl get raycluster
kubectl get pods -l ray.io/node-type=head
kubectl get pods -l ray.io/node-type=worker

# Check GPU usage on head node
kubectl exec -it <ray-head-pod> -- nvidia-smi

# Check GPU usage on worker nodes
kubectl exec -it <ray-worker-pod> -- nvidia-smi

# View logs
kubectl logs -l app.kubernetes.io/instance=vllm -f

# Port forward to access API
kubectl port-forward svc/vllm-router-service 30080:80
```

## Troubleshooting

### KubeRay operator not found

```bash
kubectl get pods -n kuberay-system
kubectl get deployment kuberay-operator -n kuberay-system
```

### Ray cluster pods not starting

Check GPU requests match availability:

```bash
# Check available GPUs
kubectl describe nodes | grep nvidia.com/gpu

# Check Ray cluster status
kubectl describe raycluster
```

### Model download issues

Ensure HF_TOKEN is set for gated models:

```bash
export HF_TOKEN=your_token_here
helm upgrade vllm vllm/vllm-stack -f config/vllm-ray-cluster-values.yaml
```

### Port forwarding issues

```bash
# Kill existing port forwards
pkill -f "port-forward"

# Start fresh
kubectl port-forward svc/vllm-router-service 30080:80
```

### Reset KubeRay deployment

```bash
# Uninstall Helm release
helm uninstall vllm

# Delete Ray cluster manually
kubectl delete raycluster --all

# Restart
./scripts/deploy-vllm-ray.sh
```

## Project Structure

```
.
├── start.sh                        # Main deployment script (supports both modes)
├── scripts/
│   ├── install-nvidia-toolkit.sh   # Nvidia toolkit installation
│   ├── install-k8s-tools.sh        # kubectl and helm installation
│   ├── install-minikube.sh         # Minikube cluster setup (for minikube mode)
│   ├── install-k8s-cluster.sh      # k3s/kubeadm setup (for kuberay mode)
│   ├── install-kuberay-operator.sh # KubeRay operator installation
│   ├── deploy-vllm.sh              # vLLM Helm deployment (minikube mode)
│   ├── deploy-vllm-ray.sh          # vLLM with Ray deployment (kuberay mode)
│   └── test-vllm.sh                # API testing script
├── config/
│   ├── vllm-values.yaml            # Helm values for minikube mode
│   └── vllm-ray-cluster-values.yaml # Helm values for kuberay mode
├── model-specs.yaml                # Model specifications
└── README.md                       # This file
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack)
- [KubeRay Documentation](https://ray.io/kuberay)
- [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Pipeline Parallelism Tutorial](https://github.com/vllm-project/production-stack/blob/main/tutorials/15-pipeline-parallelism-with-kuberay.md)
