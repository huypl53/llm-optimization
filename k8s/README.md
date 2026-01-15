# vLLM Kubernetes Deployment

This repository contains scripts for deploying vLLM (OpenAI-compatible API server for LLM inference) on a local Kubernetes cluster with GPU support.

## Prerequisites

- Ubuntu/Debian Linux system
- NVIDIA GPU with drivers installed
- sudo privileges (for system package installation)
- At least 16GB RAM and 20GB disk space

## Quick Start

### Full Deployment

Run the complete setup process:

```bash
./start.sh
```

This will:
1. Install Nvidia Container Toolkit
2. Install kubectl and helm
3. Setup Minikube with GPU support
4. Deploy vLLM with your configured models
5. Run API tests

### Step-by-Step Deployment

If you prefer to run steps individually:

```bash
# 1. Install Nvidia Container Toolkit
./scripts/install-nvidia-toolkit.sh

# 2. Install k8s tools (kubectl, helm)
./scripts/install-k8s-tools.sh

# 3. Setup Minikube with GPU support
./scripts/install-minikube.sh

# 4. Deploy vLLM
./scripts/deploy-vllm.sh

# 5. Test the deployment
./scripts/test-vllm.sh
```

## Configuration

### Model Configuration

Edit `config/vllm-values.yaml` to configure your models:

```yaml
servingEngineSpec:
  modelSpec:
    - name: "your-model-name"
      modelURL: "your-huggingface-model"
      replicaCount: 3
      requestGPU: 1
      vllmConfig:
        maxModelLen: 16384
        gpuMemoryUtilization: 0.25
```

### Environment Variables

Customize deployment with environment variables:

- `SERVICE_PORT`: Port for service forwarding (default: 30080)
- `HELM_RELEASE_NAME`: Helm release name (default: vllm)
- `VALUES_FILE`: Path to values file (default: ./config/vllm-values.yaml)

## CLI Options

```bash
# Skip Nvidia toolkit installation (if already installed)
./start.sh --skip-nvidia

# Skip k8s tools installation (if already installed)
./start.sh --skip-k8s

# Only deploy vLLM (assumes k8s is already set up)
./start.sh --only-deploy

# Only run tests (assumes vLLM is already deployed)
./start.sh --only-test

# Show help
./start.sh --help
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
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
  }'
```

Or run the test script:

```bash
./scripts/test-vllm.sh
```

## Useful k8s Commands

```bash
# Check pod status
kubectl get pods

# Check services
kubectl get svc

# View vLLM logs
kubectl logs -l app=vllm -f

# Port forward to access API
kubectl port-forward svc/vllm-router-service 30080:80

# Describe node (check GPU)
kubectl describe nodes | grep -i gpu

# Get into a pod
kubectl exec -it <pod-name> -- bash
```

## Troubleshooting

### GPU not detected

Verify GPU is available:

```bash
nvidia-smi
kubectl describe nodes | grep -i gpu
```

### GPU pod fails to start

For "too many open files" error in Minikube:

```bash
minikube ssh
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=512
```

### Port forwarding issues

Kill existing port forwards:

```bash
pkill -f "port-forward"
```

### Reset everything

```bash
# Delete Minikube cluster
minikube delete

# Uninstall Helm release
helm uninstall vllm

# Restart from scratch
./start.sh
```

## Project Structure

```
.
├── start.sh                      # Main deployment script
├── scripts/
│   ├── install-nvidia-toolkit.sh # Nvidia toolkit installation
│   ├── install-k8s-tools.sh      # kubectl and helm installation
│   ├── install-minikube.sh       # Minikube cluster setup
│   ├── deploy-vllm.sh            # vLLM Helm deployment
│   └── test-vllm.sh              # API testing script
├── config/
│   └── vllm-values.yaml          # Helm chart values
├── model-specs.yaml              # Model specifications
└── README.md                     # This file
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack)
- [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Minikube GPU Support](https://minikube.sigs.k8s.io/docs/drivers/nvidia/)
