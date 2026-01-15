# Deploylment

## Nvidia toolkit install

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2
   
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
sudo apt-get update -y

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-cuda-toolkit

sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

## k8s setup

```bash
git clone -b vllm-stack-0.1.8 https://github.com/vllm-project/production-stack.git

## Install kubectl
cd production-stack/utils
bash install-kubectl.sh

bash install-helm.sh

sudo minikube delete
sudo apt remove minikube
bash install-minikube-cluster.sh
# If gpu-operator fails to start because of the common seen “too many open files” issue for minikube (and kind), then a quick fix below may be helpful.
minikube status

# Make sure that nvidia toolkit was installed
kubectl describe nodes | grep -i gpu

# Test the gpu pod
# kubectl run gpu-test --image=nvidia/cuda:12.2.0-runtime-ubuntu22.04 --restart=Never -- nvidia-smi
# kubectl logs gpu-test

## Deploy the Helm Chart
helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
### Forward the Service Port
kubectl port-forward svc/vllm-router-service 30080:80

### API testing
curl -o- http://localhost:30080/v1/models

curl -X POST http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
	  }'
```

### k8s yaml for vLLM

```yaml
servingEngineSpec:
  strategy:
    type: Recreate
  runtimeClassName: ""
  modelSpec:
  - name: "qwen2_4B_quan"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "huypl53/Qwen3-VL-4B-Instruct-AWQ-INT4"
    replicaCount: 3
    requestCPU: 6
    requestMemory: "8Gi"
    requestGPU: 1
    limitCPU: 8
    limitMemory: "16Gi"

    vllmConfig:
      maxModelLen: 16384
      gpuMemoryUtilization: 0.25
      extraArgs: ["--disable-log-requests"]
```

- [ ]  how to run bash script in sudo mode that don’t ask password
