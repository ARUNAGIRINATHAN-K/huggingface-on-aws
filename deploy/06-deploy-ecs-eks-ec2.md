<div align="center">

# Deploy with ECS, EKS, and EC2

*Deploy Hugging Face models on AWS compute services using Hugging Face Deep Learning Containers (DLCs) for full infrastructure control and production-grade containerized workloads.*

[![Back to Overview](https://img.shields.io/badge/←%20Deploy-Overview-blue)](./01-deploy-overview.md)
[![Official Docs](https://img.shields.io/badge/HF-DLC%20Docs-yellow)](https://huggingface.co/docs/sagemaker/en/dlcs/introduction)

</div>

---

## Overview

While SageMaker and Bedrock offer fully managed deployment, AWS also supports running Hugging Face DLCs directly on raw compute services: **EC2**, **ECS**, and **EKS**. This is ideal when you need full control over infrastructure, networking, or want to integrate HF inference into an existing containerized microservices architecture.

Hugging Face provides **Inference DLCs** pre-configured with:
- `transformers`, `tokenizers`, `datasets`
- `text-generation-inference` (TGI) — for LLMs (225,000+ models)
- `text-embeddings-inference` (TEI) — for embeddings (12,000+ models)
- Full S3 model loading support with no extra configuration
- Performance optimizations for PyTorch on AWS GPU hardware

**When to use this path:**
- You need full control over container orchestration and networking
- You're integrating HF models into existing ECS or EKS infrastructure
- You want to use custom hardware, spot instances, or Fargate
- You are building a multi-service ML platform without SageMaker

---

## Available DLC Image URIs

Pull images from Amazon ECR Public or use them in task/pod definitions:

```bash
# TGI (Text Generation Inference) — for LLMs
public.ecr.aws/huggingface/text-generation-inference:latest

# TEI (Text Embeddings Inference) — for embedding models
public.ecr.aws/huggingface/text-embeddings-inference:latest

# PyTorch Inference DLC (general purpose)
763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04
```

---

## Option 1: Deploy on Amazon EC2

The simplest way to run HF inference on a raw EC2 instance.

### Step 1: Launch an EC2 Instance

```bash
# Launch a GPU instance (e.g., g4dn.xlarge for T4 GPU)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \  # Amazon Linux 2 Deep Learning AMI
  --instance-type g4dn.xlarge \
  --key-name my-key \
  --security-group-ids sg-xxxx \
  --subnet-id subnet-xxxx
```

> **Tip:** Use the **Hugging Face Neuron Deep Learning AMI** from AWS Marketplace for Trainium/Inferentia instances — it comes with all Neuron drivers pre-installed.

### Step 2: Pull and Run the TGI DLC

```bash
# SSH into your instance, then run TGI with Docker
docker run --gpus all \
  -e HF_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
  -e HUGGING_FACE_HUB_TOKEN="<YOUR_HF_TOKEN>" \
  -e MAX_INPUT_LENGTH=4096 \
  -e MAX_TOTAL_TOKENS=8192 \
  -p 8080:80 \
  public.ecr.aws/huggingface/text-generation-inference:latest
```

### Step 3: Test Your Endpoint

```bash
# Test the running TGI container
curl localhost:8080/generate \
  -X POST \
  -d '{"inputs": "Why is the sky blue?", "parameters": {"max_new_tokens": 100}}' \
  -H 'Content-Type: application/json'
```

### Load Model from S3

```bash
# Mount an S3 model using environment variables
docker run --gpus all \
  -e MODEL_ID="s3://your-bucket/your-model" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  -p 8080:80 \
  public.ecr.aws/huggingface/text-generation-inference:latest
```

---

## Option 2: Deploy on Amazon ECS (Fargate or EC2 Launch Type)

ECS is a fully managed container orchestration service. You define your HF container as an ECS Task Definition and deploy it as a Service.

### Step 1: Create an ECS Task Definition

```json
{
  "family": "hf-tgi-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "tgi-container",
      "image": "public.ecr.aws/huggingface/text-generation-inference:latest",
      "portMappings": [
        {"containerPort": 80, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "HF_MODEL_ID", "value": "mistralai/Mistral-7B-Instruct-v0.3"},
        {"name": "MAX_INPUT_LENGTH", "value": "4096"},
        {"name": "MAX_TOTAL_TOKENS", "value": "8192"}
      ],
      "secrets": [
        {
          "name": "HUGGING_FACE_HUB_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:<region>:<account>:secret:hf-token"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hf-tgi",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "tgi"
        }
      }
    }
  ]
}
```

### Step 2: Register and Deploy the Task

```bash
# Register the task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS cluster
aws ecs create-cluster --cluster-name hf-inference-cluster

# Create ECS service
aws ecs create-service \
  --cluster hf-inference-cluster \
  --service-name hf-tgi-service \
  --task-definition hf-tgi-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Step 3: Add Auto-Scaling

```bash
# Register the ECS service as a scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/hf-inference-cluster/hf-tgi-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 5

# Add CPU-based scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/hf-inference-cluster/hf-tgi-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"}
  }'
```

---

## Option 3: Deploy on Amazon EKS (Kubernetes)

EKS is ideal for organizations already running Kubernetes workloads.

### Step 1: Create a Kubernetes Deployment

```yaml
# hf-tgi-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-tgi-deployment
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hf-tgi
  template:
    metadata:
      labels:
        app: hf-tgi
    spec:
      containers:
        - name: tgi
          image: public.ecr.aws/huggingface/text-generation-inference:latest
          ports:
            - containerPort: 80
          env:
            - name: HF_MODEL_ID
              value: "meta-llama/Meta-Llama-3-8B-Instruct"
            - name: MAX_INPUT_LENGTH
              value: "4096"
            - name: MAX_TOTAL_TOKENS
              value: "8192"
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: token
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              nvidia.com/gpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: hf-tgi-service
spec:
  selector:
    app: hf-tgi
  ports:
    - port: 80
      targetPort: 80
  type: LoadBalancer
```

### Step 2: Create the HF Token Secret and Apply

```bash
# Create the Kubernetes secret for the HF token
kubectl create secret generic hf-secret \
  --from-literal=token=<YOUR_HF_TOKEN>

# Apply the deployment and service
kubectl apply -f hf-tgi-deployment.yaml

# Check status
kubectl get pods
kubectl get svc hf-tgi-service
```

### Step 3: Add Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hf-tgi-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hf-tgi-deployment
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

```bash
kubectl apply -f hpa.yaml
```

---

## TEI (Text Embeddings Inference) on EC2/ECS/EKS

For embedding models, swap the TGI image for TEI:

```bash
# Run TEI on EC2
docker run \
  -e MODEL_ID="BAAI/bge-large-en-v1.5" \
  -p 8080:80 \
  public.ecr.aws/huggingface/text-embeddings-inference:latest

# Test
curl localhost:8080/embed \
  -X POST \
  -d '{"inputs": ["Hello world", "Test sentence"]}' \
  -H 'Content-Type: application/json'
```

---

## Comparison: ECS vs EKS vs EC2

| Feature | EC2 | ECS | EKS |
|---------|-----|-----|-----|
| Management overhead | High | Medium | Medium-High |
| Kubernetes native | ❌ | ❌ | ✅ |
| Serverless option | ❌ | ✅ Fargate | ✅ Fargate |
| Auto-scaling | Manual | ✅ App Auto Scaling | ✅ HPA / KEDA |
| Best for | Dev / testing | Production containers | Kubernetes-first orgs |

---

## Useful Resources

- [ECS Documentation](https://docs.aws.amazon.com/ecs) - Amazon ECS documentation
- [EKS Documentation](https://docs.aws.amazon.com/eks) - Amazon EKS documentation
- [HF DLC Introduction](https://huggingface.co/docs/sagemaker/en/dlcs/introduction) - Hugging Face Deep Learning Container (DLC) introduction
- [HF ECR Public Gallery](https://gallery.ecr.aws/huggingface) - Hugging Face images in Amazon ECR Public gallery
- [TEI GitHub](https://github.com/huggingface/text-embeddings-inference) - Text Embeddings Inference (TEI) repository (GitHub)
- [TGI GitHub](https://github.com/huggingface/text-generation-inference) - Text Generation Inference (TGI) repository (GitHub)