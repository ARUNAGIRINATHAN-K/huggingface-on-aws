<div align="center">

# Train with ECS, EKS, and EC2

*Run Hugging Face training jobs on AWS compute services using Training DLCs for teams that need full infrastructure control or use existing container-based ML platforms.*

[![Back to Overview](https://img.shields.io/badge/←%20Train-Overview-blue)](./07-train-overview.md)
[![Official Docs](https://img.shields.io/badge/HF-DLC%20Docs-yellow)](https://huggingface.co/docs/sagemaker/en/dlcs/introduction)

</div>

---

## Overview

Hugging Face provides **Training Deep Learning Containers (DLCs)** that are pre-configured with all necessary libraries for fine-tuning. While these DLCs are natively integrated into SageMaker SDK, they can also be pulled directly from Amazon ECR and used on **EC2**, **ECS**, and **EKS** — giving you complete freedom over orchestration, networking, and compute.

**When to use this path:**
- Your team already runs ML workloads on Kubernetes (EKS) or ECS
- You want to integrate HF training into an existing CI/CD or MLOps pipeline
- You need custom hardware configurations not available in SageMaker
- You want to use your own distributed training framework (e.g., Ray, Horovod)
- You want to avoid SageMaker overhead for long-running training jobs

---

## Available Training DLC Images

```bash
# PyTorch Training DLC (GPU)
763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04

# PyTorch Training DLC (CPU)
763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-cpu-py310-ubuntu20.04

# Neuron Training DLC (Trainium)
763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-training-neuronx:1.13.1-transformers4.36.0-neuronx-py310-sdk2.15.0-ubuntu20.04
```

---

## Option 1: Train on Amazon EC2

Ideal for development, one-off training runs, or when you want full SSH access.

### Step 1: Launch a GPU EC2 Instance

```bash
# Launch a P3 instance (V100 GPU) with the Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type p3.2xlarge \
  --key-name my-keypair \
  --security-group-ids sg-xxxx \
  --subnet-id subnet-xxxx \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100}}]'
```

### Step 2: Pull the Training DLC and Run

```bash
# SSH into the instance
ssh -i my-keypair.pem ec2-user@<public-ip>

# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Pull the Hugging Face training DLC
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04

# Run training
docker run --gpus all \
  -v $(pwd)/scripts:/opt/ml/code \
  -v $(pwd)/data:/opt/ml/input/data/train \
  -v $(pwd)/output:/opt/ml/model \
  -e SM_MODEL_DIR=/opt/ml/model \
  -e SM_CHANNEL_TRAIN=/opt/ml/input/data/train \
  763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04 \
  python /opt/ml/code/train.py \
    --model_name_or_path distilbert-base-uncased \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32
```

### Step 3: Sync Your Model to S3

```bash
# After training, upload the saved model to S3
aws s3 sync ./output s3://your-bucket/models/my-fine-tuned-model/
```

---

## Option 2: Train with Amazon ECS

Use ECS to run training as a containerized batch job — ideal for pipelines and automated workflows.

### Step 1: Build a Custom Training Image

```dockerfile
# Dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04

# Copy your training script
COPY train.py /opt/ml/code/train.py

# Install additional dependencies
RUN pip install peft trl accelerate datasets

# Set entry point
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
```

```bash
# Build and push to ECR
aws ecr create-repository --repository-name hf-training

docker build -t hf-training .
docker tag hf-training:latest <account>.dkr.ecr.us-east-1.amazonaws.com/hf-training:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/hf-training:latest
```

### Step 2: Create ECS Task Definition

```json
{
  "family": "hf-training-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "8192",
  "memory": "32768",
  "containerDefinitions": [
    {
      "name": "hf-trainer",
      "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/hf-training:latest",
      "resourceRequirements": [
        {"type": "GPU", "value": "1"}
      ],
      "environment": [
        {"name": "SM_MODEL_DIR", "value": "/opt/ml/model"},
        {"name": "SM_CHANNEL_TRAIN", "value": "/opt/ml/input/data/train"},
        {"name": "HUGGING_FACE_HUB_TOKEN", "value": "<YOUR_TOKEN>"}
      ],
      "mountPoints": [
        {
          "sourceVolume": "model-output",
          "containerPath": "/opt/ml/model"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hf-training",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "train"
        }
      }
    }
  ],
  "volumes": [
    {"name": "model-output", "host": {"sourcePath": "/tmp/model"}}
  ]
}
```

### Step 3: Run the Training Task

```bash
# Register the task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Run the training job as a one-time task
aws ecs run-task \
  --cluster hf-training-cluster \
  --task-definition hf-training-task \
  --launch-type EC2 \
  --count 1
```

---

## Option 3: Train on Amazon EKS (Kubernetes)

Ideal for teams running distributed training workloads or needing Kubernetes-native orchestration.

### Single-GPU Training Job

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hf-training-job
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: <account>.dkr.ecr.us-east-1.amazonaws.com/hf-training:latest
          args:
            - "--model_name_or_path=distilbert-base-uncased"
            - "--num_train_epochs=3"
            - "--per_device_train_batch_size=32"
            - "--output_dir=/opt/ml/model"
          env:
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
          volumeMounts:
            - name: model-storage
              mountPath: /opt/ml/model
      volumes:
        - name: model-storage
          emptyDir: {}
```

```bash
kubectl apply -f training-job.yaml
kubectl logs -f job/hf-training-job
```

### Distributed Training with PyTorchJob (Kubeflow)

For multi-GPU/multi-node distributed training, use the Kubeflow `PyTorchJob` CRD:

```yaml
# distributed-training-job.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: hf-distributed-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <account>.dkr.ecr.us-east-1.amazonaws.com/hf-training:latest
              args:
                - "--model_name_or_path=meta-llama/Meta-Llama-3-8B"
                - "--num_train_epochs=1"
                - "--per_device_train_batch_size=4"
              resources:
                limits:
                  nvidia.com/gpu: "4"
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <account>.dkr.ecr.us-east-1.amazonaws.com/hf-training:latest
              resources:
                limits:
                  nvidia.com/gpu: "4"
```

---

## Use Spot Instances for Cost Savings

### EC2 Spot for Training

```bash
# Launch a spot instance for training (up to 90% cheaper)
aws ec2 request-spot-instances \
  --spot-price "0.50" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0c02fb55956c7d316",
    "InstanceType": "p3.2xlarge",
    "KeyName": "my-keypair",
    "SecurityGroupIds": ["sg-xxxx"]
  }'
```

### EKS Spot Node Group

```bash
# Add a spot GPU node group to your EKS cluster
eksctl create nodegroup \
  --cluster my-cluster \
  --name gpu-spot-ng \
  --node-type p3.2xlarge \
  --spot \
  --nodes-min 0 \
  --nodes-max 5 \
  --node-labels "role=training"
```

---

## Train on Trainium with EC2

For Trainium (trn1) instances, use the Hugging Face Neuron Deep Learning AMI:

```bash
# Subscribe to the HF Neuron DL AMI on AWS Marketplace first, then:
aws ec2 run-instances \
  --image-id <neuron-dlami-id> \
  --instance-type trn1.2xlarge \
  --key-name my-keypair

# SSH in, then run training with optimum-neuron
docker run \
  -v $(pwd)/scripts:/opt/ml/code \
  763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:1.13.1-transformers4.36.0-neuronx-py310-sdk2.15.0-ubuntu20.04 \
  python /opt/ml/code/train.py \
    --model_name_or_path bert-base-uncased \
    --num_train_epochs 3
```

---

## Upload Trained Model to S3 and Deploy

After any training job, sync the output to S3 for later deployment:

```python
import boto3

s3 = boto3.client("s3")
s3.upload_file("output/pytorch_model.bin", "your-bucket", "models/my-model/pytorch_model.bin")
s3.upload_file("output/config.json", "your-bucket", "models/my-model/config.json")
s3.upload_file("output/tokenizer.json", "your-bucket", "models/my-model/tokenizer.json")

# Or use CLI
# aws s3 sync ./output s3://your-bucket/models/my-model/
```

Then deploy with the [SageMaker SDK Deploy guide](../deploy/02-deploy-sagemaker-sdk.md).

---

## Comparison: SageMaker SDK vs ECS/EKS/EC2 for Training

| Feature | SageMaker SDK | ECS / EKS / EC2 |
|---------|--------------|-----------------|
| Infrastructure management | ✅ Fully managed | ❌ You manage it |
| Spot instances | ✅ Managed spot | Manual setup |
| Distributed training | ✅ Built-in SageMaker Distributed | Manual (Horovod, PyTorchJob) |
| Experiment tracking | ✅ SageMaker Experiments | 3rd party (MLflow, W&B) |
| Integration with existing K8s | ❌ | ✅ |
| CI/CD pipeline integration | Medium | High |
| Cost (idle) | $0 (pay per job) | EC2/ECS/EKS cluster costs |

---

## Useful Resources
## Useful Resources

- [AWS ECS GPU Docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html) - Guide for running GPU workloads on ECS
- [Available DLC Image Tags](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) - List of available DLC image tags (GitHub)
- [HF Training DLC Docs](https://huggingface.co/docs/sagemaker/en/dlcs/introduction) - Hugging Face Training DLC documentation
- [HF Training DLC GitHub](https://github.com/aws/deep-learning-containers) - AWS Deep Learning Containers repository (GitHub)
- [Kubeflow PyTorchJob](https://www.kubeflow.org/docs/components/training/pytorch) - Kubeflow documentation for PyTorchJob CRD
- [Optimum Neuron (Trainium)](https://huggingface.co/docs/optimum-neuron) - Optimum Neuron docs for Trainium