<div align="center">

![](img/cover.png)

# Awesome AWS + Hugging Face

*A curated list of high-quality resources, tools, projects, and guides for running Hugging Face models on AWS infrastructure.Individual reference guides for deploying and training Hugging Face models on AWS.*

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![AWS](https://img.shields.io/badge/AWS-Cloud-FF9900?logo=amazonaws&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)
![Last Updated](https://img.shields.io/badge/updated-March%202026-brightgreen)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Official Resources](#official-resources)
- [AWS Services for Hugging Face](#aws-services-for-hugging-face)
- [Getting Started Guides](#getting-started-guides)
- [Key Tools & Libraries](#key-tools--libraries)
- [Projects & Sample Notebooks](#projects--sample-notebooks)
- [Use Cases](#use-cases)
- [Hardware: Trainium & Inferentia](#hardware-trainium--inferentia)
- [Blog Posts & Articles](#blog-posts--articles)
- [Videos & Courses](#videos--courses)
- [Community & Support](#community--support)
- [Contributing](#contributing)

---
## Overview

AWS is the **preferred cloud provider** for Hugging Face, enabling developers to access, fine-tune, and deploy over 1 million open-source models at scale with up to **50% lower training costs** (Trainium) and **40% lower inference costs** (Inferentia2) over comparable EC2 instances.

By combining Hugging Face's open-source ecosystem with AWS's scalable, secure infrastructure, developers can go from model exploration to production in minutes.

---

## Documentation

### Deploy Models on AWS
| Document | Description |
|----------|-------------|
| [01-deploy-overview.md](./deploy/01-deploy-overview.md) | Overview of all deployment paths on AWS |
| [02-deploy-sagemaker-sdk.md](./deploy/02-deploy-sagemaker-sdk.md) | Deploy with Amazon SageMaker SDK |
| [03-deploy-sagemaker-jumpstart.md](./deploy/03-deploy-sagemaker-jumpstart.md) | Deploy with SageMaker JumpStart |
| [04-deploy-bedrock.md](./deploy/04-deploy-bedrock.md) | Deploy with AWS Bedrock Marketplace |
| [05-deploy-inference-endpoints.md](./deploy/05-deploy-inference-endpoints.md) | Deploy with Hugging Face Inference Endpoints |
| [06-deploy-ecs-eks-ec2.md](./deploy/06-deploy-ecs-eks-ec2.md) | Deploy with ECS, EKS, and EC2 |

### Train Models on AWS
| Document | Description |
|----------|-------------|
| [07-train-overview.md](./train/07-train-overview.md) | Overview of all training paths on AWS |
| [08-train-sagemaker-sdk.md](./train/08-train-sagemaker-sdk.md) | Train with SageMaker SDK |
| [09-train-ecs-eks-ec2.md](./train/09-train-ecs-eks-ec2.md) | Train with ECS, EKS, and EC2 |

---

## How to Choose

| Scenario | Recommended Path |
|----------|-----------------|
| Quick deployment, no infra management | [SageMaker JumpStart](./deploy/03-deploy-sagemaker-jumpstart.md) or [Inference Endpoints](./deploy/05-deploy-inference-endpoints.md) |
| Custom model from S3 or after fine-tuning | [SageMaker SDK](./deploy/02-deploy-sagemaker-sdk.md) |
| Need Bedrock APIs (Agents, Guardrails, KB) | [AWS Bedrock](./deploy/04-deploy-bedrock.md) |
| Full infrastructure control | [ECS / EKS / EC2](./deploy/06-deploy-ecs-eks-ec2.md) |
| Fine-tune with managed infra | [Train with SageMaker SDK](./train/08-train-sagemaker-sdk.md) |
| Custom training with containers | [Train with ECS/EKS/EC2](./train/09-train-ecs-eks-ec2.md) |

---

## Official Resources

- [Amazon/HuggingFace Hub Page](https://huggingface.co/amazon) - Amazon's official model & dataset page on Hugging Face
- [AWS SageMaker HF Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html) - Official AWS docs for Hugging Face DLCs on SageMaker
- [Hugging Face on AWS](https://aws.amazon.com/ai/hugging-face/) - Official AWS landing page for the partnership
- [Hugging Face on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-n6vsyhdjkfng2) - Access HUGS and models via AWS Marketplace
- [HF Docs: SageMaker Integration](https://huggingface.co/docs/sagemaker/index) - Full documentation for training & deploying on SageMaker

---

## AWS Services for Hugging Face

### Amazon SageMaker AI
A fully managed ML platform for the **entire model development lifecycle** from training to deployment.

- **SageMaker Studio** — Web-based IDE for ML workflows
- **SageMaker JumpStart** — One-click deployment of popular HF models (Llama 3, Mistral, Falcon 2, StarCoder)
- **SageMaker Estimator** — Fine-tune HF models with a few lines of Python
- **Managed Endpoints** — Deploy real-time or asynchronous inference with auto-scaling
- **Distributed Training** — Fully integrated with HF DLCs for multi-GPU/multi-node jobs

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    hyperparameters={
        'model_name_or_path': 'distilbert-base-uncased',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 32,
    }
)
huggingface_estimator.fit({'train': training_input_path})
```

---

### Amazon Bedrock
Serverless access to foundation models via a single API.

- Access HF models alongside Amazon Titan, Claude, and others
- Integrated with **Agents**, **Knowledge Bases**, **Guardrails**, and **Model Evaluations**
- Available on **Bedrock Marketplace** for discovering and deploying open models

---

### Deep Learning Containers (DLCs)
Pre-built, optimized Docker images with Hugging Face libraries pre-installed:

- Includes **Transformers**, **Tokenizers**, and **Datasets**
- Natively integrated with SageMaker SDK, ECS, EKS, and EC2
- No manual dependency management needed
- Regularly updated with security patches
- Available via Amazon ECR

---

### Other AWS Compute Options

| Service | Use Case |
|---------|----------|
| **Amazon EC2** (P4/P5/G5 instances) | Full control, custom ML workloads |
| **AWS ECS** | Containerized HF model serving |
| **AWS EKS** | Kubernetes-native HF deployments |
| **AWS Fargate** | Serverless containers for HF inference |

---

## Getting Started Guides

- [Custom Inference Code with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html) - Bring your own inference logic (AWS SageMaker docs)
- [Deploy LLMs with TGI on Inferentia2](https://huggingface.co/docs/sagemaker/index) - High-throughput LLM serving (Hugging Face / SageMaker)
- [Deploy HF Models with SageMaker JumpStart](https://aws.amazon.com/ai/hugging-face/) - One-click deployment walkthrough (AWS)
- [Fine-tune BERT for Text Classification on SageMaker](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face) - Step-by-step notebook-based fine-tuning (Hugging Face blog)
- [HF Estimator Training on SageMaker](https://huggingface.co/docs/sagemaker/index) - Using SageMaker SDK v2/v3 for training jobs (Hugging Face docs)
- [Training on Trainium with Optimum Neuron](https://huggingface.co/docs/optimum-neuron) - Optimize for AWS custom silicon (Optimum Neuron docs)

---

## Key Tools & Libraries

- [datasets](https://huggingface.co/docs/datasets) - Efficient dataset loading & processing (Docs: [Hugging Face Docs](https://huggingface.co/docs/datasets) · [GitHub](https://github.com/huggingface/datasets))
- [optimum-neuron](https://huggingface.co/docs/optimum-neuron) - HF models on Trainium & Inferentia (Docs: [Hugging Face Docs](https://huggingface.co/docs/optimum-neuron) · [GitHub](https://github.com/huggingface/optimum-neuron))
- [peft](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning (LoRA, QLoRA) (Docs: [Hugging Face Docs](https://huggingface.co/docs/peft) · [GitHub](https://github.com/huggingface/peft))
- [smolagents](https://huggingface.co/docs/smolagents) - Lightweight agentic AI framework (Docs: [Hugging Face Docs](https://huggingface.co/docs/smolagents) · [GitHub](https://github.com/huggingface/smolagents))
- [sagemaker (Python SDK)](https://sagemaker.readthedocs.io) - Train & deploy on SageMaker (Docs: [Hugging Face Docs](https://sagemaker.readthedocs.io) · [GitHub](https://github.com/aws/sagemaker-python-sdk))
- [text-generation-inference](https://github.com/huggingface/text-generation-inference) - Production LLM serving stack ([GitHub](https://github.com/huggingface/text-generation-inference))
- [transformers](https://huggingface.co/docs/transformers) - Core library for 100k+ HF models (Docs: [Hugging Face Docs](https://huggingface.co/docs/transformers) · [GitHub](https://github.com/huggingface/transformers))
- [trl](https://huggingface.co/docs/trl) - Reinforcement Learning from Human Feedback (Docs: [Hugging Face Docs](https://huggingface.co/docs/trl) · [GitHub](https://github.com/huggingface/trl))

---

## Projects & Sample Notebooks

### Official AWS + HF Sample Repos

- [aws/amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples) - Broad SageMaker examples including HF ([GitHub](https://github.com/aws/amazon-sagemaker-examples))
- [huggingface/notebooks](https://github.com/huggingface/notebooks/tree/main/sagemaker) - Official HF SageMaker example notebooks ([GitHub](https://github.com/huggingface/notebooks/tree/main/sagemaker))
- [aws-samples/sample-healthcare-agent-with-smolagents-on-aws](https://github.com/aws-samples/sample-healthcare-agent-with-smolagents-on-aws) - Multi-model agentic AI system on AWS with smolagents ([GitHub](https://github.com/aws-samples/sample-healthcare-agent-with-smolagents-on-aws))


### Community Projects

- Batch Inference with SageMaker Processing - Cost-efficient large-scale batch jobs
- Fine-tuning Llama 3 on SageMaker - QLoRA fine-tuning with PEFT + SageMaker
- RAG Pipeline with HF + Amazon Bedrock - Retrieval-Augmented Generation using OpenSearch + HF embeddings
- Serverless HF Inference on Lambda - Lightweight inference with Lambda + ECR

---

## Use Cases

| Use Case | Recommended Models | AWS Services |
|----------|-------------------|--------------|
| **Text Summarization** | Meta Llama 3, BART | SageMaker Endpoints |
| **Code Generation** | StarCoder, CodeLlama | SageMaker JumpStart |
| **Chat / Virtual Assistants** | Llama 3 Instruct, Falcon 2 | SageMaker + Bedrock |
| **Semantic Search / RAG** | BGE, E5, all-MiniLM | SageMaker + OpenSearch |
| **Time Series Forecasting** | Amazon Chronos | SageMaker |
| **Image Generation** | Stable Diffusion | SageMaker (GPU) |
| **Document Classification** | DistilBERT, RoBERTa | Inferentia2 |
| **Agentic AI Systems** | Any LLM + smolagents | SageMaker + Bedrock + Fargate |

---

## Blog Posts & Articles

| Title | Author | Date |
|-------|--------|------|
| [Hugging Face Transforming AI Adoption](https://aws.amazon.com/isv/resources/hugging-face-transforming-ai-adoption/) | AWS Editorial | Feb 2025 |
| [The Partnership: Amazon SageMaker and Hugging Face](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face) | Hugging Face | 2021 |
| [Agentic AI with smolagents on AWS](https://aws.amazon.com/blogs/machine-learning/agentic-ai-with-multi-model-framework-using-hugging-face-smolagents-on-aws/) | AWS ML Blog | Feb 2026 |
| [Deploy Hugging Face models easily with SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker) | Hugging Face | 2022 |
| [AWS Inferentia2 Hugging Face HUGS](https://huggingface.co/blog/inferentia-llama) | Hugging Face | 2024 |

---

## Videos & Courses

- [AWS Machine Learning University](https://aws.amazon.com/machine-learning/mlu/) - Free ML courses by AWS
- [Fine-Tuning LLMs on AWS Trainium](https://www.youtube.com/results?search_query=aws+trainium+hugging+face) - AWS re:Invent talks (YouTube)
- [Getting Started: HF Models on SageMaker](https://www.youtube.com/watch?v=ok3hetb42gU) - Hugging Face YouTube
- [Hugging Face Course](https://huggingface.co/course) - Free NLP/ML course (deployable on AWS)

---

## Community & Support

- [AWS Developer Slack](https://aws.amazon.com/developer/community) - AWS developer community and resources
- [AWS re:Post (ML)](https://repost.aws/tags/TAMNxP3i3xTSuqq_ZtS7mE4g/amazon-sage-maker) - Community Q&A for AWS machine learning
- [Hugging Face Discord](https://discord.gg/huggingface) - Real-time community chat for Hugging Face
- [Hugging Face Forums](https://discuss.huggingface.co) - Official Hugging Face discussion forums
- [Hugging Face GitHub Issues](https://github.com/huggingface) - Report issues and browse repositories on GitHub

---

## Contributing

Contributions are welcome! Please:

1. Fork this repository
2. Add your resource under the appropriate section
3. Ensure links are active and descriptions are concise
4. Open a Pull Request with a brief description

> **Guidelines:** Resources should be directly related to AWS + Hugging Face integration. Prefer official docs, maintained repos, and high-quality tutorials.

---

*⭐ Star this repo if you find it useful! Maintained with ❤️ by the community.*