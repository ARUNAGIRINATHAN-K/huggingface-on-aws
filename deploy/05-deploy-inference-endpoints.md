<div align="center">

# Deploy with Hugging Face Inference Endpoints

*Deploy any model from the Hugging Face Hub as a fully managed, production-ready endpoint on AWS infrastructure in minutes.*

[![Back to Overview](https://img.shields.io/badge/←%20Deploy-Overview-blue)](./01-deploy-overview.md)
[![Official Site](https://img.shields.io/badge/HF-Inference%20Endpoints-yellow)](https://endpoints.huggingface.co)

</div>

---

## Overview

Hugging Face Inference Endpoints is a **fully managed service hosted by Hugging Face** that deploys models on AWS infrastructure (or other clouds) without you needing to manage any AWS resources directly. It is the fastest path from a model on the Hub to a live production endpoint.

Inference Endpoints supports:
- **225,000+ text generation models** via TGI (Text Generation Inference)
- **12,000+ embedding / re-ranking models** via TEI (Text Embeddings Inference)
- **AWS Inferentia2** for cost-efficient LLM inference
- **Scale-to-zero** for development and cost savings
- **OpenAI-compatible Messages API** — drop-in replacement for OpenAI SDK

**When to use this path:**
- You want the fastest time to a production endpoint
- You don't want to manage AWS infrastructure at all
- You need OpenAI-compatible API for an existing app
- You want to test a model quickly before committing to SageMaker

---

## Prerequisites

- A Hugging Face account at [huggingface.co](https://huggingface.co)
- Billing configured on your HF account
- A Hugging Face API token (`HF_TOKEN`)

---

## Option 1: Deploy via the Hugging Face Hub (UI)

1. Go to any model page on [hf.co/models](https://huggingface.co/models)
2. Click **Deploy** → **Inference Endpoints**
3. You'll be redirected to the Inference Endpoints dashboard
4. Select your cloud provider: **Amazon Web Services**
5. Choose your AWS region and hardware:
   - CPU: from `$0.03/hr`
   - GPU (T4): from `$0.50/hr`
   - Inferentia2 (`inf2-small`): `$0.75/hr`
   - Inferentia2 (`inf2-xlarge`): `$12/hr`
6. Configure scaling (including **scale to zero**)
7. Click **Create Endpoint** — your endpoint is ready in minutes

---

## Option 2: Deploy via the `huggingface_hub` Python SDK

```python
from huggingface_hub import HfApi

api = HfApi(token="<YOUR_HF_TOKEN>")

# Create an Inference Endpoint
endpoint = api.create_inference_endpoint(
    name="my-llama3-endpoint",
    repository="meta-llama/Meta-Llama-3-8B-Instruct",
    framework="pytorch",
    accelerator="gpu",
    instance_size="x2",
    instance_type="nvidia-l4",
    region="us-east-1",
    vendor="aws",
    min_replica=0,   # Scale to zero when idle
    max_replica=2,   # Max 2 replicas under load
    task="text-generation",
)

print(endpoint.url)  # e.g., https://xyz.us-east-1.aws.endpoints.huggingface.cloud
```

---

## Option 3: Deploy on AWS Inferentia2

Inferentia2 instances provide up to 40% cost savings over equivalent GPU instances:

```python
endpoint = api.create_inference_endpoint(
    name="llama3-inferentia2",
    repository="meta-llama/Meta-Llama-3-8B-Instruct",
    framework="pytorch",
    accelerator="inf2",
    instance_size="small",   # inf2-small: 2 cores, 32GB — $0.75/hr
    region="us-east-1",
    vendor="aws",
    task="text-generation",
)
```

**Available Inferentia2 flavors:**
| Flavor | Cores | Memory | Price | Best For |
|--------|-------|--------|-------|----------|
| `inf2-small` | 2 | 32 GB | $0.75/hr | Llama 3 8B, Mistral 7B |
| `inf2-xlarge` | 24 | 384 GB | $12/hr | Llama 3 70B |

---

## Sending Requests to Your Endpoint

### Using the HF Hub SDK

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="https://<endpoint-url>",
    api_key="<YOUR_HF_TOKEN>"
)

# Text generation
response = client.text_generation(
    "What is the capital of France?",
    max_new_tokens=100
)
print(response)
```

### Using the OpenAI SDK (Messages API)

Endpoints are compatible with the OpenAI Messages API — no code changes needed for existing apps:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<endpoint-url>/v1",
    api_key="<YOUR_HF_TOKEN>"
)

response = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in 3 sentences."}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)
```

### Using `requests`

```python
import requests

API_URL = "https://<endpoint-url>"
headers = {"Authorization": f"Bearer <YOUR_HF_TOKEN>"}

payload = {
    "inputs": "Once upon a time",
    "parameters": {"max_new_tokens": 100, "temperature": 0.8}
}

response = requests.post(API_URL, headers=headers, json=payload)
print(response.json())
```

---

## Embedding Models with TEI

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="https://<tei-endpoint-url>",
    api_key="<YOUR_HF_TOKEN>"
)

# Get text embeddings
embeddings = client.feature_extraction(
    text=["Hello world", "This is a test"],
    normalize=True
)

print(f"Embedding shape: {len(embeddings[0])} dimensions")
```

---

## Auto-scaling and Scale-to-Zero

Inference Endpoints supports automatic scaling based on traffic:

```python
# Enable scale-to-zero (scales down to 0 replicas when idle)
endpoint = api.create_inference_endpoint(
    name="my-endpoint",
    repository="mistralai/Mistral-7B-Instruct-v0.3",
    ...
    min_replica=0,   # Allows scaling to zero
    max_replica=3,   # Max replicas under heavy load
)
```

> Scale-to-zero endpoints have a cold start time of ~30–60 seconds. Use `min_replica=1` for production workloads requiring consistent latency.

---

## Comparison: Inference Endpoints vs SageMaker

| Feature | HF Inference Endpoints | SageMaker SDK |
|---------|----------------------|---------------|
| Managed by | Hugging Face | AWS (you configure) |
| Setup complexity | Very low | Medium |
| Custom inference code | Limited | Full support |
| VPC / network isolation | Limited | Full support |
| AWS IAM integration | No | Yes |
| OpenAI-compatible API | ✅ Built-in | Manual setup |
| Scale to zero | ✅ Built-in | Serverless only |
| Best for | Fast prototyping, SaaS | Enterprise, custom infra |

---

## Useful Resources

- [HF Hub SDK Docs](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints) - Hugging Face Hub SDK documentation for Inference Endpoints
- [Inferentia2 on Inference Endpoints](https://huggingface.co/blog/inferentia-inference-endpoints) - Blog post on Inferentia2 support for Inference Endpoints
- [Inference Endpoints Dashboard](https://endpoints.huggingface.co) - Hugging Face Inference Endpoints dashboard
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference) - Text Generation Inference documentation (TGI)
- [TEI Documentation](https://huggingface.co/docs/text-embeddings-inference) - Text Embeddings Inference documentation (TEI)