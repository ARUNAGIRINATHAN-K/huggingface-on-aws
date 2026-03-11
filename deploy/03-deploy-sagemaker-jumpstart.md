<div align="center">

# Deploy with Amazon SageMaker JumpStart

*One-click deployment of popular Hugging Face foundation models directly inside your AWS VPC using SageMaker JumpStart.*

[![Back to Overview](https://img.shields.io/badge/←%20Deploy-Overview-blue)](./01-deploy-overview.md)
[![Official Docs](https://img.shields.io/badge/HF-Official%20Docs-yellow)](https://huggingface.co/docs/sagemaker/en/tutorials/jumpstart/jumpstart-quickstart)

</div>

---

## Overview

Amazon SageMaker JumpStart is a curated model catalog that lets you deploy the most popular open Hugging Face models **with one click** — inside your own AWS account. Hugging Face maintains a dedicated section of the JumpStart catalog featuring models like Meta Llama 3, Mistral, Falcon 2, and StarCoder, all pre-configured with optimal instance types and Hugging Face DLC containers.

**When to use this path:**
- You want the fastest possible path from model selection to running endpoint
- You don't want to write deployment code
- You need network-isolated (VPC) deployment of popular open models
- You want sensible default performance configurations out of the box

---

## Prerequisites

```bash
pip install "sagemaker<3.0.0"
```

You need an AWS account, IAM role with SageMaker permissions, and access to the SageMaker console or a SageMaker Notebook Instance.

---

## Option 1: Deploy via the SageMaker Console (No Code)

1. Open the **Amazon SageMaker console**
2. In the left sidebar, navigate to **JumpStart → Models**
3. Search for or browse to the **Hugging Face** section
4. Click a model (e.g., `Llama 3 8B Instruct`)
5. Click **Deploy** → review configuration → click **Deploy**
6. Wait for the endpoint status to become `InService` (typically 5–15 min)

> **Tip:** You can also find JumpStart-eligible models directly on the Hugging Face Hub. Open a model page → click **Deploy** → **Amazon SageMaker** → **JumpStart tab** to get a ready-to-use code snippet.

---

## Option 2: Deploy via the SageMaker Python SDK

```python
from sagemaker.jumpstart.model import JumpStartModel
import sagemaker

role = sagemaker.get_execution_role()

# Initialize a JumpStart model by its model_id
model = JumpStartModel(
    model_id="huggingface-llm-qwen2-5-14b-instruct",
    role=role
)

# Retrieve example payloads for this model
example_payloads = model.retrieve_all_examples()

# Deploy the model (takes several minutes depending on model size)
predictor = model.deploy()

# Run inference with example payloads
for payload in example_payloads:
    response = predictor.predict(payload.body)
    print("Input:\n", payload.body[payload.prompt_key])
    print("Output:\n", response[0]["generated_text"])
    print("=" * 50)

# Clean up when done
predictor.delete_endpoint()
```

---

## Option 3: Deploy via the Hugging Face Hub

Navigate to any JumpStart-eligible model on the Hub:

```
1. Go to https://huggingface.co/models
2. Find your model (e.g., meta-llama/Meta-Llama-3-8B-Instruct)
3. Click the "Deploy" button
4. Select "Amazon SageMaker" → "JumpStart" tab
5. Copy the generated code snippet
6. Paste into a SageMaker Notebook and run
```

---

## Popular Model IDs for JumpStart

| Model | JumpStart Model ID |
|-------|--------------------|
| Meta Llama 3 8B Instruct | `huggingface-llm-meta-llama-3-8b-instruct` |
| Meta Llama 3 70B Instruct | `huggingface-llm-meta-llama-3-70b-instruct` |
| Mistral 7B Instruct | `huggingface-llm-mistral-7b-instruct` |
| Falcon 7B Instruct | `huggingface-llm-falcon-7b-instruct-bf16` |
| Qwen 2.5 14B Instruct | `huggingface-llm-qwen2-5-14b-instruct` |
| StarCoder 2 15B | `huggingface-llm-starcoder2-15b` |
| BGE Large Embeddings | `huggingface-sentencesimilarity-bge-large-en` |

> Browse the full catalog at: [SageMaker JumpStart Foundation Models](https://aws.amazon.com/sagemaker/jumpstart/getting-started/)

---

## Connect a JumpStart Endpoint to Bedrock

After deployment, you can register the endpoint in Amazon Bedrock Marketplace to use it with Bedrock's high-level APIs (Agents, Guardrails, Knowledge Bases):

```python
import boto3

# Get the endpoint ARN from the SageMaker console or SDK
endpoint_arn = "arn:aws:sagemaker:<region>:<account-id>:endpoint/<endpoint-name>"

# Use with Bedrock's converse API
bedrock = boto3.client("bedrock-runtime")

response = bedrock.converse(
    modelId=endpoint_arn,
    messages=[{
        "role": "user",
        "content": [{"text": "Explain quantum computing in simple terms."}]
    }],
    inferenceConfig={"maxTokens": 512, "temperature": 0.7}
)

print(response["output"]["message"]["content"][0]["text"])
```

---

## Deploy on Trainium / Inferentia2 via JumpStart

JumpStart supports deploying supported models on AWS custom silicon for lower cost:

```python
from sagemaker.jumpstart.model import JumpStartModel

# Deploy Llama 3 on Inferentia2 for cost-efficient inference
model = JumpStartModel(
    model_id="huggingface-llm-meta-llama-3-8b-instruct",
    instance_type="ml.inf2.xlarge"  # Inferentia2 instance
)

predictor = model.deploy()
```

---

## Key Behaviors

- **Default instance type** is automatically selected for each model based on its size and architecture
- Models run inside your **own VPC** — data never leaves your AWS account
- All endpoints support **auto-scaling** via SageMaker Application Auto Scaling
- Endpoints can be **monitored** via CloudWatch metrics out of the box
- JumpStart uses Hugging Face **DLCs under the hood** — fully maintained and updated

---

## Useful Resources

- [AWS JumpStart Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models.html) - AWS documentation for SageMaker JumpStart foundation models
- [JumpStart Model IDs Reference](https://sagemaker.readthedocs.io/en/stable/doc_utils/jumpstart.html) - Reference of JumpStart model identifiers (ReadTheDocs)
- [JumpStart Quickstart (HF Docs)](https://huggingface.co/docs/sagemaker/en/tutorials/jumpstart/jumpstart-quickstart) - Hugging Face tutorial for JumpStart quickstart
- [SageMaker JumpStart Console](https://console.aws.amazon.com/sagemaker/home#/jumpstart) - JumpStart model catalog in the SageMaker console