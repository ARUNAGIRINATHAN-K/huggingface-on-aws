<div align="center">

# Deploy with AWS Bedrock Marketplace

*Deploy open Hugging Face models via Amazon Bedrock Marketplace and invoke them using Bedrock's unified API including Agents, Knowledge Bases, and Guardrails.*

[![Back to Overview](https://img.shields.io/badge/←%20Deploy-Overview-blue)](./01-deploy-overview.md)
[![Official Docs](https://img.shields.io/badge/HF-Official%20Docs-yellow)](https://huggingface.co/docs/sagemaker/en/tutorials/bedrock/bedrock-quickstart)

</div>

---

## Overview

Amazon Bedrock enables developers to build and scale generative AI applications through a **single unified API**. With **Bedrock Marketplace**, Hugging Face open models are deployed via SageMaker JumpStart infrastructure under the hood, but invoked through Bedrock's high-level APIs including Agents, Knowledge Bases, Guardrails, and Model Evaluations.

As of early 2026, Bedrock Marketplace supports **83+ Hugging Face models**, including Llama 3, Mistral, Gemma, and Falcon.

**When to use this path:**
- You want to build GenAI apps using Bedrock's high-level APIs
- You need AWS-native Agents, Knowledge Bases (RAG), or Guardrails
- You want to standardize inference code across multiple model providers
- You're building in a compliance-sensitive environment (HIPAA, FedRAMP)

---

## Prerequisites

```bash
pip install boto3 "sagemaker<3.0.0"
aws configure
```

You need:
- AWS account with Bedrock access enabled in your region
- IAM role with `AmazonBedrockFullAccess` and `AmazonSageMakerFullAccess`
- A deployed SageMaker JumpStart endpoint (see [JumpStart guide](./03-deploy-sagemaker-jumpstart.md))

---

## Step 1: Deploy a Model via Bedrock Marketplace (Console)

1. Open the **Amazon Bedrock console**
2. Navigate to **Bedrock Marketplace** (or **Model Catalog**)
3. Search for a Hugging Face model (e.g., `Llama 3 8B Instruct`)
4. Click the model → **Subscribe** (if required) → **Deploy**
5. Configure instance type and endpoint name
6. Click **Deploy** — this creates a SageMaker JumpStart endpoint behind the scenes
7. Once `InService`, copy the **Endpoint ARN**

---

## Step 2: Query the Model with the Bedrock Converse API

```python
import boto3

# Initialize Bedrock Runtime client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Use the SageMaker endpoint ARN as the modelId
endpoint_arn = "arn:aws:sagemaker:<region>:<account-id>:endpoint/<endpoint-name>"

# Build inference config
inference_config = {
    "maxTokens": 256,
    "temperature": 0.1,
    "topP": 0.95
}

additional_fields = {
    "parameters": {
        "repetition_penalty": 1.05
    }
}

# Send request using the Converse API
response = bedrock.converse(
    modelId=endpoint_arn,
    messages=[{
        "role": "user",
        "content": [{"text": "Explain the difference between RAG and fine-tuning."}]
    }],
    inferenceConfig=inference_config,
    additionalModelRequestFields=additional_fields
)

print(response["output"]["message"]["content"][0]["text"])
```

> **Note:** The same `modelId=endpoint_arn` pattern works with `InvokeModel`, `Knowledge Bases (RetrieveAndGenerate)`, `Agents`, and `Guardrails` — no code changes needed.

---

## Use with InvokeModel API

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime")
endpoint_arn = "arn:aws:sagemaker:<region>:<account-id>:endpoint/<name>"

body = json.dumps({
    "inputs": "What are the benefits of transformer architectures?",
    "parameters": {"max_new_tokens": 200, "temperature": 0.7}
})

response = bedrock.invoke_model(
    modelId=endpoint_arn,
    body=body,
    contentType="application/json",
    accept="application/json"
)

result = json.loads(response["body"].read())
print(result)
```

---

## Use with Knowledge Bases (RAG)

Bedrock Knowledge Bases allow you to connect your HF model to a vector store for retrieval-augmented generation:

```python
import boto3

bedrock_agent = boto3.client("bedrock-agent-runtime")

response = bedrock_agent.retrieve_and_generate(
    input={"text": "What does our company policy say about remote work?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "<YOUR_KB_ID>",
            "modelArn": endpoint_arn  # Your HF model endpoint ARN
        }
    }
)

print(response["output"]["text"])
```

---

## Use with Bedrock Guardrails

Apply content filtering, PII redaction, and topic blocking to your HF model:

```python
response = bedrock.converse(
    modelId=endpoint_arn,
    messages=[{
        "role": "user",
        "content": [{"text": "Tell me about our product pricing."}]
    }],
    inferenceConfig={"maxTokens": 300},
    guardrailConfig={
        "guardrailIdentifier": "<GUARDRAIL_ID>",
        "guardrailVersion": "DRAFT",
        "trace": "enabled"
    }
)
```

---

## Supported Models on Bedrock Marketplace

| Model | Provider | Use Case |
|-------|----------|----------|
| Meta Llama 3 8B / 70B Instruct | Meta | Chat, reasoning |
| Mistral 7B / Mixtral 8x7B | Mistral AI | Chat, code |
| Gemma 2 27B Instruct | Google | Chat, multilingual |
| Falcon 7B / 40B | TII | Text generation |
| StarCoder 2 | HF + ServiceNow | Code generation |
| BGE Embeddings | BAAI | Semantic search, RAG |

---

## Pricing

- You pay **SageMaker compute costs** for the underlying endpoint
- **Standard Bedrock API pricing** applies for API calls (tokens in/out)
- No additional Marketplace fees for Hugging Face models
- Bedrock Agents and Knowledge Bases have their own per-request pricing

---

## When NOT to Use Bedrock

| Scenario | Better Option |
|----------|--------------|
| Deploying a custom fine-tuned model | [SageMaker SDK](./02-deploy-sagemaker-sdk.md) |
| Need full infrastructure control | [ECS/EKS/EC2](./06-deploy-ecs-eks-ec2.md) |
| Lowest latency, no Bedrock overhead | [SageMaker SDK](./02-deploy-sagemaker-sdk.md) |
| Model not in Bedrock Marketplace catalog | [JumpStart](./03-deploy-sagemaker-jumpstart.md) |

---

## Useful Resources

- [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock) - Bedrock console for managing marketplace deployments
- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference) - Bedrock API reference documentation
- [Bedrock Guardrails Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) - Bedrock guardrails and content controls
- [HF Bedrock Blog Post](https://huggingface.co/blog/bedrock-marketplace) - Blog post about Bedrock Marketplace and Hugging Face models
- [HF Bedrock Quickstart](https://huggingface.co/docs/sagemaker/en/tutorials/bedrock/bedrock-quickstart) - Hugging Face quickstart for Bedrock Marketplace