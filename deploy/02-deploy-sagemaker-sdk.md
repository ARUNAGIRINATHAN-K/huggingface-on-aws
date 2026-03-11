<div align="center">

# Deploy with Amazon SageMaker SDK

*Deploy pre-trained or fine-tuned Hugging Face models as managed, scalable SageMaker endpoints using the SageMaker Python SDK.*

[![Back to Overview](https://img.shields.io/badge/←%20Deploy-Overview-blue)](./01-deploy-overview.md)
[![Official Docs](https://img.shields.io/badge/HF-Official%20Docs-yellow)](https://huggingface.co/docs/sagemaker/inference)

</div>

---


Amazon SageMaker is a fully managed AWS service for building, training, and deploying ML models at scale. The SageMaker SDK provides a seamless, purpose-built integration for Hugging Face models you can deploy both your own fine-tuned models and any of the 1M+ models from the Hugging Face Hub with minimal code.

**When to use this path:**
- You have a fine-tuned model saved in S3 and want to serve it
- You want programmatic control over endpoint configuration
- You need SageMaker enterprise features (monitoring, autoscaling, VPC, logging)
- You want to deploy directly after a SageMaker training job

---

## Prerequisites

```bash
# Install dependencies
pip install "sagemaker<3.0.0" transformers torch boto3

# Configure AWS CLI
aws configure
```

You also need an IAM role with `AmazonSageMakerFullAccess` permissions.

---

## Option 1: Deploy a Model from the Hugging Face Hub

Deploy any public model directly from the Hub by specifying the `model_id` and `task`:

```python
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

# Set up SageMaker session
sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define Hub model and task
hub = {
    'HF_MODEL_ID': 'distilbert-base-uncased-distilled-squad',  # Model ID from hf.co/models
    'HF_TASK': 'question-answering'                            # Inference task
}

# Create HuggingFaceModel
huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
)

# Deploy to SageMaker endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Run inference
response = predictor.predict({
    "inputs": {
        "question": "What is SageMaker?",
        "context": "Amazon SageMaker is a fully managed ML service by AWS."
    }
})
print(response)

# Clean up
predictor.delete_endpoint()
```

---

## Option 2: Deploy a Fine-tuned Model from S3

If you've already trained and saved a model to S3:

```python
from sagemaker.huggingface.model import HuggingFaceModel

# Path to your model artifacts in S3
# Must contain: model weights + tokenizer files, packaged as model.tar.gz
model_s3_uri = 's3://your-bucket/your-model/model.tar.gz'

huggingface_model = HuggingFaceModel(
    model_data=model_s3_uri,    # S3 path to your model.tar.gz
    role=role,
    transformers_version='4.36',
    pytorch_version='2.1.0',
    py_version='py310',
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.2xlarge'  # Use GPU for larger models
)

# Inference
data = {"inputs": "Translate this text to French: Hello, world!"}
response = predictor.predict(data)
print(response)

predictor.delete_endpoint()
```

---

## Option 3: Deploy Directly After Training

After a SageMaker training job, call `.deploy()` on the estimator:

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='./scripts',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    role=role,
    transformers_version='4.36',
    pytorch_version='2.1.0',
    py_version='py310',
    hyperparameters={
        'model_name_or_path': 'distilbert-base-uncased',
        'num_train_epochs': 3,
    }
)

huggingface_estimator.fit({'train': 's3://your-bucket/train'})

# Deploy immediately after training
predictor = huggingface_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
```

---

## Option 4: Deploy an LLM with TGI (Text Generation Inference)

For large language models, use the TGI DLC for optimized throughput and streaming:

```python
import json
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
region = sess.boto_region_name

# TGI DLC image URI
tgi_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:2.4-tgi2.3-gpu-py311-cu124-ubuntu22.04"

hub = {
    'HF_MODEL_ID': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'SM_NUM_GPUS': json.dumps(1),
    'HUGGING_FACE_HUB_TOKEN': '<YOUR_HF_TOKEN>',  # Required for gated models
    'MAX_INPUT_LENGTH': json.dumps(4096),
    'MAX_TOTAL_TOKENS': json.dumps(8192),
    'MESSAGES_API_ENABLED': json.dumps(True)
}

huggingface_model = HuggingFaceModel(
    image_uri=tgi_image,
    env=hub,
    role=role,
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g5.2xlarge',
    container_startup_health_check_timeout=300,
    endpoint_name='llama3-8b-tgi-endpoint'
)

# Chat completion request
response = predictor.predict({
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ]
})
print(response)

predictor.delete_endpoint()
```

---

## Endpoint Configuration Options

```python
# Serverless Inference (pay-per-use, scales to zero)
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=10
)

predictor = huggingface_model.deploy(serverless_inference_config=serverless_config)

# Async Inference (for long-running requests)
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path='s3://your-bucket/async-output/'
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.2xlarge',
    async_inference_config=async_config
)
```

---

## Custom Inference Code

Override the default inference pipeline by providing an `inference.py` script:

```python
# inference.py
from transformers import pipeline

def model_fn(model_dir):
    """Load model from model_dir."""
    return pipeline("text-classification", model=model_dir)

def predict_fn(data, model):
    """Run inference."""
    inputs = data.pop("inputs", data)
    return model(inputs)
```

```python
# In your deploy script, point to the custom script
huggingface_model = HuggingFaceModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    entry_point='inference.py',  # Custom inference script
    source_dir='./code',
    transformers_version='4.36',
    pytorch_version='2.1.0',
    py_version='py310',
)
```

---

## HF_TASK Reference

| Task | `HF_TASK` Value |
|------|----------------|
| Text Classification | `text-classification` |
| Token Classification | `token-classification` |
| Question Answering | `question-answering` |
| Text Generation | `text-generation` |
| Summarization | `summarization` |
| Translation | `translation` |
| Fill Mask | `fill-mask` |
| Feature Extraction | `feature-extraction` |
| Image Classification | `image-classification` |
| Object Detection | `object-detection` |

---

## Useful Resources

- [AWS SageMaker HF Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html) - Official AWS docs for Hugging Face DLCs on SageMaker
- [Available DLC Images](https://huggingface.co/docs/sagemaker/dlcs) - Hugging Face DLC images and tags (docs)
- [HF SageMaker Inference Docs](https://huggingface.co/docs/sagemaker/inference) - Hugging Face documentation for SageMaker inference
- [HF SageMaker Notebooks](https://github.com/huggingface/notebooks/tree/main/sagemaker) - Example notebooks demonstrating SageMaker workflows (GitHub)
- [SageMaker Python SDK Docs](https://sagemaker.readthedocs.io) - SageMaker Python SDK documentation