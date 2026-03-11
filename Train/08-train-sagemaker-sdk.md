<div align="center">

# Train with Amazon SageMaker SDK

*Fine-tune Hugging Face models using the SageMaker Python SDK with managed infrastructure, spot instance savings, and one-click deployment after training.*

[![Back to Overview](https://img.shields.io/badge/←%20Train-Overview-blue)](./07-train-overview.md)
[![Official Docs](https://img.shields.io/badge/HF-Official%20Docs-yellow)](https://huggingface.co/docs/sagemaker/en/train)

</div>

---

## Overview

Amazon SageMaker handles the entire training lifecycle: provisioning instances, loading data from S3, running your training script inside a Hugging Face DLC container, and saving the model back to S3 — all fully managed. You only write a `train.py` script; SageMaker handles the rest.

**Why SageMaker for Training?**
- Training instances live **only for the duration of the job** — pay per second, no idle costs
- Supports **EC2 Spot instances** (up to 90% cost savings)
- Integrated with **SageMaker Distributed Training** libraries for multi-GPU/multi-node
- Built-in experiment tracking, metrics, and debugging
- Direct handoff to deployment after training

---

## Prerequisites

```bash
pip install "sagemaker<3.0.0" transformers datasets torch accelerate peft

aws configure  # Set up AWS credentials
```

You need:
- AWS account + IAM role with `AmazonSageMakerFullAccess`
- An S3 bucket for storing datasets and model artifacts

---

## Step 1: Prepare Your Training Script

Create a `train.py` script using the Hugging Face `Trainer` API:

```python
# train.py
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized = dataset.map(tokenize, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
```

> **SageMaker Environment Variables:** `SM_MODEL_DIR` is the path where your model is saved (synced to S3 after training). `SM_CHANNEL_TRAIN` is the path to the training data.

---

## Step 2: Launch a SageMaker Training Job

```python
import sagemaker
from sagemaker.huggingface import HuggingFace

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define hyperparameters
hyperparameters = {
    "model_name_or_path": "distilbert-base-uncased",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "learning_rate": 5e-5,
}

# Create HuggingFace Estimator
huggingface_estimator = HuggingFace(
    entry_point="train.py",          # Your training script
    source_dir="./scripts",          # Directory containing train.py
    instance_type="ml.p3.2xlarge",   # GPU instance
    instance_count=1,
    role=role,
    transformers_version="4.36",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters=hyperparameters,
)

# Start training (data loaded from S3)
huggingface_estimator.fit({
    "train": "s3://your-bucket/imdb/train",
    "test": "s3://your-bucket/imdb/test"
})

print(f"Training job: {huggingface_estimator.latest_training_job.name}")
```

---

## Step 3: Deploy After Training

```python
# One line to deploy your fine-tuned model
predictor = huggingface_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Test
response = predictor.predict({"inputs": "This movie was absolutely fantastic!"})
print(response)  # [{"label": "POSITIVE", "score": 0.998}]

predictor.delete_endpoint()
```

---

## Cost Optimization: Spot Instances (Up to 90% Savings)

```python
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./scripts",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.36",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters=hyperparameters,
    # Enable managed spot training
    use_spot_instances=True,
    max_run=3600,          # Max 1 hour training time
    max_wait=7200,         # Max 2 hours wait (including interruptions)
    checkpoint_s3_uri="s3://your-bucket/checkpoints/"  # Required for spot
)

huggingface_estimator.fit({"train": training_input})
```

---

## Distributed Training (Multi-GPU / Multi-Node)

### Data Parallelism (SageMaker Distributed Data Parallel)

```python
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./scripts",
    instance_type="ml.p3.16xlarge",  # 8 GPUs
    instance_count=2,                # 2 nodes = 16 GPUs total
    role=role,
    transformers_version="4.36",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters=hyperparameters,
    distribution={
        "smdistributed": {
            "dataparallel": {"enabled": True}
        }
    }
)
```

### MPI-based Distribution (DeepSpeed)

```python
distribution = {
    "mpi": {
        "enabled": True,
        "processes_per_host": 8,
        "custom_mpi_options": "--NCCL_DEBUG INFO"
    }
}

huggingface_estimator = HuggingFace(
    ...
    distribution=distribution
)
```

---

## Fine-Tuning with LoRA / QLoRA (PEFT)

For large models, use parameter-efficient fine-tuning to reduce memory usage:

```python
# peft_train.py (excerpt)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

```python
# Launch on SageMaker with a G5 instance for LLM QLoRA
huggingface_estimator = HuggingFace(
    entry_point="peft_train.py",
    instance_type="ml.g5.2xlarge",   # A10G GPU, 24GB VRAM
    instance_count=1,
    role=role,
    transformers_version="4.36",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters={
        "model_name_or_path": "meta-llama/Meta-Llama-3-8B",
        "HUGGING_FACE_HUB_TOKEN": "<YOUR_TOKEN>",
        "num_train_epochs": 1,
    }
)
```

---

## Train on AWS Trainium (Up to 50% Lower Cost)

Use `optimum-neuron` to train on AWS Trainium instances:

```bash
pip install optimum[neuronx]
```

```python
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    instance_type="ml.trn1.2xlarge",  # Trainium instance
    instance_count=1,
    role=role,
    transformers_version="4.36",
    pytorch_version="2.1.0",
    py_version="py310",
    hyperparameters=hyperparameters,
    environment={"NEURON_RT_NUM_CORES": "2"}
)
```

---

## Load Training Script from GitHub

You can load training scripts directly from a GitHub repo:

```python
huggingface_estimator = HuggingFace(
    entry_point="examples/text-classification/run_glue.py",
    source_dir=".",
    git_config={
        "repo": "https://github.com/huggingface/transformers.git",
        "branch": "main"
    },
    instance_type="ml.p3.2xlarge",
    ...
)
```

---

## Monitor Training Metrics

```python
# Define custom metric regexes to capture from logs
metric_definitions = [
    {"Name": "train:loss", "Regex": "'loss': ([0-9\\.]+)"},
    {"Name": "eval:accuracy", "Regex": "'eval_accuracy': ([0-9\\.]+)"},
]

huggingface_estimator = HuggingFace(
    ...
    metric_definitions=metric_definitions
)

# After training, view metrics
training_job_name = huggingface_estimator.latest_training_job.name
# View in: SageMaker Console → Training Jobs → [job name] → Metrics
```

---

## Useful Resources
## Useful Resources

- [HF SageMaker Train Docs](https://huggingface.co/docs/sagemaker/en/train) - Hugging Face training docs for SageMaker
- [Optimum Neuron (Trainium)](https://huggingface.co/docs/optimum-neuron) - Optimum Neuron docs for Trainium
- [PEFT Library](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning (PEFT) docs
- [SageMaker SDK Quickstart](https://huggingface.co/docs/sagemaker/en/tutorials/sagemaker-sdk/sagemaker-sdk-quickstart) - Quickstart for the SageMaker SDK
- [SageMaker Spot Training Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html) - SageMaker managed spot training documentation
- [TRL (RLHF / SFT) Library](https://huggingface.co/docs/trl) - TRL library docs for RLHF / SFT