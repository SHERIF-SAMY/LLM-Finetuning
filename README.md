# README.md: News LLM Analyzer Notebook

This notebook demonstrates a comprehensive workflow for fine-tuning a Large Language Model (LLM) for news analysis tasks, including details extraction and translation. It covers data preparation, LLM-based data generation (knowledge distillation), LoRA fine-tuning using LLaMA-Factory, model evaluation, VLLM deployment for optimized inference, and load testing with Locust.

## Table of Contents
1. [Setup and Dependencies](#1-setup-and-dependencies)
2. [Data Preparation and Pydantic Models](#2-data-preparation-and-pydantic-models)
3. [Base Model Loading and Initial Evaluation](#3-base-model-loading-and-initial-evaluation)
4. [Knowledge Distillation (Data Generation for Fine-tuning)](#4-knowledge-distillation-data-generation-for-fine-tuning)
5. [Finetuning Dataset Formatting](#5-finetuning-dataset-formatting)
6. [LLaMA-Factory Integration and Fine-tuning](#6-llama-factory-integration-and-fine-tuning)
7. [New Finetuned Model Evaluation](#7-new-finetuned-model-evaluation)
8. [VLLM Deployment](#8-vllm-deployment)
9. [Inference with VLLM](#9-inference-with-vllm)
10. [Load Testing with Locust](#10-load-testing-with-locust)
11. [Qwen2.5 Chinese Character Tip](#11-qwen25-chinese-character-tip)

---

## 1. Setup and Dependencies
This section handles the initial environment setup, including mounting Google Drive and installing all necessary Python packages. Version pinning is used for critical libraries to ensure reproducibility.

- **Mount Google Drive**: Accesses files stored in your Google Drive.
- **Install Libraries**: Installs `transformers`, `peft`, `accelerate`, `datasets`, `openai`, `wandb`, `groq`, and `json_repair`. Note the specific version requirements and `llamafactory`'s installation.
- **LLaMA-Factory**: Clones the LLaMA-Factory repository and installs it in editable mode. This framework is used for efficient LLM fine-tuning.
- **Hugging Face and W&B Login**: Authenticates with Hugging Face Hub and Weights & Biases for model access and experiment tracking.

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/gdrive')

# Install core dependencies (specific versions)
# !pip install -qU transformers==4.57.1 datasets==3.2.0 optimum==1.24.0
# !pip install -qU openai==1.61.0 wandb
# !pip install groq
# !pip install peft==0.17.1
# !pip install accelerate==1.8.1

# Install json_repair
!pip install json_repair

# Clone and install LLaMA-Factory
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
!cd LLaMA-Factory && pip install -e .

# Login to Hugging Face and Weights & Biases
from google.colab import userdata
import wandb
from huggingface_hub import login

wandb.login(key=userdata.get('wandb'))
hf_token = userdata.get('HF')
login(token=hf_token)
!huggingface-cli whoami
```

## 2. Data Preparation and Pydantic Models
This section defines the structure for extracting and translating news data using Pydantic models. It also sets up the prompting messages for the LLM.

- **Story Content**: A sample Arabic news story is provided as `story`.
- **Pydantic Models**: 
    - `NewsDetails`: Defines the schema for extracting `story_title`, `story_keywords`, `story_summary`, `story_category`, and `story_entities` (with specific `EntityType`).
    - `TranslatedStory`: Defines the schema for `translated_title` and `translated_text` for translation tasks.
- **LLM Prompting Messages**: `details_extraction_messages` and `translated_messages` are created based on these Pydantic schemes to guide the LLM in generating structured JSON output.

```python
# Example of Pydantic models for structured data extraction and translation
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json

# ... (NewsDetails and TranslatedStory Pydantic model definitions as in notebook)

# Example of messages for LLM prompting
details_extraction_messages = [
    # ... (system and user messages for news details extraction)
]

translated_messages = [
    # ... (system and user messages for story translation)
]
```

## 3. Base Model Loading and Initial Evaluation
This part loads a base LLM (Qwen2.5-1.5B-Instruct) and demonstrates how to use it for inference directly.

- **Model Loading**: `AutoModelForCausalLM` and `AutoTokenizer` are used to load the pre-trained `Qwen/Qwen2.5-1.5B-Instruct` model.
- **`generate_resp` Function**: A helper function `generate_resp` is defined to abstract the model inference process, including applying chat templates and decoding responses.
- **`parse_json` Function**: A function using `json_repair` is provided to robustly parse potentially malformed JSON output from the LLM.
- **Initial Inference**: The base model is used to extract details and translate the `story`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda"
torch_dtype = None # or torch.bfloat16 if supported

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map= "auto",
    torch_dtype= torch_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

def generate_resp(messages):
    # ... (implementation as in notebook)
    return response

import json_repair
def parse_json(text):
    # ... (implementation as in notebook)
    return None

# Example usage:
response_details = generate_resp(details_extraction_messages)
parsed_details = parse_json(response_details)

response_translation = generate_resp(translated_messages)
parsed_translation = parse_json(response_translation)
```

## 4. Knowledge Distillation (Data Generation for Fine-tuning)
This section outlines the process of generating a synthetic dataset for fine-tuning using a powerful cloud LLM (Groq's Llama-3.3-70b-versatile). This is a knowledge distillation step to create task-specific training data.

- **Raw Data Loading**: Loads a raw news dataset (`news-sample.jsonl`).
- **Groq API Calls**: Iterates through the raw data, uses the Groq API with the `llama-3.3-70b-versatile` model to perform details extraction and translation, based on the Pydantic schemas defined earlier.
- **Rate Limiting Handling**: Includes `time.sleep` to manage API rate limits.
- **Output Storage**: Saves the generated structured JSON responses into `stf.jsonl` files.
- **Cost Estimation**: Tracks prompt and completion tokens to estimate API costs.

```python
import time
from groq import Groq
from tqdm.auto import tqdm
from os.path import join
import random

# ... (cloud_model_id, price_per_1m_input_tokens, price_per_1m_output_tokens, groq_client initialization)

raw_data_path = join(data_dir, "news-sample.jsonl")
raw_data = []
# ... (load raw_data from jsonl file)

save_to_details = join(data_dir, "stf_details.jsonl")
save_to_translation = join(data_dir, "stf_translation.jsonl")

for story in tqdm(raw_data):
    # Generate details extraction messages and call Groq API
    # ... (similar to E9B9fotGCeHs cell in notebook)
    # Introduce delay: time.sleep(1.5)
    # Parse and save response to stf_details.jsonl

    # Generate translation messages and call Groq API
    # ... (similar to WU8iGDIbabTo cell in notebook)
    # Parse and save response to stf_translation.jsonl

# Print total cost
```

## 5. Finetuning Dataset Formatting
The generated `stf.jsonl` files are then formatted into a structure suitable for LLaMA-Factory's fine-tuning input.

- **System Message**: A generic system message is defined for the fine-tuned model.
- **Data Transformation**: Each record from the `stf.jsonl` is transformed into a dictionary with `system`, `instruction`, `input`, `output`, and `history` fields, aligning with LLaMA-Factory's expected format.
- **Splitting Data**: The formatted data is split into training and validation sets and saved as `train.json` and `val.json` in the specified directory.

```python
sft_data_path = join(data_dir, "stf.jsonl") # Assuming a combined sft.jsonl from previous step
llm_finetuning_data = []

system_message = """You are a professional NLP data parser. ...""" # as defined in notebook

for line in open(sft_data_path):
    # ... (load rec, format into llm_finetuning_data as in notebook)

random.Random(45).shuffle(llm_finetuning_data)

# Split and save to train.json and val.json
# ... (as in Tyufq0twOXCJ cell in notebook)
```

## 6. LLaMA-Factory Integration and Fine-tuning
This section integrates the prepared dataset with LLaMA-Factory for fine-tuning the Qwen2.5 base model using LoRA.

- **Update `dataset_info.json`**: Programmatically adds the paths and column mappings for the `news_finetune_train` and `news_finetune_val` datasets to LLaMA-Factory's configuration.
- **Create `news_finetune.yaml`**: Generates a YAML configuration file that specifies:
    - **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
    - **Finetuning Method**: LoRA (`lora_rank: 64`, `lora_target: all`)
    - **Datasets**: `news_finetune_train` and `news_finetune_val`
    - **Training Parameters**: `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, etc.
    - **Output**: `output_dir` for checkpoints, `report_to: wandb`, and `push_to_hub` settings.
- **Run LLaMA-Factory Train**: Executes the fine-tuning process using `llamafactory-cli train` with the generated YAML configuration.

```python
# Update LLaMA-Factory's dataset_info.json
import json
import os

dataset_info_path = '/content/LLaMA-Factory/data/dataset_info.json'
# ... (code to update dataset_info.json as in df63434c cell)

# Create news_finetune.yaml configuration file
%%writefile /content/LLaMA-Factory/examples/train_lora/news_finetune.yaml
# ... (YAML content as in 1B9O3TrGS11Z cell)

# Run fine-tuning
!cd LLaMA-Factory/ && llamafactory-cli train /content/LLaMA-Factory/examples/train_lora/news_finetune.yaml
```

## 7. New Finetuned Model Evaluation
After fine-tuning, the updated model (base model + LoRA adapter) is loaded and evaluated.

- **Load Base Model**: Reloads the `Qwen/Qwen2.5-1.5B-Instruct` base model.
- **Load Adapter**: Uses `model.load_adapter()` to load the fine-tuned LoRA weights from the `finetuned_model_id` (e.g., from `/gdrive/MyDrive/Fine-tuning LLM/Data/llamafactory-finetune-data/checkpoints`).
- **Inference**: Uses the `generate_resp` function with the loaded adapter to perform inference on new inputs.

```python
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype = torch_dtype
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load fine-tuned adapter
finetuned_model_id = "/gdrive/MyDrive/Fine-tuning LLM/Data/llamafactory-finetune-data/checkpoints"
model.load_adapter(finetuned_model_id)

# Example inference with fine-tuned model
response = generate_resp(translated_messages) # or details_extraction_messages
parsed_response = parse_json(response)
print(parsed_response)
```

## 8. VLLM Deployment
This section demonstrates how to deploy the fine-tuned model with VLLM for high-throughput and low-latency inference.

- **VLLM Serve Command**: Uses `nohup vllm serve` to run a VLLM server in the background, specifying the base model, data types, GPU memory utilization, and enabling LoRA with the fine-tuned adapter.
- **Log Monitoring**: `!tail -n 30 nohup.out` is used to check the VLLM server logs.

```python
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_model_id = "/gdrive/MyDrive/youtube-resources/llm-finetuning/models" # Or your actual checkpoints dir

!nohup vllm serve "{base_model_id}" --dtype=half --gpu-memory-utilization 0.8 --max_lora_rank 64 --enable-lora --lora-modules news-lora="{adapter_model_id}" & # Run in background

# Check logs after a short wait to ensure server starts
# !sleep 30 # wait for server to start
!tail -n 30 nohup.out
```

## 9. Inference with VLLM
Interacting with the locally deployed VLLM server for inference.

- **Tokenizer**: Loads the tokenizer for the base model.
- **Prompt Formatting**: Formats the input messages using `tokenizer.apply_chat_template`.
- **HTTP Request**: Sends a POST request to the VLLM server's `/v1/completions` endpoint with the formatted prompt and model ID.

```python
import requests
from transformers import AutoTokenizer # Assuming it's already imported above

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Example: using translated_messages for inference
prompt_vllm = tokenizer.apply_chat_template(
    translated_messages,
    tokenize=False,
    add_generation_prompt=True
)

vllm_model_id = "news-lora"
llm_response = requests.post("http://localhost:8000/v1/completions", json={
    "model": vllm_model_id,
    "prompt": prompt_vllm,
    "max_tokens": 1000,
    "temperature": 0.3
})

print(llm_response.json())
```

## 10. Load Testing with Locust
This section sets up and runs a load test on the VLLM server using Locust to evaluate its performance under stress.

- **Locust File (`locust.py`)**: A Python script is generated that defines a `HttpUser` class. This class simulates users sending POST requests to the VLLM completion endpoint with dynamically generated prompts (using `faker`).
- **Load Test Execution**: `!locust` command is used to run the load test in headless mode, specifying the host, number of users, spawn rate, test duration, and an HTML report output.
- **Token Calculation**: After the load test, `vllm_tokens.txt` (where responses are saved) is parsed to calculate total input and output tokens processed by VLLM.

```python
# Generate locust.py file
%%writefile locust.py
# ... (locust script content as in TYcuOUtB7q1Y cell)

# Run Locust load test
!locust --headless -f locust.py --host=http://localhost:8000 -u 20 -r 1 -t "60s" --html=locust_results.html

# Process VLLM tokens from load test
vllm_tokens = [
    json.loads(line.strip())
    for line in open("./vllm_tokens.txt") if line.strip() != ""
]

# Calculate and print total input/output tokens
total_input_tokens = sum([ len(tokenizer.encode(rec['prompt'])) for rec in vllm_tokens ])
total_output_tokens = sum([ len(tokenizer.encode(rec['response'])) for rec in vllm_tokens ])

print(f"Total Input Tokens: {total_input_tokens}")
print(f"Total Output Tokens: {total_output_tokens}")
```

## 11. Qwen2.5 Chinese Character Tip
Provides a custom `Generator` class with a `logits_processor` to prevent Qwen2.5 from generating Chinese characters, which can sometimes occur unexpectedly with multilingual models.

```python
# Generator class to ban Chinese characters
import torch
class Generator:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.mask = None

    def generate(self, messages:list, max_new_tokens: int=2000, temperature:float=0.1):
        def logits_processor(token_ids, logits):
            # ... (implementation as in ungxEdlh_chB cell)
            return logits
        
        # ... (rest of the generate method using logits_processor)
        return response

# Example usage:
# my_generator = Generator(model, tokenizer)
# response_no_chinese = my_generator.generate(messages)
```
