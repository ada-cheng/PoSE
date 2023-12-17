from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from huggingface_hub import login
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda"
name = "meta-llama/Llama-2-7b-hf"
config=AutoConfig.from_pretrained(name,cache_dir="./llama2_7b")
model=AutoModelForCausalLM.from_pretrained(name,cache_dir="./llama2_7b")
model.save_pretrained("./llama2_7b")