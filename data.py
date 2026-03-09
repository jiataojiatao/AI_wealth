from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和分词器
model_id = " Qwen2-0.5B" # 练习建议用小模型，如 Qwen2-0.5B 或 OPT-350M
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # 必须设置 pad_token

# 2. 加载本地数据集
dataset = load_dataset("json", data_files="train_sft.jsonl", split="train")
