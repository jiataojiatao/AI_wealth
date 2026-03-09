from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和分词器
model_id = " Qwen2-0.5B" # 练习建议用小模型，如 Qwen2-0.5B 或 OPT-350M
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # 必须设置 pad_token

# 2. 加载本地数据集
dataset = load_dataset("json", data_files="train_sft.jsonl", split="train")

# 3. 设置训练参数
sft_config = SFTConfig(
    output_dir="./sft_results",
    max_seq_length=512,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    dataset_text_field="messages", # 指向数据集中的对话列
    packing=False # 初学者建议设为 False，更直观
)

# 4. 初始化 Trainer
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

# 5. 开始执行
# trainer.train()