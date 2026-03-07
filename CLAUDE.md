# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project focuses on Reinforcement Learning from Human Feedback (RLHF) and Supervised Fine-Tuning (SFT) using Hugging Face's TRL (Transformer Reinforcement Learning) library.

## Key Dependencies

- `trl`: Hugging Face's Transformer Reinforcement Learning library
- `transformers`: Hugging Face transformers for model loading
- `datasets`: Dataset handling and processing
- `accelerate`: Distributed training support
- `bitsandbytes`: Quantization for efficient training
- `peft`: Parameter-efficient fine-tuning (LoRA, etc.)

## Dataset Format

SFT datasets use JSONL format with the following structure:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Common Commands

### Install Dependencies
```bash
pip install trl transformers datasets accelerate bitsandbytes peft
```

### Run SFT Training
```bash
python sft_train.py
```

## Architecture Notes

- `sft_dataset/`: Contains training data in JSONL format
- `sft_train.py`: Main training script for supervised fine-tuning