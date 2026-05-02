"""
finetune.py — Fine-tune a small LLM on research paper QA pairs using QLoRA.

This script:
  1. Prepares a QA dataset (from JSON or auto-generated from chunks)
  2. Applies 4-bit quantization (QLoRA / bitsandbytes)
  3. Attaches LoRA adapters (rank-8, target: q_proj, v_proj)
  4. Trains using TRL's SFTTrainer
  5. Saves adapter weights to models/lora_adapter/

Usage:
    python src/finetune.py
    python src/finetune.py --dataset_path data/qa_pairs.json --epochs 3
"""

import argparse
import json
from pathlib import Path

from loguru import logger
from src.utils import cfg


# Dataset

SAMPLE_QA_PAIRS = [
    {
        "question": "What is the self-attention mechanism?",
        "context":  "Self-attention allows the model to weigh the importance of different words "
                    "in a sequence relative to each other when encoding a representation.",
        "answer":   "Self-attention is a mechanism that computes a weighted sum of all positions "
                    "in a sequence, allowing the model to capture dependencies regardless of distance.",
    },
    {
        "question": "What does BERT stand for?",
        "context":  "BERT stands for Bidirectional Encoder Representations from Transformers, "
                    "introduced by Devlin et al. (2019) at Google.",
        "answer":   "BERT stands for Bidirectional Encoder Representations from Transformers.",
    },
    {
        "question": "What is the key contribution of the ResNet paper?",
        "context":  "ResNet introduced residual connections (skip connections) that allow gradients "
                    "to flow directly through layers, enabling training of very deep networks.",
        "answer":   "ResNet's key contribution is the residual (skip) connection, which solves the "
                    "vanishing gradient problem and enables training of networks with hundreds of layers.",
    },
]

def load_dataset(dataset_path: str | None) -> list[dict]:
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} QA pairs from '{dataset_path}'")
        return data

    logger.warning("No dataset path provided. Using built-in sample QA pairs.")
    return SAMPLE_QA_PAIRS

def format_chat_prompt(sample: dict) -> str:
    """
    Format a QA pair into the TinyLlama / Mistral chat template.
    <|system|> ... <|user|> ... <|assistant|> ...
    """
    return (
        f"<|system|>\nYou are a research assistant. Answer using the provided context.\n"
        f"Context: {sample['context']}\n</s>\n"
        f"<|user|>\n{sample['question']}\n</s>\n"
        f"<|assistant|>\n{sample['answer']}\n</s>"
    )

# Fine-tuning
def finetune(
    model_id: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    epochs: int = 2,
    lora_rank: int = 8,
    lora_alpha: int = 16,
):
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Run: pip install peft trl bitsandbytes")
        return

    model_id   = model_id   or cfg.HF_MODEL_ID
    output_dir = output_dir or cfg.LORA_ADAPTER_PATH
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load & format dataset 
    raw_data  = load_dataset(dataset_path)
    formatted = [{"text": format_chat_prompt(s)} for s in raw_data]
    dataset   = Dataset.from_list(formatted)
    logger.info(f"Dataset: {len(dataset)} samples")

    # 2. Tokenizer
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Quantization config (QLoRA = 4-bit NF4) 
    use_cuda = torch.cuda.is_available()
    bnb_config = None
    if use_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit QLoRA quantization enabled.")
    else:
        logger.warning("CUDA not available — training in fp32 on CPU (slow, use GPU for real runs).")

    # 4. Load base model 
    logger.info(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else "cpu",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 5. LoRA config 
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA adapters attached | "
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=use_cuda,
        logging_steps=5,
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="none",                     
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        peft_config=lora_config,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # 8. Save
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.success(f"LoRA adapter saved → '{output_dir}'")
    logger.info("Set LORA_ADAPTER_PATH in .env and LLM_PROVIDER=huggingface to use it.")


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM with QLoRA")
    parser.add_argument("--model_id",      default=None,  help="HuggingFace model ID")
    parser.add_argument("--dataset_path",  default=None,  help="Path to JSON QA pairs")
    parser.add_argument("--output_dir",    default=None,  help="Where to save LoRA weights")
    parser.add_argument("--epochs",        type=int, default=2)
    parser.add_argument("--lora_rank",     type=int, default=8)
    args = parser.parse_args()

    finetune(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
    )