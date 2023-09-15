from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pdb
from MWDatasets import MWDataset_turn
import datasets
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str,
                    default="data/mw21_10p_train_v3.json",
                    help="training data file (few-shot or full shot)")
parser.add_argument('--test_fn', type=str,
                    default="data/mw21_100p_test.json",
                    help="training data file (few-shot or full shot)")

parser.add_argument('--base_model_name', type=str,
                    default='meta-llama/Llama-2-7b-chat-hf')

parser.add_argument('--save_dir', type=str,
                    default='expts/debug')


args = parser.parse_args()


if __name__ == "__main__":
    train_data = MWDataset_turn(args.train_fn).as_dict()
    train_data = datasets.Dataset.from_dict(train_data)
    # train_loader = DataLoader(train_data, batch_size=4,
    #                           shuffle=False, collate_fn=train_data.collate_fn)
    test_data = MWDataset_turn(args.train_fn).as_dict()
    test_data = datasets.Dataset.from_dict(test_data)
    # test_loader = DataLoader(test_data, batch_size=4,
    #                          shuffle=False, collate_fn=test_data.collate_fn)
    # test_data = MWDataset_turn(args.test_fn)

    llama_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, trust_remote_code=True)

    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # Training Params
    train_params = TrainingArguments(
        output_dir=f"{args.save_dir}",
        num_train_epochs=0.01,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params
    )
    # Training
    fine_tuning.train()
    fine_tuning.model.save_pretrained(f"{args.save_dir}/model")
