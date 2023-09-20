from peft import LoraConfig, PeftModel, prepare_model_for_int8_training, get_peft_model
from trl import SFTTrainer
import pdb
from MWDatasets import MWDataset_dial, MWDataset_turn
import datasets
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline,  EvalPrediction
import numpy as np
import json
from transformers import EarlyStoppingCallback
from transformers.pipelines.pt_utils import KeyDataset
import datasets
import random
from tqdm import tqdm

import evaluator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str,
                    default="data/mw21_10p_train_v3.json",
                    help="training data file (few-shot or full shot)")
parser.add_argument('--val_fn', type=str,
                    default="data/mw21_100p_dev.json",
                    help="training data file (few-shot or full shot)")
parser.add_argument('--test_fn', type=str,
                    default="data/mw21_100p_test.json",
                    help="training data file (few-shot or full shot)")
parser.add_argument('--base_model_name', type=str,
                    default='meta-llama/Llama-2-7b-chat-hf')

parser.add_argument('--save_dir', type=str,
                    default='expts/debug')
parser.add_argument('--save_model', type=int,
                    default=1)
parser.add_argument('--report', type=int,
                    default=1)
parser.add_argument('--seed', type=int,
                    default=42)
parser.add_argument('--log_step_num', type=int)
parser.add_argument('--save_step_num', type=int)
parser.add_argument('--epoch_num', type=float)
parser.add_argument('--turn_level', type=int, default=1)
parser.add_argument('--short_testing', type=int, default=0)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--mwz_ver', type=str, default="2.1",
                    choices=['2.1', '2.4'], help="version of MultiWOZ")

args = parser.parse_args()


# def compute_metrics(pred):
#     global pred_result, gold_result
#     labels = pred.label_ids
#     preds = pred.predictions

#     labels = np.where(labels != -100, labels, 0)
#     labels = llama_tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # preds to text

#     # find maximuum logit
#     preds = np.argmax(preds, axis=-1)
#     preds = llama_tokenizer.batch_decode(preds, skip_special_tokens=True)
#     jga = 0
#     for l, p in zip(labels, preds):
#         if l == p:
#             jga += 1
#     pred_result = preds
#     gold_result = labels
#     return {
#         'jga': jga/len(labels)
#     }


def init_experiment(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    ### Response:
    {data_point["output"]}"""


def compute_perplexity(p: torch.Tensor) -> torch.Tensor:
    if isinstance(p, torch.Tensor):
        return torch.exp(p)
    elif isinstance(p, (float, int, np.float32)):
        return torch.exp(torch.tensor(p))
    else:
        raise ValueError("Input must be a PyTorch tensor, float, or int")


def compute_metrics(p: EvalPrediction) -> dict:
    perplexity = compute_perplexity(p.predictions.mean())
    return {"perplexity": perplexity.item()}


if __name__ == "__main__":
    init_experiment(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    # save args
    with open(f"{args.save_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(args)
    print("using turn level" if args.turn_level else "using dialog level")

    Dataset = MWDataset_turn if args.turn_level else MWDataset_dial

    train_data = Dataset(args.train_fn, 'train').as_dict()
    train_data = datasets.Dataset.from_dict(train_data)
    # train_loader = DataLoader(train_data, batch_size=4,
    #                           shuffle=False, collate_fn=train_data.collate_fn)
    val_data = Dataset(args.val_fn, 'val').as_dict(
        short=args.short_testing)
    val_data = datasets.Dataset.from_dict(val_data)

    test_data = Dataset(args.test_fn, 'test').as_dict(
        short=args.short_testing)
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
    # base_model.config.pretraining_tp = 1

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        # target_modules=['q_proj','v_proj']
    )

    # Training Params
    model = prepare_model_for_int8_training(base_model)
    model = get_peft_model(model, peft_parameters)
    model.print_trainable_parameters()

    train_params = TrainingArguments(
        output_dir=f"{args.save_dir}",
        num_train_epochs=args.epoch_num,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=args.save_step_num,
        logging_steps=args.log_step_num,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        eval_steps=args.save_step_num,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        run_name=args.save_dir,
        report_to='wandb' if args.report else 'none',
        prediction_loss_only=True,
        save_total_limit=5
    )

    fine_tuning = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # compute_metrics=compute_metrics,
    # Training
    try:
        fine_tuning.train(resume_from_checkpoint=True)
    except ValueError:
        print("No checkpoint found, starting from scratch")
        fine_tuning.train()

    score = fine_tuning.evaluate()
    print(score)

    # Save score
    with open(f"{args.save_dir}/score.json", 'w') as f:
        json.dump(score, f, indent=4)

    id, preds, labels = evaluator.run(args, test_data, model, llama_tokenizer)

    result = [
        {'id': id,
         'pred': p,
         'gold': g}
        for (id, p, g) in zip(id,  preds, labels)
    ]
    with open(f"{args.save_dir}/raw_result.json", 'w') as f:
        json.dump(result, f, indent=4)
