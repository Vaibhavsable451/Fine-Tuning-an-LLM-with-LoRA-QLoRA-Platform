import argparse, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.prepare import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--data", default="data/train.jsonl")
ap.add_argument("--out", default="outputs/adapter")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--qlora", action="store_true", help="4-bit NF4 quantisation (needs CUDA + bitsandbytes)")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

kwargs = {}
if a.qlora:
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    kwargs["device_map"] = "auto"
else:
    kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(a.model, **kwargs)
if a.qlora:
    model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, LoraConfig(
    r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

ds = load(a.data, tok)
args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    logging_steps=1, save_strategy="no", report_to="none", warmup_ratio=0.05,
    lr_scheduler_type="cosine", bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit" if a.qlora else "adamw_torch")
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
print("Saved adapter to", a.out)
