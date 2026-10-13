import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from src.prepare import build_messages

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="outputs/adapter")
ap.add_argument("--context", default="Unused vacation days carry over up to a maximum of 5 days.")
ap.add_argument("question", nargs="*")
a = ap.parse_args()

cfg = PeftConfig.from_pretrained(a.adapter)
tok = AutoTokenizer.from_pretrained(a.adapter)
base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path)
model = PeftModel.from_pretrained(base, a.adapter).eval()
q = " ".join(a.question) or "How many days carry over?"
prompt = tok.apply_chat_template(build_messages({"context": a.context, "question": q}),
                                 tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=80, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
