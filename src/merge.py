import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="outputs/adapter"); ap.add_argument("--out", default="outputs/merged")
a = ap.parse_args()
cfg = PeftConfig.from_pretrained(a.adapter)
base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path)
merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
merged.save_pretrained(a.out); AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
print("Merged model saved to", a.out)
