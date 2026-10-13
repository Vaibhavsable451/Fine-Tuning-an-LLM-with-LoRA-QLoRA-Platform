import json
from datasets import Dataset

SYSTEM = ("Answer using only the provided context. If the answer is not in the context, "
          "reply exactly: I don't know based on the provided context.")

def build_messages(ex):
    user = f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

def tokenize_example(ex, tok, max_len=1024):
    """Mask the prompt so loss is computed ONLY on the answer tokens."""
    prompt = tok.apply_chat_template(build_messages(ex), tokenize=False, add_generation_prompt=True)
    answer = ex["answer"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

def load(path, tok):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ds = Dataset.from_list(rows)
    return ds.map(lambda e: tokenize_example(e, tok), remove_columns=ds.column_names)
