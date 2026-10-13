# 08 - Fine-Tuning an LLM with LoRA / QLoRA (domain Q&A, grounded style)

Teach a small open model to answer in a **grounded, cite-or-refuse** style using parameter-efficient fine-tuning.

```
data/train.jsonl  ->  src/prepare.py (chat template + loss only on answer)  ->  src/train.py (LoRA/QLoRA)
                                                                           ->  src/merge.py  ->  src/infer.py
```

## Quickstart
```bash
pip install -r requirements.txt
python -m src.train --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3            # CPU/small GPU friendly
python -m src.train --model meta-llama/Llama-3.2-3B-Instruct --qlora         # 4-bit, ~8GB VRAM (Linux+CUDA)
python -m src.infer --adapter outputs/adapter "How many vacation days carry over?"
python -m src.merge --adapter outputs/adapter --out outputs/merged           # optional: single merged model
```

## Dataset format (`data/train.jsonl`)
```json
{"context": "...passage...", "question": "...", "answer": "...grounded answer... or 'I don't know based on the provided context.'"}
```
Include **unanswerable** questions with refusal answers (about 20-30%). That is what reduces hallucination. The provided file is a tiny seed; use 500-5,000 high-quality examples for real gains.

## When to fine-tune vs RAG
- RAG: knowledge that changes, citations, access control.
- Fine-tune: style/format, tool-calling behaviour, refusal calibration, domain vocabulary.
- Best: both (fine-tuned model + RAG).
