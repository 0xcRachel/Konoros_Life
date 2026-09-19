# Konoros-Sec (yAI) — v1.1

Defensive, authorized security assistant. Không exploit / persistence / credential theft / evasion.

## Kiến trúc (chuẩn GPT hiện đại)
- Decoder-only Transformer: RMSNorm, RoPE (có offset cho KV-cache), SwiGLU, tie-weights
- **GQA**: `n_kv_heads` (tiny MHA 4/4, small GQA 8q/2kv) — `config/model/small.yaml`
- **KV-cache generate**: prefill 1 lần, decode 1 token/step, sliding-window re-prefill khi đầy context
- **Train**: AMP bf16, cosine+warmup, grad accumulation, gradient checkpointing, resume, kill-switch
- **Epochs**: `training.epochs>0` thì `max_steps = epochs * steps_per_epoch` (tiny 3 epochs/batch 64, small 2 epochs/eff-batch 64)
- Vocab 320 byte-level + special `<think>/<answer>/<user>/...` (reasoning-ready từ v0.1)

## Dữ liệu tinh hoa (data_prime/) — đã fetch GĐ1
- `data_prime/stackoverflow_elite/api_prime.py` — SE API, chỉ accepted + q_score>=5 + a_score>=3 + code>=3 dòng (đã lấy 32 bản ghi elite)
- `data_prime/stackoverflow_elite/dump_prime.py` — GĐ2 Colab: stream Posts.xml.7z, không bung full
- `data_prime/wiki_elite/hf_stream.py` — stream HF wikipedia en+vi, lọc bài dài + bỏ stub (cần `pip install datasets`, chạy Colab)
- `data_prime/common/` — clean_text (unicode/boilerplate), dedup MinHash 0.85, license+manifest
- `data_prime/build/merge_prime.py` — gộp → `data/raw/prime_all.jsonl` (hiện 280 bản ghi / 0.9MB sau dedup; elite 154 + batch cũ)
- Ngưỡng lọc trong `data_prime/configs/elite_filters.yaml`
- Colab 83GB RAM: gắn `SE_API_KEY` (stackapps, miễn phí) → `api_prime --max 2000` + `hf_stream` + dump GĐ2 (lệnh trong notebook cell 1 + cuối)

## Tối ưu token in/out
- **Input (packing):** `prepare_data.py` encode từng doc + EOS phân cách, nối liền zero-padding; in `[budget]` steps/epoch cho tiny/small
- **Output (phân nhánh):** KV-cache decode + sliding-window re-prefill; `generate --best-of N` sample N nhánh, chấm format/ít lặp/đủ dài, trả nhánh tốt nhất

## Dữ liệu batch cũ (giữ lại)
- `data/raw/so_all.jsonl` — **208 Q&A** (python 90, javascript 40, linux 20, security 39, networking 19) via Stack Exchange API
- `data/raw/so_python.jsonl`, `so_security.jsonl` — batch đầu
- `data/raw/web_docs.txt` — Python tutorial + secrets/hashlib + MDN Web Security (allowlist + robots.txt)
- Lấy thêm: `python data_collectors/stackoverflow/fetch_all.py --out data/raw/so_all.jsonl --max-answers 500` (quota ẩn danh ~300 req/ngày/IP, reset hàng ngày; key miễn phí tại stackapps.com cho 10k/ngày)

## Học tăng cường preference (DPO, v1.2)
- `training/dpo.py` — Direct Preference Optimization: loss `-log sigmoid(beta * margin)`, ref model frozen, mask prompt, không cần reward model
- Data: `data/sft/security_prefs.jsonl` (8 cặp chosen/rejected: benign giúp > từ chối/khuyên bậy, harmful từ chối > tuân thủ)
- Eval: `python -m security.eval.pref_eval --ckpt <dpo.pt>` (preference accuracy)
- Colab: SFT → `training.dpo --steps 40` → GRPO (base từ `experiments/v1.2/dpo.pt`)

## RL suy nghĩ nhiều bước (GRPO, v1.3)
- `training/rewards.py` — R = 0.4 format (`<think>/<answer>`) + 0.5 safety (benign giúp / harmful từ chối) + 0.1 think_shape
- `training/rollout.py` — sample G completions/prompt + parse think/answer
- `training/grpo.py` — Group Relative Policy Optimization: advantage chuẩn hoá theo nhóm, clipped ratio, KL penalty về ref model, không cần value network
- Data: `data/sft/security_reasoning.jsonl` (8 mẫu think traces defensive)
- Eval: `python -m security.eval.reasoning_eval --ckpt <grpo.pt>`
- Colab: SFT warmup `security_reasoning.jsonl` → `training.grpo --steps 50 --G 4` → reasoning_eval

## Chạy trên Colab (KHÔNG train máy local)
Mở `notebooks/konoros_sec_v11_colab.ipynb` và chạy từng cell:
```bash
pip install -r requirements.txt
python -m pytest test/ -q
python scripts/prepare_data.py --input data/raw/prime_all.jsonl,data/raw/train.txt --train-out data/processed/train.bin --val-out data/processed/val.bin
python -m training.train --config config/model/tiny.yaml     # smoke test
python -m training.train --config config/model/small.yaml    # scale up khi tiny loss giảm
python -m evaluation.evaluate --config config/model/tiny.yaml --ckpt <ckpt>.pt
python -m training.sft --config config/model/tiny.yaml --data data/sft/security_sft.jsonl --base-ckpt <pretrain.pt> --out experiments/v1.1/sft.pt --max-steps 60 --lr 1e-5
python -m security.eval.safety_eval --config config/model/tiny.yaml --ckpt experiments/v1.1/sft.pt
```

## Cấu trúc
- `model/` — base LM (RMSNorm, RoPE, SwiGLU, tie-weights), reasoning-ready vocab 320
- `tokenizer/` — byte-level + special `<think>/<answer>/<user>/...`
- `training/` — pretrain (`train.py`) + SFT v1.1 (`sft.py`, `chat_template.py`)
- `data_collectors/` — StackOverflow (SE API, CC-BY-SA), web docs allowlist, user inbox (consent bắt buộc)
- `security/` — scope_validator, authorization (human approval), sandbox whitelist, audit, redaction, safety_eval
- `data/sft/security_sft.jsonl` — 6 mẫu SFT defensive mẫu
- `notebooks/konoros_sec_v11_colab.ipynb` — test trên Colab

## Chạy nhanh (local / Colab)
```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python -m pytest test/ -q
python -m training.train --config config/model/tiny.yaml
python -m evaluation.evaluate --config config/model/tiny.yaml --ckpt experiments/v0.1/checkpoints/<ckpt>.pt
python -m training.sft --config config/model/tiny.yaml --data data/sft/security_sft.jsonl --base-ckpt <pretrain.pt> --out experiments/v1.1/sft.pt --max-steps 60 --lr 1e-5
python -m security.eval.safety_eval --config config/model/tiny.yaml --ckpt experiments/v1.1/sft.pt
```

## Data SO hợp pháp
```bash
python data_collectors/stackoverflow/collect_so.py --tag python --pages 2 --out data/raw/so_python.jsonl
```
Chỉ dùng SE API / dump, giữ attribution + license CC-BY-SA, redact PII.

## Nguyên tắc
1. Chỉ scan target có authorization + human approval.
2. Tool whitelist trong sandbox, audit mọi hành động intrusive.
3. OSINT nguồn công khai, tôn trọng ToS/robots.txt/privacy.
4. OPSEC phòng thủ: redaction, secret scan, secure logging.
