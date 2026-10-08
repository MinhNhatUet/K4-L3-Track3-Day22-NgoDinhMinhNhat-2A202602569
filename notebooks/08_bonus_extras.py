# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB8 — β-sweep (+6) và đẩy adapter lên Hugging Face Hub (+3)
#
# Cần NB1–NB3 đã chạy. β = 0.1 dùng lại adapter của NB3, chỉ huấn luyện thêm
# β = 0.05 và 0.5 (~40–60 phút mỗi lần trên T4).

# %%
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

from lab22 import config as C

assert (C.DPO_ADAPTER / "dpo_metrics.json").exists(), "Run NB3 first"

# %% [markdown]
# ## 1. β-sweep

# %%
b010 = C.ADAPTERS / "dpo-b0.10"
if not (b010 / "dpo_metrics.json").exists():
    shutil.copytree(C.DPO_ADAPTER, b010, dirs_exist_ok=True)
for beta in ("0.05", "0.5"):
    out = C.ADAPTERS / f"dpo-b{float(beta):.2f}"
    if (out / "dpo_metrics.json").exists():
        print(f"skip β={beta}: {out} đã có")
        continue
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_dpo.py"), "--beta", beta, "--output-dir", str(out)], check=True)

subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_judge.py"), "--plot-sweep",
                "--sweep-dir", str(C.ADAPTERS), "--output", str(C.SCREENSHOTS / "bonus-beta-sweep.png")], check=True)
for path in sorted(C.ADAPTERS.glob("dpo-b*/dpo_metrics.json")):
    m = json.loads(path.read_text())
    print(f"β={m['beta']:<5} held-out margin {m.get('eval_reward_gap')}  acc {m.get('eval_reward_accuracy')}  "
          f"chosen {m.get('eval_chosen_reward')}  rejected {m.get('eval_rejected_reward')}")

# %% [markdown]
# **Đọc kết quả:** β lớn giữ policy gần reference hơn (log-ratio nhỏ), nhưng margin = β·log-ratio
# nên có thể vẫn lớn. So độ chính xác reward held-out để biết β nào phân biệt tốt nhất.
#
# ## 2. Đẩy adapter DPO + model card lên Hugging Face Hub
#
# Thêm Colab Secret `HF_TOKEN` (quyền *write*). Đặt `HF_REPO_ID` nếu muốn tên repo khác.

# %%
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
if not token:
    print("Không có HF_TOKEN → bỏ qua bước đẩy lên Hub.")
else:
    api = HfApi(token=token)
    repo_id = os.environ.get("HF_REPO_ID") or f"{api.whoami()['name']}/lab22-qwen3-4b-vi-dpo"
    metrics = json.loads((C.DPO_ADAPTER / "dpo_metrics.json").read_text())
    summary_path = C.EVAL_DIR / "judge_summary.json"
    judge = json.loads(summary_path.read_text()).get("overall", {}) if summary_path.exists() else {}
    (C.DPO_ADAPTER / "README.md").write_text(f"""---
base_model: {C.BASE_MODEL}
library_name: peft
language: [vi]
tags: [dpo, lora, trl, unsloth]
---
# Lab 22 · Qwen3-4B SFT + DPO (tiếng Việt)

LoRA DPO adapter huấn luyện trên mô hình SFT đã gộp (`models/sft-merged`, 1k VN Alpaca)
với {C.PREF_DATASET} (lọc tiếng Việt, 800 cặp train / 100 held-out).

- β = {metrics['beta']}, lr = {metrics['lr']}, loss = {metrics['loss_type']}
- Held-out: margin {metrics['eval_reward_gap']}, reward accuracy {metrics['eval_reward_accuracy']}
- Chẩn đoán reward: {metrics['diagnosis']}
- Win rate DPO vs SFT (NB4): {judge.get('dpo_win_rate')} (CI95 {judge.get('win_rate_ci95')})

Adapter trỏ tới mô hình SFT đã gộp, không phải trọng số gốc; cần gộp SFT trước khi nạp.
Chỉ dùng cho học tập: dữ liệu sở thích không ghi giấy phép.
""", encoding="utf-8")
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(C.DPO_ADAPTER), ignore_patterns=["*checkpoint*"])
    print(f"https://huggingface.co/{repo_id}")
