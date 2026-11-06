#!/usr/bin/env python
"""Step 2: fine-tune Qwen-Good and Qwen-Bad with identical LoRA settings (Unsloth).

Both arms start from the same base model and use the same hyperparameters, seed, number
of examples and number of optimiser steps, so any difference in behaviour comes from the
data. Loss is computed on assistant turns only (system, user and tool tokens are masked).

Each arm trains in its own Python process so GPU memory is fully released between them.
The merged 16-bit model is written to checkpoints/qwen_<arm>/ and loads directly in vLLM.

Usage:
    python scripts/02_train_pair.py                    # both arms
    python scripts/02_train_pair.py --arms good        # one arm
    python scripts/02_train_pair.py --base-model Qwen/Qwen3-14B
"""

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locpair.config import load_config, read_jsonl, seed_everything  # noqa: E402

ARMS = ("good", "bad")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_arm(cfg, arm, base_model):
    # Unsloth patches transformers and trl, so it must be imported first.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from locpair.chat import render

    tcfg, paths = cfg["train"], cfg["paths"]
    seed_everything(cfg["seed"])
    data_file = Path(paths["data_dir"]) / f"qwen_{arm}_sft.jsonl"
    out_dir = Path(paths["checkpoints_dir"]) / f"qwen_{arm}"
    rows = read_jsonl(data_file)
    print(f"[{arm}] {len(rows)} examples from {data_file}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=tcfg["max_seq_len"], dtype=None,
        load_in_4bit=tcfg["load_in_4bit"])
    model = FastLanguageModel.get_peft_model(
        model, r=tcfg["lora_rank"], lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"], bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=cfg["seed"])

    texts = [render(tokenizer, r["messages"], add_generation_prompt=False) for r in rows]
    dataset = Dataset.from_list([{"text": t} for t in texts])

    # trl renamed a few arguments across versions; pass whichever this version accepts.
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    length_key = "max_length" if "max_length" in sft_params else "max_seq_length"
    args = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=tcfg["num_epochs"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        **{length_key: tcfg["max_seq_len"]},
    )
    trainer_kwargs = {"model": model, "train_dataset": dataset, "args": args}
    tok_key = "processing_class" if "processing_class" in inspect.signature(
        SFTTrainer.__init__).parameters else "tokenizer"
    trainer_kwargs[tok_key] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")

    start = time.time()
    result = trainer.train()
    minutes = (time.time() - start) / 60
    print(f"[{arm}] trained in {minutes:.1f} min, final loss {result.training_loss:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    meta = {"arm": arm, "base_model": base_model, "train_config": tcfg, "seed": cfg["seed"],
            "n_examples": len(rows), "data_file": str(data_file), "data_sha256": sha256(data_file),
            "global_steps": result.global_step, "training_loss": result.training_loss,
            "minutes": round(minutes, 1), "log_history": trainer.state.log_history}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[{arm}] saved merged 16-bit model to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--base-model", default=None, help="override train.base_model")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = args.base_model or cfg["train"]["base_model"]

    if args._child:
        train_arm(cfg, args.arms[0], base)
        return

    sizes = {a: len(read_jsonl(Path(cfg["paths"]["data_dir"]) / f"qwen_{a}_sft.jsonl"))
             for a in ARMS}
    if len(set(sizes.values())) != 1:
        sys.exit(f"SFT sets differ in size {sizes}; rerun 01_prepare_data.py")
    for arm in args.arms:
        cmd = [sys.executable, __file__, "--arms", arm, "--base-model", base, "--_child"]
        if args.config:
            cmd += ["--config", args.config]
        print(f"=== training {arm} ({base}) ===", flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"training {arm} failed")
    print("Done. Next: python scripts/03_run_eval.py --n-episodes 35")


if __name__ == "__main__":
    main()
