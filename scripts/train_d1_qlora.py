#!/usr/bin/env python3
"""Train the D1 Qwen2.5-0.5B QLoRA adapter on IME task groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from transformers import TrainerCallback

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DEFAULT_SEED = 20260822


def prompt_text(left: str, reading: str, right: str) -> str:
    return f"<left>{left}</left>\n<reading>{reading}</reading>\n<right>{right}</right>\n<answer>"


@dataclass
class TokenizationStats:
    examples: int = 0
    context_truncated: int = 0
    dropped: int = 0
    supervised_tokens: int = 0
    input_tokens: int = 0


class ImeDataset:
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.stats = TokenizationStats()
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer must define eos_token_id")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                left, right = str(row["left_context"]), str(row["right_context"])
                reading, answer = str(row["reading"]), str(row["positive"])
                prompt_ids = tokenizer.encode(prompt_text(left, reading, right), add_special_tokens=False)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos]
                truncated = False
                while len(prompt_ids) + len(answer_ids) > max_length and (left or right):
                    truncated = True
                    if len(left) >= len(right) and left:
                        left = left[1:]
                    elif right:
                        right = right[:-1]
                    prompt_ids = tokenizer.encode(prompt_text(left, reading, right), add_special_tokens=False)
                if len(prompt_ids) + len(answer_ids) > max_length:
                    self.stats.dropped += 1
                    continue
                input_ids = prompt_ids + answer_ids
                self.items.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": [-100] * len(prompt_ids) + answer_ids,
                    }
                )
                self.stats.examples += 1
                self.stats.context_truncated += int(truncated)
                self.stats.supervised_tokens += len(answer_ids)
                self.stats.input_tokens += len(input_ids)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CausalPaddingCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        import torch

        record = {
            "event": "log",
            "time": time.time(),
            "step": state.global_step,
            "num_input_tokens_seen": getattr(state, "num_input_tokens_seen", None),
            **(logs or {}),
        }
        if torch.cuda.is_available():
            record["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        for key in ("loss", "eval_loss", "grad_norm"):
            value = record.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise FloatingPointError(
                    f"non-finite {key} at optimizer step {state.global_step}: {value}"
                )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.handle.write(json.dumps({"event": "checkpoint", "time": time.time(), "step": state.global_step}) + "\n")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.handle.flush()
        self.handle.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    from importlib.metadata import version

    packages = ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "datasets")
    return {package: version(package) for package in packages}


def hardware_probe() -> dict[str, Any]:
    import torch

    probe: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        probe.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "driver": subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            }
        )
    return probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int, default=-1, help="probe/resume drill override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.train.is_file() or not args.dev.is_file():
        raise SystemExit("--train and --dev must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise SystemExit("D1 QLoRA training requires a CUDA GPU")
    random.seed(args.seed)
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    probe = hardware_probe()
    (args.output_dir / "hardware_probe.json").write_text(
        json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ImeDataset(args.train, tokenizer, args.max_length)
    dev_dataset = ImeDataset(args.dev, tokenizer, args.max_length)
    if len(train_dataset) != 10_000 and args.max_steps < 0:
        raise ValueError(f"D1 requires exactly 10,000 usable train groups, got {len(train_dataset)}")

    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        fp16=args.dtype == "fp16",
        bf16=args.dtype == "bf16",
        tf32=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        include_num_input_tokens_seen=True,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=False,
    )
    callback = JsonlMetricsCallback(args.output_dir / "training_log.jsonl")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CausalPaddingCollator(tokenizer.pad_token_id),
        callbacks=[callback],
    )
    started = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    wall_seconds = time.time() - started
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(args.output_dir / "final_adapter")

    metrics = dict(result.metrics)
    observed_input_tokens = getattr(trainer.state, "num_input_tokens_seen", 0)
    metrics.update(
        {
            "wall_seconds_observed": wall_seconds,
            "observed_input_tokens": observed_input_tokens,
            "observed_input_tokens_per_second": observed_input_tokens / wall_seconds,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trainable_parameters": trainable_parameters,
            "total_parameters_loaded": total_parameters,
        }
    )
    manifest = {
        "arm": "D1",
        "base_model": args.model,
        "base_model_license": "Apache-2.0",
        "base_model_card": "https://huggingface.co/Qwen/Qwen2.5-0.5B",
        "seed": args.seed,
        "objective": "causal LM loss on answer surface tokens and EOS only",
        "prompt_format": "<left>...</left>\\n<reading>...</reading>\\n<right>...</right>\\n<answer>",
        "config": vars(args) | {"train": str(args.train), "dev": str(args.dev), "output_dir": str(args.output_dir)},
        "data_sha256": {"train": file_sha256(args.train), "dev": file_sha256(args.dev)},
        "tokenization": {"train": asdict(train_dataset.stats), "dev": asdict(dev_dataset.stats)},
        "hardware_probe": probe,
        "metrics": metrics,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
