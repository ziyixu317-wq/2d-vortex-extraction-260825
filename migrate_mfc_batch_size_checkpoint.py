"""Explicitly migrate a completed-epoch mixed W1-P checkpoint from batch 32 to 64."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

import run_mfc_mixed_w1p_training as training


def migrate_checkpoint(trainer, *, checkpoint, output, dataset_config, split_config,
                       sampling_config, old_config, new_config, device="cpu", seed=0,
                       gpu_memory_limit_gib=18.):
    """Restore through the strict trainer interface, then save a new sampling stage."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    old_sampling = old_config["sampling"]
    new_sampling = copy.deepcopy(sampling_config)
    new_sampling["batch_size"] = 64
    expected_config_sampling = copy.deepcopy(old_sampling)
    expected_config_sampling["batch_size"] = 64
    if old_sampling["batch_size"] != 32 or new_config["sampling"] != expected_config_sampling:
        raise ValueError("Migration permits only sampling.batch_size 32 -> 64")
    if any(sampling_config.get(key) != value for key, value in old_sampling.items()):
        raise ValueError("Original sampling metadata differs from the old configuration")
    scientific_config = {key: old_config[key] for key in ("model", "train")}
    if scientific_config != {key: new_config[key] for key in ("model", "train")}:
        raise ValueError("Migration must preserve the original model/train configuration")
    loaded = trainer.load_checkpoint(checkpoint, expected_dataset_config=dataset_config,
        expected_split_config=split_config, expected_sampling_config=sampling_config,
        device=device, restore_rng=True, strict_cuda_rng=torch.device(device).type == "cuda")
    extra = copy.deepcopy(loaded["extra_metadata"])
    if extra["training_config"] != scientific_config or trainer.seed != seed:
        raise ValueError("Checkpoint training config or seed differs from the original run")
    progress = extra["mixed_training_progress"]
    epoch = progress["next_epoch"]
    history = progress["history"]
    if progress["next_update"] != 0 or progress["epoch_rows"]:
        raise ValueError("Batch-size migration requires a completed epoch boundary")
    if (epoch != loaded["epoch"] or len(history) != epoch
            or any(row["epoch"] != index or row["completed_epoch"] != index + 1
                   for index, row in enumerate(history))
            or (history and history[-1]["global_step"] != trainer.global_step)
            or loaded["metrics"].get("history") != history):
        raise ValueError("Checkpoint requires complete, consistent epoch history")
    if not 0 <= epoch < new_config["train"]["epochs"]:
        raise ValueError("Migration requires a remaining training stage")
    origin = str(Path(checkpoint).resolve())
    samples_per_epoch = sampling_config["samples_per_epoch"]
    old_updates = math.ceil(samples_per_epoch / 32)
    new_updates = math.ceil(samples_per_epoch / 64)
    progress["sampling_stages"] = [
        {"start_epoch": 0, "end_epoch": epoch, "epoch_range": [0, epoch], "batch_size": 32,
         "sampling_config": copy.deepcopy(sampling_config),
         "source_checkpoint": origin, "updates_per_epoch": old_updates,
         "last_batch_size": samples_per_epoch % 32 or 32},
        {"start_epoch": epoch, "end_epoch": new_config["train"]["epochs"],
         "epoch_range": [epoch, new_config["train"]["epochs"]], "batch_size": 64,
         "sampling_config": copy.deepcopy(new_sampling), "source_checkpoint": origin,
         "updates_per_epoch": new_updates, "last_batch_size": samples_per_epoch % 64 or 64},
    ]
    extra["batch_size_stage_plan"] = {"prefix_completed_epochs": epoch,
        "samples_per_epoch": samples_per_epoch,
        "planned_total_updates": epoch * old_updates + (new_config["train"]["epochs"] - epoch) * new_updates,
        "planned_total_samples": new_config["train"]["epochs"] * samples_per_epoch}
    extra["runtime_execution"] = {"activation_checkpointing": False,
                                  "gpu_memory_limit_gib": gpu_memory_limit_gib}
    for model in (trainer.student, trainer.teacher):
        for module in model.modules():
            if hasattr(module, "activation_checkpointing"):
                module.activation_checkpointing = False
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    migrated = checkpoint_dir / "migrated.pt"
    trainer.save_checkpoint(migrated, epoch=epoch, dataset_config=dataset_config,
        split_config=split_config, sampling_config=new_sampling, metrics=loaded["metrics"],
        calibration_policy=loaded["calibration_policy"], anchor_hash=loaded["anchor_hash"],
        extra_metadata=extra)
    training._write_json(output / "history.json", history)
    training._write_json(output / "progress.json", {"next_epoch": epoch, "next_update": 0,
        "global_step": trainer.global_step, "checkpoint": str(migrated)})
    return {"checkpoint": str(migrated), "next_epoch": epoch, "global_step": trainer.global_step,
            "sampling_stages": progress["sampling_stages"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache",
                 "old-config", "new-config", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=18.)
    args = parser.parse_args(argv)
    old_config = json.loads(args.old_config.read_text(encoding="utf-8"))
    new_config = json.loads(args.new_config.read_text(encoding="utf-8"))
    if old_config["sampling"]["batch_size"] != 32 or old_config["sampling"]["samples_per_epoch"] != 20_000:
        raise ValueError("The original mixed run requires batch 32 and 20000 samples per epoch")
    if old_config["train"].get("amp", False) or old_config["train"].get("data_parallel", False):
        raise ValueError("Migration uses the original FP32 single-device configuration")
    if args.output.exists():
        raise FileExistsError(args.output)
    device = training.apply_cuda_limit(args.device, args.gpu_memory_limit_gib)
    _, datasets, splits, sampling = training.load_sources(args, old_config)
    trainer = training.create_trainer(old_config, seed=0, device=device)
    result = migrate_checkpoint(trainer, checkpoint=args.checkpoint, output=args.output,
        dataset_config=datasets, split_config=splits, sampling_config=sampling,
        old_config=old_config, new_config=new_config, device=device, seed=0,
        gpu_memory_limit_gib=args.gpu_memory_limit_gib)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
