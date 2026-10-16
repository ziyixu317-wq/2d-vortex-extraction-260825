"""Run paired seeds 0/1 for W3 on the seven-source P95/P80 development base."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import mfc_mixed_w1p as mixed

import run_mfc_mixed_w1p_training as base
import ticket09_seven_source_w3 as branch
import ticket10_seven_source_protocol as stages
import w3
from migrate_mfc_batch_size_checkpoint import migrate_checkpoint


def create_trainer(cfg, *, seed, device):
    paired = base.create_trainer(cfg, seed=seed, device=device)
    adapter = w3.TrajectoryEmbeddingAdapter(paired.student, classification_mode="W1-P")
    with w3._seeded_view_rng(seed + 60_013, torch.device(device)):
        projection = w3.TrajectoryProjectionHead(adapter.embedding_dim).to(device)
    paired.optimizer.add_param_group({"params": list(projection.parameters()),
                                     "lr": cfg["train"]["lr"],
                                     "weight_decay": cfg["train"]["weight_decay"]})
    return branch.SevenSourceW3Trainer(
        paired.student, paired.optimizer, scheduler=paired.scheduler,
        teacher=paired.teacher, seed=seed, sampling_source=paired.sampling_source,
        config=paired.config, grad_clip_norm=paired.grad_clip_norm, teacher_device=device,
        projection_head=projection, w3_config=branch.SevenSourceW3Config(**cfg["w3"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=18.)
    parser.add_argument("--checkpoint-every-updates", type=int, default=25)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=0)
    parser.add_argument("--measure-updates", action="store_true")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not 1 <= args.epochs <= 100:
        raise ValueError("Paired development training has 1..100 epochs")
    if cfg["sampling"]["samples_per_epoch"] != 20000 or cfg["sampling"]["batch_size"] != 32:
        raise ValueError("The registered stage sequence starts with batch32 and 20000 patches")
    if cfg["train"].get("amp") or cfg["train"].get("data_parallel"):
        raise ValueError("Paired training is FP32 on one GPU")
    device = base.apply_cuda_limit(args.device, args.gpu_memory_limit_gib)
    sources, datasets, splits, sample = base.load_sources(args, cfg)
    trainer = create_trainer(cfg, seed=args.seed, device=device)
    if args.measure_updates:
        return measure_updates(trainer, sources, args, cfg, device)
    args.output.mkdir(parents=True, exist_ok=True)
    registration = {"model": cfg["model"], "train": cfg["train"]}
    record = {**cfg["run_metadata"], "w3": cfg["w3"], "seed": args.seed,
              "sampling_stages": stages.SAMPLING_STAGES,
              "supervision_base": "seven_source_W1-P", "input_root": str(args.input_root),
              "ticket03_root": str(args.ticket03_root), "mfc_flow": str(args.mfc_flow),
              "mfc_mask": str(args.mfc_mask), "mfc_cache": str(args.mfc_cache)}
    base._write_json(args.output / "execution_record.json", record)
    runtime = {"activation_checkpointing": False, "gpu_memory_limit_gib": args.gpu_memory_limit_gib}
    if args.prefetch_batches:
        runtime["prefetch_batches"] = args.prefetch_batches
    old_output, new_output = args.output / "stage32", args.output / "stage64"
    started = time.monotonic()
    def remaining():
        return None if args.max_seconds is None else max(1e-9, args.max_seconds - (time.monotonic() - started))
    common = dict(dataset_config=datasets, split_config=splits, seed=args.seed, device=device,
                  checkpoint_every_updates=args.checkpoint_every_updates, config=registration,
                  runtime_execution=runtime, checkpoint_metadata={"ticket09_execution_record": record})
    if not new_output.exists():
        resume = old_output / "checkpoints" / "latest.pt" if old_output.exists() else None
        result = base.run_training(trainer, sources, output=old_output, sampling_config=sample,
            epochs=min(25, args.epochs), resume=resume, max_seconds=remaining(), **common)
        if result["status"] != "complete" or args.epochs <= 25:
            return write_status(args.output, old_output, result)
        new_cfg = stages.stage_config(cfg, 64)
        migrated = migrate_checkpoint(trainer, checkpoint=old_output / "checkpoints" / "latest.pt",
            output=new_output, dataset_config=datasets, split_config=splits, sampling_config=sample,
            old_config=cfg, new_config=new_cfg, device=device, seed=args.seed,
            gpu_memory_limit_gib=args.gpu_memory_limit_gib)
        resume = Path(migrated["checkpoint"])
    else:
        resume = new_output / "checkpoints" / "latest.pt"
        if not resume.exists():
            resume = new_output / "checkpoints" / "migrated.pt"
    new_sample = copy.deepcopy(sample)
    new_sample["batch_size"] = 64
    result = base.run_training(trainer, sources, output=new_output, sampling_config=new_sample,
        epochs=args.epochs, resume=resume, max_seconds=remaining(), **common)
    return write_status(args.output, new_output, result)


def write_status(output, stage, result):
    status = {"status": result["status"], "global_step": result["global_step"],
              "completed_epochs": result["next_epoch"], "next_update": result["next_update"],
              "latest_checkpoint": str(stage / "checkpoints" / "latest.pt"),
              "final_checkpoint": str(stage / "checkpoints" / "epoch_0100.pt")
                  if result["next_epoch"] == 100 else None}
    base._write_json(output / "status.json", status)
    print(json.dumps(status), flush=True)
    return 0


def measure_updates(trainer, sources, args, cfg, device):
    args.output.mkdir(parents=True, exist_ok=False)
    for model in (trainer.student, trainer.teacher):
        for module in model.modules():
            if hasattr(module, "activation_checkpointing"):
                module.activation_checkpointing = False
    rows = []
    for batch_size, epoch in ((32, 0), (64, 25)):
        specs = mixed.build_epoch_specs(sources, samples_per_epoch=20000,
            batch_size=batch_size, positive_fraction=.5, seed=args.seed, epoch=epoch)
        for spec in specs[:7]:
            extracted = time.monotonic()
            batch, failed = base.make_batch(sources, spec, args.seed)
            extraction_seconds = time.monotonic() - extracted
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            stats = trainer.train_step(batch, epoch=epoch, device=device)
            torch.cuda.synchronize(device)
            row = {"batch_size": batch_size, "source_id": spec["dataset_index"],
                   "extraction_seconds": extraction_seconds,
                   "update_seconds": time.monotonic() - started,
                   "stats": stats, "cuda_resources": base.cuda_resources(device, started)}
            rows.append(row)
            base._write_json(args.output / "measurement.json", {"status": "running", "rows": rows})
            print(json.dumps({"event": "measurement", **row}), flush=True)
    base._write_json(args.output / "measurement.json", {"status": "complete", "rows": rows,
        "final_test_source_read": False, "formal_training_updates": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
