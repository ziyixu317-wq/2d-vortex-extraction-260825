"""Train fresh seven-source W1-P seeds through the registered two stages."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import run_mfc_mixed_w1p_training as training
from migrate_mfc_batch_size_checkpoint import migrate_checkpoint
import ticket10_seven_source_protocol as protocol


def run_seed(args, config):
    if args.seed not in (1, 2):
        raise ValueError("Fresh Ticket11 runs use seed1 or seed2; audited seed0 is reused")
    base = Path(args.output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    record = protocol.execution_record(config)
    record["execution"] = {"seed": args.seed, "output": str(base),
        "input_root": str(args.input_root), "ticket03_root": str(args.ticket03_root),
        "mfc_flow": str(args.mfc_flow), "mfc_mask": str(args.mfc_mask), "mfc_cache": str(args.mfc_cache)}
    training._write_json(base / "execution_record.json", record)
    old = protocol.stage_config(config, 32)
    new = protocol.stage_config(config, 64)
    device = training.apply_cuda_limit(args.device, args.gpu_memory_limit_gib)
    sources, datasets, splits, old_sampling = training.load_sources(args, old)
    new_sampling = copy.deepcopy(old_sampling)
    new_sampling["batch_size"] = 64
    trainer = training.create_trainer(old, seed=args.seed, device=device)
    stages = [base / "stage32", base / "stage64"]
    runtime = {"activation_checkpointing": False, "gpu_memory_limit_gib": args.gpu_memory_limit_gib}
    if getattr(args, "prefetch_batches", 0):
        runtime["prefetch_batches"] = args.prefetch_batches
    metadata = {"ticket11_execution_record": protocol.checkpoint_record(record)}
    scientific = {k: config[k] for k in ("model", "train")}
    started = time.monotonic()
    # --max-seconds limits actual updates; it never changes the registered budget.
    def remaining():
        return None if args.max_seconds is None else max(1e-9, args.max_seconds - (time.monotonic() - started))

    if not stages[1].exists():
        resume = stages[0] / "checkpoints" / "latest.pt" if stages[0].exists() else None
        result = training.run_training(trainer, sources, output=stages[0], dataset_config=datasets,
            split_config=splits, sampling_config=old_sampling, epochs=25, seed=args.seed, device=device,
            resume=resume, checkpoint_every_updates=args.checkpoint_every_updates,
            max_seconds=remaining(), config=scientific, runtime_execution=runtime,
            checkpoint_metadata=metadata)
        if result["status"] != "complete":
            return _status(base, result, stages[0])
        migrated = migrate_checkpoint(trainer, checkpoint=stages[0] / "checkpoints" / "latest.pt",
            output=stages[1], dataset_config=datasets, split_config=splits, sampling_config=old_sampling,
            old_config=old, new_config=new, device=device, seed=args.seed,
            gpu_memory_limit_gib=args.gpu_memory_limit_gib)
        resume = Path(migrated["checkpoint"])
    else:
        latest = stages[1] / "checkpoints" / "latest.pt"
        resume = latest if latest.exists() else stages[1] / "checkpoints" / "migrated.pt"
    result = training.run_training(trainer, sources, output=stages[1], dataset_config=datasets,
        split_config=splits, sampling_config=new_sampling, epochs=100, seed=args.seed, device=device,
        resume=resume, checkpoint_every_updates=args.checkpoint_every_updates,
        max_seconds=remaining(), config=scientific, runtime_execution=runtime,
        checkpoint_metadata=metadata)
    return _status(base, result, stages[1])


def _status(base, result, stage):
    status = {"status": result["status"], "completed_epochs": result["next_epoch"],
        "next_update": result["next_update"], "global_step": result["global_step"],
        "latest_checkpoint": str(stage / "checkpoints" / "latest.pt"),
        "final_checkpoint": str(stage / "checkpoints" / "epoch_0100.pt") if result["next_epoch"] == 100 else None,
        "history": str(stage / "history.json"), "registered_epochs": 100}
    training._write_json(base / "status.json", status)
    print(json.dumps(status), flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/ticket11_seven_source_core.json"))
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=18.)
    parser.add_argument("--checkpoint-every-updates", type=int, default=25)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_seed(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
