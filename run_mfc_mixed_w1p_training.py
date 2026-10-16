"""Train fresh seven-source W1-P, using the existing resumable trainer closure."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

import extractor
import mfc_mixed_w1p as mixed
from mixed_batch_prefetch import batch_stream
from mfc_pathline_source import MFCPathlineSource
import ticket04_real_training as legacy
import train_kaggle
import w1_p
import weak_supervision_contract as contract


def exclude_incomplete_legacy(batch, *, t_scale):
    """Retain artifact validation/provenance, exclude padded endpoint failures."""
    paths = np.asarray(batch.pathlines)
    # The old normalization maps the actual physical window end to t_scale.
    # Float32 normalized-time tolerance is 1e-5 (4e-5 of a window at scale .25).
    failed = ~np.isclose(paths[:, -1, :, extractor.CH_T], t_scale, rtol=0, atol=1e-5)
    known = np.asarray(batch.label_mask, dtype=bool) & ~failed
    provenance = copy.deepcopy(batch.contract_batch.provenance)
    provenance["mixed_training_pathline_failure"] = {
        "criterion": "normalized final physical time differs from window end",
        "normalized_time_atol": 1e-5,
        "failed_count": int(failed.sum()),
        "known_excluded_count": int((failed & np.asarray(batch.label_mask)).sum()),
        "all_rk4_internal_support_verified": False,
    }
    return w1_p.build_w1_p_batch(
        paths, batch.labels, known, ~known, batch.solid_mask,
        invalid_mask=np.asarray(batch.invalid_mask, dtype=bool) | failed,
        sampling_source=batch.sampling_source, split_name="train",
        provenance=provenance, dummy_field=batch.dummy_field,
    ), failed


def make_batch(sources, spec, seed):
    source = sources[spec["dataset_index"]]
    batch = source.make_batch(spec, seed=seed)
    if source.source_id < 6:
        return exclude_incomplete_legacy(batch, t_scale=source.config.t_scale)
    failed = np.asarray(batch.contract_batch.provenance["failed_pathline_mask"], dtype=bool)
    return batch, failed


def apply_cuda_limit(device, gib):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Mixed W1-P training requires an available CUDA device")
    if not np.isfinite(gib) or gib <= 0:
        raise ValueError("GPU memory limit must be positive finite GiB")
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1., float(gib) * 1024 ** 3 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    print(json.dumps({"device": str(device), "gpu_memory_limit_gib": gib,
                      "allocator_fraction": fraction}), flush=True)
    return device


def create_trainer(cfg, *, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed(seed)
    model = train_kaggle.build_model_from_config(cfg, mode=contract.MODE_W1_P)
    model.to(device)
    c = cfg["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = train_kaggle.TwoStepLR(optimizer, c["lr"], c["second_lr"], c["warmup_epochs"])
    return w1_p.W1PTrainer(
        model, optimizer, scheduler=scheduler, seed=seed,
        sampling_source=contract.LABEL_SOURCE_P95P80,
        config=w1_p.W1PConfig(ramp_up_epochs=12),
        grad_clip_norm=c["grad_clip"], teacher_device=device,
    )


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def cuda_resources(device, started):
    free, total = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    return {"allocated_gib": torch.cuda.memory_allocated(device) / gib,
            "reserved_gib": torch.cuda.memory_reserved(device) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
            "driver_free_gib": free / gib, "driver_total_gib": total / gib,
            "elapsed_training_seconds": time.monotonic() - started}


def _summary(rows):
    result = {"updates": len(rows), "samples": sum(r["samples"] for r in rows),
              "failed_pathlines": sum(r["failed_pathlines"] for r in rows)}
    if not rows:
        return result
    for key in ("loss", "anchor_loss", "pseudo_loss", "consistency_loss",
                "anchor_coverage", "pseudo_acceptance", "unknown_coverage", "invalid_coverage"):
        result[key] = float(np.mean([r[key] for r in rows]))
    for key in ("anchor_count", "anchor_positive_count", "anchor_negative_count", "unknown_count",
                "invalid_count", "solid_count", "pseudo_eligible_count", "pseudo_accepted_count"):
        result[key] = sum(int(r[key]) for r in rows)
    result["failed_rate"] = result["failed_pathlines"] / sum(r["pathline_count"] for r in rows)
    return result


def run_training(trainer, sources, *, output, dataset_config, split_config,
                 sampling_config, epochs, seed, device, resume=None,
                 checkpoint_every_updates=25, max_seconds=None, config=None,
                 runtime_execution=None,
                 checkpoint_metadata=None,
                 spec_builder=mixed.build_epoch_specs, batch_builder=make_batch):
    """Ordered batch producer; checkpoint progress names the next update.

    Batch32 uses 625 updates/epoch; batch64 uses 313 with a final batch32.
    The injectable seams allow small
    CPU tests of the existing trainer's optimizer/EMA/RNG resume behavior.
    """
    output = Path(output).resolve()
    if checkpoint_every_updates <= 0 or epochs <= 0:
        raise ValueError("epoch and checkpoint intervals must be positive")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        resume = Path(resume).resolve()
        if not output.is_dir() or resume.parent != output / "checkpoints":
            raise ValueError("Resume must use this existing output's checkpoint directory")
    cpdir = output / "checkpoints"
    cpdir.mkdir(exist_ok=True)
    latest = cpdir / "latest.pt"
    progress = {"next_epoch": 0, "next_update": 0, "epoch_rows": [], "history": []}
    retained_metadata = {}
    if resume is not None:
        try:
            loaded = trainer.load_checkpoint(
                resume, expected_dataset_config=dataset_config,
                expected_split_config=split_config, expected_sampling_config=sampling_config,
                device=device, restore_rng=True, strict_cuda_rng=torch.device(device).type == "cuda")
        except torch.cuda.OutOfMemoryError as exc:
            recovery = cpdir / "failure_resume.pt"
            if resume != recovery:
                shutil.copyfile(resume, recovery)
            _write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                "phase": "checkpoint_restore", "recovery_checkpoint": str(recovery)})
            raise
        extra = loaded["extra_metadata"]
        if extra["training_config"] != config or trainer.seed != seed:
            raise ValueError("Resume training config or seed differs from saved run")
        progress = extra["mixed_training_progress"]
        retained_metadata = copy.deepcopy(extra)

    # Execution settings are applied after complete state/RNG restoration and
    # the original training config check; they do not change that config.
    if runtime_execution is not None:
        runtime_execution = copy.deepcopy(runtime_execution)
        for model in (trainer.student, trainer.teacher):
            for module in model.modules():
                if hasattr(module, "activation_checkpointing"):
                    module.activation_checkpointing = runtime_execution["activation_checkpointing"]

    def save(path=latest):
        extra_metadata = copy.deepcopy(retained_metadata)
        extra_metadata.update(copy.deepcopy(checkpoint_metadata or {}))
        extra_metadata.update({"mixed_training_progress": copy.deepcopy(progress), "training_config": config})
        if runtime_execution is not None:
            extra_metadata["runtime_execution"] = runtime_execution
        trainer.save_checkpoint(
            path, epoch=progress["next_epoch"], dataset_config=dataset_config,
            split_config=split_config, sampling_config=sampling_config,
            metrics={"history": progress["history"], "partial_epoch": _summary(progress["epoch_rows"])},
            extra_metadata=extra_metadata)
        _write_json(output / "history.json", progress["history"])
        _write_json(output / "progress.json", {"next_epoch": progress["next_epoch"],
            "next_update": progress["next_update"], "global_step": trainer.global_step,
            "checkpoint": str(path)})

    save()
    started = time.monotonic()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        while progress["next_epoch"] < epochs:
            epoch = progress["next_epoch"]
            specs = spec_builder(sources, samples_per_epoch=sampling_config["samples_per_epoch"],
                batch_size=sampling_config["batch_size"], positive_fraction=sampling_config["positive_fraction"],
                seed=seed, epoch=epoch)
            prefetch_batches = (runtime_execution or {}).get("prefetch_batches", 0)
            with batch_stream(sources, specs, seed, batch_builder,
                              start_update=progress["next_update"],
                              prefetch_batches=prefetch_batches) as stream:
                batch_requested_at = time.perf_counter()
                for update, batch, failed in stream:
                    data_wait_seconds = time.perf_counter() - batch_requested_at
                    if max_seconds is not None and time.monotonic() - started >= max_seconds:
                        save()
                        return {"status": "time_limit", "global_step": trainer.global_step, **progress}
                    spec = specs[update]
                    update_started_at = time.perf_counter()
                    stats = trainer.train_step(batch, epoch=epoch, device=device)
                    row = {**stats, "source_id": int(spec["dataset_index"]), "update": update,
                           "samples": int(batch.pathlines.shape[0]), "pathline_count": int(np.asarray(failed).size),
                           "failed_pathlines": int(np.asarray(failed).sum()),
                           "lr": float(trainer.optimizer.param_groups[0]["lr"])}
                    progress["epoch_rows"].append(row)
                    progress["next_update"] = update + 1
                    log = {"event": "update", **row}
                    if torch.device(device).type == "cuda":
                        log["cuda_resources"] = cuda_resources(device, started)
                    log["pipeline_resources"] = {"prefetch_batches": prefetch_batches,
                        "data_wait_seconds": data_wait_seconds,
                        "update_seconds": time.perf_counter() - update_started_at}
                    print(json.dumps(log), flush=True)
                    if trainer.global_step % checkpoint_every_updates == 0:
                        save()
                    batch_requested_at = time.perf_counter()
            trainer.scheduler.step(epoch)
            rows = progress["epoch_rows"]
            entry = {"epoch": epoch, "completed_epoch": epoch + 1, "global_step": trainer.global_step,
                     **_summary(rows), "sources": {str(s.source_id): {"name": s.name,
                         **_summary([r for r in rows if r["source_id"] == s.source_id])} for s in sources}}
            progress["history"].append(entry)
            progress.update(next_epoch=epoch + 1, next_update=0, epoch_rows=[])
            save()
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                save(cpdir / f"epoch_{epoch + 1:04d}.pt")
            print(json.dumps({"event": "epoch", **entry}), flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        # Optimizer OOM may mutate parameters/state; retain the coherent last
        # committed checkpoint rather than serialize the possibly partial step.
        recovery = cpdir / "failure_resume.pt"
        shutil.copyfile(latest, recovery)
        _write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
            "attempted_epoch": progress["next_epoch"], "attempted_update": progress["next_update"],
            "actual_global_step": trainer.global_step, "recovery_checkpoint": str(recovery),
            "recovery_progress": json.loads((output / "progress.json").read_text(encoding="utf-8"))})
        raise
    return {"status": "complete", "global_step": trainer.global_step, **progress}


def load_sources(args, cfg):
    sampling = cfg["sampling"]
    extraction = mixed.ExtractionConfig(**{k: tuple(sampling[k]) if k in ("patch_size", "stride", "groups")
                                          else sampling[k] for k in asdict(mixed.ExtractionConfig())})
    manifest = legacy.protocol.read_p95p80_protocol_manifest(args.ticket03_root / "p95p80_protocol_manifest.json")
    views = [legacy.load_dataset_view(args.input_root, args.ticket03_root, name, manifest,
        patch_size=extraction.patch_size, stride=extraction.stride, t_win=extraction.t_win,
        window_step=extraction.window_step, groups=extraction.groups, delta_frac=extraction.delta_frac)
        for name in legacy.DATASET_ORDER]
    source = MFCPathlineSource(args.mfc_flow, mask_path=args.mfc_mask,
        frame_ranges={"train": (0, 400), "calibration": (400, 501)})
    cache = mixed.MFCReferenceCache(args.mfc_cache, source)
    mfc = mixed.MFCSourceAdapter(source, cache, config=extraction)
    mfc.prepare_sampling_pools()
    print(json.dumps({"event": "source_pools", "source_id": mfc.source_id, "name": mfc.name,
        "cache_prepared_frames": len(cache.prepared_frames),
        "complete_train_windows": len(source.window_starts("train", t_win=extraction.t_win,
                                                          step=extraction.window_step)),
        "sampleable_train_windows": len(mfc.batch_frame_choices),
        "sampleable_frame_starts": list(mfc.batch_frame_choices)}), flush=True)
    sources = [mixed.LegacySourceAdapter(v, config=extraction) for v in views] + [mfc]
    datasets = legacy._dataset_config(views)
    datasets["dataset_order"].append(mfc.name)
    datasets["datasets"].append({"dataset_name": mfc.name, "source_id": mfc.source_id, "source": source.metadata,
        "reference_cache": str(cache.root), "reference_metadata": cache.metadata})
    splits = legacy._split_config(manifest, views)
    splits["dataset_order"].append(mfc.name)
    splits["dataset_train_ranges"][mfc.name] = [[0, 400]]
    sample = legacy._sampling_config(samples_per_epoch=20000, batch_size=sampling["batch_size"], positive_fraction=.5,
                                    **asdict(extraction))
    sample.update(source_schedule="batch_round_robin_epoch_remainder_rotation", source_count=7,
                  legacy_incomplete_endpoint_policy="exclude_known_pseudo_consistency", num_workers=0)
    return sources, datasets, splits, sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache", "config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=6.)
    parser.add_argument("--activation-checkpointing", choices=("config", "on", "off"), default="config",
        help="runtime recomputation setting; config keeps the model config's setting")
    parser.add_argument("--checkpoint-every-updates", type=int, default=25)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.seed != 0 or not 1 <= args.epochs <= 100:
        raise ValueError("This mixed run uses fresh seed 0 and 1..100 epochs")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg["sampling"]["batch_size"] not in (32, 64) or cfg["sampling"]["samples_per_epoch"] != 20000:
        raise ValueError("Mixed training requires batch 32 or 64 and 20000 samples per epoch")
    if cfg["train"].get("amp", False) or cfg["train"].get("data_parallel", False):
        raise ValueError("This entry uses the paired FP32 single-device training configuration")
    if args.resume is None and args.output.exists():
        raise FileExistsError(args.output)
    if args.resume is not None and (not args.output.is_dir()
            or args.resume.resolve().parent != args.output.resolve() / "checkpoints"):
        raise ValueError("Resume must use this existing output's checkpoint directory")
    device = apply_cuda_limit(args.device, args.gpu_memory_limit_gib)
    sources, datasets, splits, sample = load_sources(args, cfg)
    try:
        trainer = create_trainer(cfg, seed=args.seed, device=device)
    except torch.cuda.OutOfMemoryError as exc:
        if args.resume is None:
            args.output.mkdir(parents=True, exist_ok=False)
        _write_json(args.output / "initialization_failure.json", {
            "type": type(exc).__name__, "error": str(exc), "phase": "model_or_ema_initialization",
            "recovery_checkpoint": None if args.resume is None else str(args.resume),
            "gpu_memory_limit_gib": args.gpu_memory_limit_gib})
        raise
    result = run_training(trainer, sources, output=args.output, dataset_config=datasets,
        split_config=splits, sampling_config=sample, epochs=args.epochs, seed=args.seed, device=device,
        resume=args.resume, checkpoint_every_updates=args.checkpoint_every_updates,
        max_seconds=args.max_seconds, config={"model": cfg["model"], "train": cfg["train"]},
        runtime_execution={
            "activation_checkpointing": (cfg["model"].get("encoder_args", {}).get("activation_checkpointing", False)
                if args.activation_checkpointing == "config" else args.activation_checkpointing == "on"),
            "gpu_memory_limit_gib": args.gpu_memory_limit_gib})
    print(json.dumps({"status": result["status"], "global_step": result["global_step"],
                      "completed_epochs": result["next_epoch"], "next_update": result["next_update"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
