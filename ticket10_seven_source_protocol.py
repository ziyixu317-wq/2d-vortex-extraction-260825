"""Readable development execution record for the seven-source mainline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

SOURCES = ["boussinesq", "cylinder2d", "doublegyre2d", "fourcenters2d",
           "jungtelziemniak2d", "pipedcylinder2d", "mfc_airfoil"]
SAMPLING_STAGES = [
    {"epoch_range": [0, 25], "batch_size": 32, "updates_per_epoch": 625,
     "last_batch_size": 32, "source_rotation": "2 * absolute_epoch mod 7"},
    {"epoch_range": [25, 100], "batch_size": 64, "updates_per_epoch": 313,
     "last_batch_size": 32, "source_rotation": "5 * absolute_epoch mod 7"},
]


def stage_config(config, batch_size):
    result = copy.deepcopy(config)
    result["sampling"]["batch_size"] = batch_size
    return result


def build_staged_epoch_specs(sources, **kwargs):
    from mfc_mixed_w1p import build_epoch_specs
    kwargs["batch_size"] = 32 if kwargs["epoch"] < 25 else 64
    return build_epoch_specs(sources, **kwargs)


def execution_record(config):
    """Register choices before new seeds; no final-test access is requested."""
    from w1_p import W1PConfig
    from weak_supervision_contract import FEATURE_SCHEMA_7
    c = config["train"]
    return {
        "record_version": "ticket10-seven-source-development-v1",
        "scope": "seven-source train/calibration development",
        "source_order": SOURCES, "seeds": [0, 1, 2], "epochs": 100,
        "samples_per_epoch": 20000, "total_samples": 2000000, "total_updates": 39100,
        "sampling_stages": copy.deepcopy(SAMPLING_STAGES),
        "sampling": copy.deepcopy(config["sampling"]),
        "scientific_configuration": {k: copy.deepcopy(config[k]) for k in ("model", "train")},
        "feature_schema": FEATURE_SCHEMA_7.as_dict(),
        "optimizer": {"class": "torch.optim.AdamW", "constructor": {
            "lr": c["lr"], "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": c["weight_decay"], "amsgrad": False, "maximize": False,
            "foreach": None, "capturable": False, "differentiable": False, "fused": None},
            "parameters": "all student parameters; one parameter group"},
        "scheduler": {"class": "train_kaggle.TwoStepLR", "constructor": {
            "lr": c["lr"], "second_lr": c["second_lr"], "warmup_epochs": c["warmup_epochs"]},
            "call": "step(zero-based absolute epoch) after completed epoch",
            "first_lower_lr_completed_epoch": 62},
        "initialization": "fresh Python/NumPy/Torch/CUDA seed; model default initialization; EMA deep clone",
        "w1_p": W1PConfig(ramp_up_epochs=12).as_dict(),
        "grad_clip_norm": c["grad_clip"], "loss": "known BCE + ramp*(pseudo BCE + teacher consistency)",
        "normalization": "independent source-specific train-only statistics",
        "legacy_source_artifact": "Ticket03 standard-IVD/P95P80 manifest and source-specific train/calibration ranges",
        "mfc": {"train": [0, 400], "calibration": [400, 501], "native_dt": 0.005,
            "reference": "full501 shared standard-IVD five-state cache; train-only P95/P80 and statistics",
            "haller": "unavailable: full native600s timed out; no MFC Hybrid/W2 claim"},
        "unknown_semantics": "unknown/solid/invalid/failed paths excluded by existing masks; no negative conversion",
        "integration": "actual stored end-start span/(L-1); legacy endpoint failures excluded",
        "methods": {"unconditional": ["W1-P"], "conditional": {"W3": "Ticket09 paired seven-source seed0/1 go/no-go"},
            "historical_only": ["six-source W1-Hybrid", "six-source W2-on-Hybrid", "six-source adaptive W2", "B0"]},
        "probability_policy": {"global_probability_threshold": 0.1,
            "selection": "registered before fresh seed1/2; inherits prior calibration-only global point",
            "candidate_rule_if_future_selection_authorized": "common observed-known support; source-macro F1, then IoU, then worst-source F1, then registered order",
            "per_dataset_retune": False, "test_retune": False},
        "variance_gate": {"value": 0.001, "role": "historical three-view diagnostic only; absent from W1-P loss"},
        "report_support": "direct observed-known support; weak-label agreement and missing coverage separately",
        "final_test": {"sealed": True, "source_read": False, "existing_guard": "retained; Ticket12 development report only"},
        "runtime": {"device": "single visible physical GPU, logical cuda:0", "gpu_memory_limit_gib": 18,
                    "activation_checkpointing": False, "amp": False, "data_parallel": False, "num_workers": 0},
        "seed0_reuse": {"checkpoint": "/data/xuziyi/mainline-mfc-mixed-w1p-v6/artifacts/mixed-w1p-seed0-stage64-v1/checkpoints/epoch_0100.pt",
            "constructor_evidence": "original run_mfc_mixed_w1p_training.create_trainer and JSON config; constructor not originally saved in checkpoint",
            "training_replay": 0},
        "literature": {"VortexTransformer": ["T76G9Z3A", "D8QQDR55"]},
    }


def audit_seed0(checkpoint, config):
    """Read an existing completed checkpoint and expose evidence for reuse."""
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    p = saved["extra_metadata"]["mixed_training_progress"]
    history = p["history"]
    expected = {k: config[k] for k in ("model", "train")}
    stages = p.get("sampling_stages", [])
    groups = saved["optimizer"]["param_groups"]
    matched = (saved["mode"] == "W1-P" and saved["seed"] == 0 and saved["epoch"] == 100 and saved["global_step"] == 39100
        and p["next_epoch"] == 100 and p["next_update"] == 0 and not p["epoch_rows"]
        and len(history) == 100 and sum(h["samples"] for h in history) == 2000000
        and saved["extra_metadata"]["training_config"] == expected
        and saved["dataset_config"]["dataset_order"] == SOURCES
        and all(k in saved for k in ("student", "teacher", "optimizer", "scheduler", "rng_state"))
        and saved["scheduler"]["epoch"] == 99
        and len(groups) == 1 and groups[0]["betas"] == (0.9, 0.999)
        and groups[0]["eps"] == 1e-8 and groups[0]["weight_decay"] == config["train"]["weight_decay"]
        and not saved["split_config"]["test_source_read"]
        and [(s["epoch_range"], s["batch_size"], s["updates_per_epoch"]) for s in stages]
            == [([0, 25], 32, 625), ([25, 100], 64, 313)]
        and all(h["epoch"] == i and h["samples"] == 20000
                and h["updates"] == (625 if i < 25 else 313) for i, h in enumerate(history)))
    return {"reusable": matched, "checkpoint": str(Path(checkpoint).resolve()), "seed": saved["seed"],
            "epochs": saved["epoch"], "global_step": saved["global_step"], "history_epochs": len(history),
            "total_samples": sum(h["samples"] for h in history), "sampling_stages": stages,
            "split_config": saved["split_config"],
            "source_totals": {name: {key: sum(h["sources"][str(i)][key] for h in history)
                for key in ("updates", "samples", "failed_pathlines")} for i, name in enumerate(SOURCES)},
            "state_fields": [k for k in ("student", "teacher", "optimizer", "scheduler", "rng_state") if k in saved]}


def checkpoint_record(record):
    """Store training constructors while keeping access declarations in the sidecar.

    The existing closure rejects test-only keys, including sealed declarations.
    Its guard remains intact; the full readable record lives beside checkpoints.
    """
    keys = ("record_version", "scope", "source_order", "seeds", "epochs", "samples_per_epoch",
            "total_samples", "total_updates", "sampling_stages", "sampling", "optimizer",
            "scheduler", "initialization", "w1_p", "grad_clip_norm", "loss", "normalization",
            "legacy_source_artifact", "mfc", "unknown_semantics", "integration", "runtime", "execution")
    result = {k: copy.deepcopy(record[k]) for k in keys if k in record}
    result["global_probability_threshold"] = record["probability_policy"]["global_probability_threshold"]
    return result


def write_record(path, config):
    Path(path).write_text(json.dumps(execution_record(config), indent=2, ensure_ascii=False), encoding="utf-8")
