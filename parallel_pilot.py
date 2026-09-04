"""WS-9 independent-method CUDA pilot orchestration.

五个前置方法之间没有训练权重依赖，可以由独立 Python 进程分别占用
不同的 multi-GPU group；同一方法的 prescribed batch 不被调用方缩小，
而是由 ``torch.nn.DataParallel`` 在该 group 内 scatter。W3 必须等待 W2
calibration 的 global gate，因此仍在父进程中以 single GPU 执行。worker
只读取 train split 和 calibration Haller GT，不会触碰 test Haller artifact；
父进程在全部 calibration 冻结后才做最终 test evaluation。
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import pathlib
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

import evaluation_report
import e2e_weak_supervision as e2e
import torch
import w2
import weak_supervision_contract as contract


PRE_METHOD_ORDER = (
    contract.MODE_B0,
    contract.MODE_B1,
    contract.MODE_W1_P,
    contract.MODE_W1_H,
    contract.MODE_W2,
)

TRAINING_PROGRESS_FORMAT = "ticket09-training-progress-v1"
TRAINING_PROGRESS_INTERVAL_EPOCHS = 10


def _checkpoint_name(mode: str) -> str:
    return f"{mode.lower().replace('-', '_')}_pilot.pt"


def _emit_epoch(mode: str, epoch: int, epochs: int, stats: Mapping[str, Any]) -> None:
    """让长时 worker 的 stdout/log 可直接显示当前 epoch。"""
    payload = e2e._jsonable({
        "event": "epoch_complete",
        "mode": mode,
        "epoch": epoch,
        "epochs": epochs,
        "steps": stats.get("steps"),
        "loss": stats.get("loss"),
        "global_step": stats.get("global_step"),
    })
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _progress_hash(value: Mapping[str, Any]) -> str:
    """Hash only JSON-safe run identity; tensor/model state stays out of it."""
    import hashlib

    payload = json.dumps(
        e2e._jsonable(dict(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_torch_save(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Write a progress blob atomically so an interrupted write is never resumed."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _progress_directory(output_dir: pathlib.Path, mode: str) -> pathlib.Path:
    return pathlib.Path(output_dir) / "training_progress" / mode.lower().replace("-", "_")


def _latest_progress_path(output_dir: pathlib.Path, mode: str) -> pathlib.Path | None:
    paths = sorted(_progress_directory(output_dir, mode).glob("epoch_*.pt"))
    return paths[-1] if paths else None


def _move_optimizer_state(optimizer: Any, device: str | torch.device) -> None:
    """Keep Adam state on the same primary device as its rebuilt parameters."""
    target = torch.device(device)

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(target)
        if isinstance(value, Mapping):
            return {key: move(child) for key, child in value.items()}
        if isinstance(value, list):
            return [move(child) for child in value]
        if isinstance(value, tuple):
            return tuple(move(child) for child in value)
        return value

    for state_key, state in list(optimizer.state.items()):
        optimizer.state[state_key] = move(state)


def _save_training_progress(
    method: e2e.PilotMethod,
    *,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    path: pathlib.Path,
    context: Mapping[str, Any],
) -> pathlib.Path:
    """Persist complete trainer state without inventing calibration decisions."""
    trainer = method.trainer
    if trainer is None:
        raise TypeError(
            f"{method.mode} training progress 需要 PilotMethod.trainer；"
            "callback-only fake method 不能用于 real parallel pilot"
        )
    student = getattr(trainer, "student", None)
    optimizer = getattr(trainer, "optimizer", None)
    scheduler = getattr(trainer, "scheduler", None)
    if student is None or optimizer is None or scheduler is None:
        raise TypeError(f"{method.mode} trainer 缺少可恢复 student/optimizer/scheduler")
    teacher = getattr(trainer, "teacher", None)
    projection_head = getattr(trainer, "projection_head", None)
    clean_context = e2e._jsonable(dict(context))
    blob = {
        "format_version": TRAINING_PROGRESS_FORMAT,
        "mode": method.mode,
        "epoch": int(epoch),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "seed": int(getattr(trainer, "seed", 0)),
        "metrics": e2e._jsonable(dict(metrics)),
        "history": e2e._jsonable([dict(item) for item in history]),
        "student": contract._component_state(student, name="progress.student"),
        "teacher": (
            None if teacher is None
            else contract._component_state(teacher, name="progress.teacher")
        ),
        "projection_head": (
            None if projection_head is None
            else contract._component_state(
                projection_head, name="progress.projection_head"
            )
        ),
        "optimizer": contract._state_copy(
            optimizer.state_dict(), name="progress.optimizer"
        ),
        "scheduler": contract._state_copy(
            scheduler.state_dict(), name="progress.scheduler"
        ),
        "rng_state": contract._normalize_rng_state(contract.capture_rng_state()),
        "anchor_hash": getattr(trainer, "anchor_hash", None),
        "anchor_metadata": e2e._jsonable(
            getattr(trainer, "anchor_metadata", None)
        ),
        "calibration_selection": e2e._jsonable(
            getattr(trainer, "calibration_selection", None)
        ),
        "context": clean_context,
        "context_hash": _progress_hash(clean_context),
    }
    _atomic_torch_save(path, blob)
    return path


def _load_training_progress(
    method: e2e.PilotMethod,
    *,
    path: pathlib.Path,
    expected_context: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Strictly restore a progress blob before continuing at the next epoch."""
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ValueError("training progress 未通过 safe weights_only 加载") from exc
    if not isinstance(blob, Mapping):
        raise ValueError("training progress 顶层必须是 object")
    if blob.get("format_version") != TRAINING_PROGRESS_FORMAT:
        raise ValueError("training progress format_version 不匹配")
    if blob.get("mode") != method.mode:
        raise ValueError(
            f"training progress mode 不匹配：expected={method.mode!r} "
            f"actual={blob.get('mode')!r}"
        )
    context = blob.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("training progress 缺少 context")
    if blob.get("context_hash") != _progress_hash(context):
        raise ValueError("training progress context_hash 校验失败")
    expected = e2e._jsonable(dict(expected_context))
    if dict(context) != expected:
        raise ValueError("training progress 与当前 pilot 参数/数据/设备组不一致")
    trainer = method.trainer
    if trainer is None:
        raise TypeError(f"{method.mode} training progress 缺少 concrete trainer")
    student = getattr(trainer, "student", None)
    optimizer = getattr(trainer, "optimizer", None)
    scheduler = getattr(trainer, "scheduler", None)
    if student is None or optimizer is None or scheduler is None:
        raise TypeError(f"{method.mode} trainer 缺少可恢复 student/optimizer/scheduler")
    saved_anchor_hash = blob.get("anchor_hash")
    current_anchor_hash = getattr(trainer, "anchor_hash", None)
    if saved_anchor_hash != current_anchor_hash:
        raise ValueError(f"{method.mode} training progress anchor_hash 不一致")
    if e2e._jsonable(blob.get("anchor_metadata")) != e2e._jsonable(
        getattr(trainer, "anchor_metadata", None)
    ):
        raise ValueError(f"{method.mode} training progress anchor_metadata 不一致")
    if e2e._jsonable(blob.get("calibration_selection")) != e2e._jsonable(
        getattr(trainer, "calibration_selection", None)
    ):
        raise ValueError(f"{method.mode} training progress calibration_selection 不一致")
    contract._load_state(student, blob["student"], name="progress.student")
    teacher = getattr(trainer, "teacher", None)
    if (teacher is None) != (blob.get("teacher") is None):
        raise ValueError(f"{method.mode} training progress teacher presence 不一致")
    if teacher is not None:
        contract._load_state(teacher, blob["teacher"], name="progress.teacher")
    projection_head = getattr(trainer, "projection_head", None)
    if (projection_head is None) != (blob.get("projection_head") is None):
        raise ValueError(
            f"{method.mode} training progress projection_head presence 不一致"
        )
    if projection_head is not None:
        contract._load_state(
            projection_head, blob["projection_head"], name="progress.projection_head"
        )
    optimizer.load_state_dict(blob["optimizer"])
    _move_optimizer_state(optimizer, device)
    scheduler.load_state_dict(blob["scheduler"])
    trainer.global_step = int(blob["global_step"])
    trainer.seed = int(blob["seed"])
    trainer.last_metrics = dict(blob["metrics"])
    contract.restore_rng_state(blob["rng_state"], strict_cuda=True)
    history = blob.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("training progress history 必须是非空 list")
    return {
        "epoch": int(blob["epoch"]),
        "global_step": int(blob["global_step"]),
        "metrics": dict(blob["metrics"]),
        "history": [dict(item) for item in history],
        "path": str(path),
    }


def _train_calibrate_checkpoint(
    method: e2e.PilotMethod,
    *,
    config: e2e.PilotConfig,
    checkpoint_path: pathlib.Path,
    final_gate: float | None = None,
    progress_context: Mapping[str, Any] | None = None,
    progress_output_dir: pathlib.Path | None = None,
    resume: bool = False,
    progress_device: str | None = None,
) -> dict[str, Any]:
    """完成一个 method 的 train/calibration/checkpoint round-trip，但不读 test。"""
    mode = method.mode
    if mode == contract.MODE_W3:
        if method.variance_gate is None:
            raise ValueError("W3 必须显式记录预注册的 train-time variance_gate")
        if method.metadata.get("variance_gate_source") not in (
            None,
            "pre_registered_training_config",
        ):
            raise ValueError("W3 training gate 不能来自 calibration selection")

    e2e._seed_everything(config.seed)
    history: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None
    resumed_from: str | None = None
    start_epoch = 1
    if resume:
        if progress_context is None or progress_output_dir is None:
            raise ValueError("resume 需要 progress_context 和 progress_output_dir")
        progress_path = _latest_progress_path(progress_output_dir, mode)
        if progress_path is not None:
            loaded_progress = _load_training_progress(
                method,
                path=progress_path,
                expected_context=progress_context,
                device=(config.device if progress_device is None else progress_device),
            )
            start_epoch = int(loaded_progress["epoch"]) + 1
            history = list(loaded_progress["history"])
            last = dict(loaded_progress["metrics"])
            resumed_from = str(progress_path)
            print(json.dumps({
                "event": "training_resume",
                "mode": mode,
                "epoch": int(loaded_progress["epoch"]),
                "next_epoch": start_epoch,
                "checkpoint": str(progress_path),
            }, ensure_ascii=False, sort_keys=True), flush=True)
    for epoch in range(start_epoch, config.epochs + 1):
        batches = e2e._guarded_batches(method, epoch)
        stats = e2e._ensure_metrics(
            method.train_epoch(batches, epoch), mode=mode, epoch=epoch
        )
        if not batches.seen:
            raise ValueError(f"{mode} epoch={epoch} train_batches 为空")
        history.append(stats)
        last = stats
        _emit_epoch(mode, epoch, config.epochs, stats)
        if (
            progress_context is not None
            and progress_output_dir is not None
            and epoch % TRAINING_PROGRESS_INTERVAL_EPOCHS == 0
        ):
            progress_path = _progress_directory(progress_output_dir, mode) / (
                f"epoch_{epoch:03d}.pt"
            )
            saved_progress = _save_training_progress(
                method,
                epoch=epoch,
                history=history,
                metrics=stats,
                path=progress_path,
                context=progress_context,
            )
            print(json.dumps({
                "event": "training_checkpoint",
                "mode": mode,
                "epoch": epoch,
                "checkpoint": str(saved_progress),
            }, ensure_ascii=False, sort_keys=True), flush=True)
    if last is None:
        raise ValueError(f"{mode} 没有完成任何 epoch")

    threshold, selected = e2e._calibrate_method(method, config)
    if mode == contract.MODE_W2:
        if method.metadata.get("variance_gate_source") not in (
            None,
            "pre_registered_training_config",
        ):
            raise ValueError("W2 training gate 不能来自 calibration selection")
        if method.variance_gate is not None and float(method.variance_gate) != float(
            selected.variance_gate
        ):
            raise ValueError(
                "W2 trainer 的冻结 variance_gate 与 calibration-selected global gate 不一致"
            )
        gate = float(selected.variance_gate)
    else:
        gate = None if final_gate is None else float(final_gate)

    policy = e2e._make_policy(
        method=mode,
        threshold=threshold,
        gate=gate,
        calibration_selection=selected,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saved = method.save_checkpoint(
        checkpoint_path,
        epoch=config.epochs,
        metrics=last,
        calibration_policy=policy,
    )
    saved_path = checkpoint_path if saved is None else pathlib.Path(saved)
    if not saved_path.exists():
        raise FileNotFoundError(f"{mode} save_checkpoint 未生成文件：{saved_path}")
    loaded = method.load_checkpoint(saved_path)
    roundtrip = e2e._validate_roundtrip(
        loaded,
        method=method,
        config=config,
        threshold=threshold,
        gate=gate,
    )
    return {
        "status": "complete",
        "mode": mode,
        "role": method.role,
        "headline_eligible": method.role == "headline_candidate" and mode != contract.MODE_B1,
        "warm_start_aux": method.warm_start_aux,
        "training_variance_gate": method.variance_gate,
        "epochs": config.epochs,
        "seed": config.seed,
        "history": history,
        "final_metrics": last,
        "calibration": e2e._jsonable(selected),
        "prediction_threshold": float(threshold),
        "variance_gate": gate,
        "calibration_policy": policy,
        "checkpoint": str(saved_path),
        "checkpoint_roundtrip": roundtrip,
        "training_progress": {
            "format_version": TRAINING_PROGRESS_FORMAT,
            "interval_epochs": TRAINING_PROGRESS_INTERVAL_EPOCHS,
            "resumed": resumed_from is not None,
            "resumed_from": resumed_from,
            "directory": str(_progress_directory(
                progress_output_dir, mode
            )) if progress_output_dir is not None else None,
        },
        "haller_anchor": (
            None
            if method.anchor_metadata is None
            else e2e._jsonable(dict(method.anchor_metadata))
        ),
    }


def _release_method(
    data: Any,
    methods: dict[str, Any],
    mode: str,
) -> None:
    """释放同一 GPU 上已完成 method 的 trainer/adapter 引用。"""
    methods[mode] = None
    adapter_name = f"_adapter_{mode.replace('-', '_')}"
    if hasattr(data, adapter_name):
        setattr(data, adapter_name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pilot_config_payload(config: e2e.PilotConfig) -> dict[str, Any]:
    return {
        "dataset_config": e2e._jsonable(config.dataset_config),
        "split_config": e2e._jsonable(config.split_config),
        "sampling_config": e2e._jsonable(config.sampling_config),
        "epochs": config.epochs,
        "seed": config.seed,
        "device": config.device,
        "max_steps": config.max_steps,
        "variance_candidates": (
            None
            if config.variance_candidates is None
            else list(config.variance_candidates)
        ),
    }


def _parallel_worker_entry(payload: Mapping[str, Any]) -> None:
    """spawn worker；每个 worker 在一个 multi-GPU group 内顺序训练 methods。"""
    # 延迟导入避免 run_ws9_pilot -> parallel_pilot 的 import cycle。
    import run_ws9_pilot as runner

    worker_id = str(payload["worker_id"])
    device_group = tuple(str(item) for item in payload["device_group"])
    if not device_group:
        raise ValueError(f"parallel worker {worker_id} 缺少 device_group")
    device = device_group[0]
    modes = tuple(str(mode) for mode in payload["modes"])
    result_dir = pathlib.Path(str(payload["result_dir"])) / worker_id
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        primary_device = torch.device(device)
        if primary_device.type == "cuda":
            if primary_device.index is None:
                raise ValueError(f"worker primary device 必须是 cuda:N：{device!r}")
            torch.cuda.set_device(int(primary_device.index))
        print(json.dumps({
            "event": "worker_start",
            "worker": worker_id,
            "device": device,
            "devices": list(device_group),
            "modes": list(modes),
        }, ensure_ascii=False, sort_keys=True), flush=True)
        config_payload = dict(payload["pilot_config"])
        config = e2e.PilotConfig(**config_payload)
        data = runner.RealPilotData(
            weak_root=pathlib.Path(str(payload["weak_root"])),
            haller_root=pathlib.Path(str(payload["haller_root"])),
            model_config=dict(payload["model_config"]),
            manifest_summary=dict(payload["manifest_summary"]),
            batch_size=int(payload["batch_size"]),
            samples_per_epoch=int(payload["samples_per_epoch"]),
            data_workers=int(payload["data_workers"]),
            eval_samples_per_dataset=int(payload["eval_samples_per_dataset"]),
            eval_batch_size=int(payload["eval_batch_size"]),
            seed=int(payload["seed"]),
            device=device,
            ramp_up_epochs=int(payload["ramp_up_epochs"]),
            parallel_devices=device_group,
        )
        train_config = dict(payload["train_config"])
        output_dir = pathlib.Path(str(payload["output_dir"]))
        for mode in modes:
            # Build only the active method so a worker never holds five
            # student/teacher pairs on the same multi-GPU group at once.
            e2e._seed_everything(config.seed)
            methods = data.build_methods(
                pilot_config=config,
                lr=float(train_config["lr"]),
                second_lr=float(train_config["second_lr"]),
                warmup_epochs=int(train_config["warmup_epochs"]),
                weight_decay=float(train_config["weight_decay"]),
                grad_clip=float(train_config["grad_clip"]),
                modes=(mode,),
            )
            method = methods.get(mode)
            if not isinstance(method, e2e.PilotMethod):
                raise TypeError(f"parallel worker 缺少 method={mode}")
            progress_context = dict(payload["progress_context"])
            progress_context.update({
                "mode": mode,
                "device_group": list(device_group),
            })
            result = _train_calibrate_checkpoint(
                method,
                config=config,
                checkpoint_path=output_dir / _checkpoint_name(mode),
                progress_context=progress_context,
                progress_output_dir=output_dir,
                resume=bool(payload["resume"]),
                progress_device=device,
            )
            result["parallel_worker"] = worker_id
            result["parallel_device"] = device
            result["parallel_devices"] = list(device_group)
            e2e._write_json(result_dir / f"{mode}.json", result)
            print(json.dumps({
                "event": "method_complete",
                "mode": mode,
                "worker": worker_id,
                "device": device,
                "devices": list(device_group),
                "checkpoint": result["checkpoint"],
            }, ensure_ascii=False, sort_keys=True), flush=True)
            _release_method(data, methods, mode)
            del method
        e2e._write_json(result_dir / "worker_status.json", {
            "status": "complete",
            "worker": worker_id,
            "device": device,
            "devices": list(device_group),
            "modes": list(modes),
        })
        print(json.dumps({
            "event": "worker_complete",
            "worker": worker_id,
            "device": device,
            "devices": list(device_group),
            "modes": list(modes),
        }, ensure_ascii=False, sort_keys=True), flush=True)
    except BaseException as exc:
        e2e._write_json(result_dir / "worker_status.json", {
            "status": "error",
            "worker": worker_id,
            "device": device,
            "devices": list(device_group),
            "modes": list(modes),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def _typed_calibration(mode: str, value: Mapping[str, Any]) -> Any:
    """从 worker 的 JSON selection 恢复父进程 W3/selection 所需的 typed object。"""
    if mode == contract.MODE_W2:
        return w2.W2CalibrationSelection(
            prediction_threshold=float(value["prediction_threshold"]),
            variance_gate=float(value["variance_gate"]),
            objective_value=float(value["objective_value"]),
            dataset_names=tuple(str(item) for item in value["dataset_names"]),
            record_hashes=tuple(str(item) for item in value["record_hashes"]),
            candidate_count=int(value["candidate_count"]),
            selection_hash=str(value["selection_hash"]),
            objective=str(value.get("objective", "f1")),
            source=str(value.get("source", w2.W2_CALIBRATION_SOURCE)),
        )
    return evaluation_report.ThresholdSelection(
        threshold=float(value.get("threshold", value["prediction_threshold"])),
        objective_value=float(value["objective_value"]),
        dataset_names=tuple(str(item) for item in value["dataset_names"]),
        record_hashes=tuple(str(item) for item in value["record_hashes"]),
        candidate_count=int(value["candidate_count"]),
        metrics=dict(value["metrics"]),
        objective=str(value.get("objective", "f1")),
        source=str(value.get("source", evaluation_report.CALIBRATION_SOURCE)),
    )


def _validate_parallel_devices(devices: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(device).strip() for device in devices if str(device).strip())
    if len(normalized) < 4 or len(normalized) % 2 != 0:
        raise ValueError(
            "parallel pilot 需要偶数个、至少四个 distinct CUDA device；"
            "每个独立 worker 使用两张卡以保持 batch/训练参数不变"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("parallel pilot devices 不能重复")
    if not torch.cuda.is_available():
        raise RuntimeError("parallel pilot 请求 CUDA，但当前 CUDA 不可用")
    count = int(torch.cuda.device_count())
    for device in normalized:
        parsed = torch.device(device)
        if parsed.type != "cuda" or parsed.index is None:
            raise ValueError(f"parallel pilot device 必须是带 index 的 cuda:N：{device!r}")
        if int(parsed.index) >= count:
            raise ValueError(f"parallel pilot device 超出 device_count={count}：{device!r}")
    return normalized


def _make_parallel_groups(
    devices: Sequence[str],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Pair devices, then distribute independent methods across the pairs."""
    normalized = tuple(devices)
    device_groups = [
        tuple(normalized[index:index + 2])
        for index in range(0, len(normalized), 2)
    ]
    mode_groups: list[list[str]] = [[] for _ in device_groups]
    for index, mode in enumerate(PRE_METHOD_ORDER):
        mode_groups[index % len(device_groups)].append(mode)
    return [
        (device_group, tuple(modes))
        for device_group, modes in zip(device_groups, mode_groups)
        if modes
    ]


def run_parallel_pilot(
    *,
    weak_root: pathlib.Path,
    haller_root: pathlib.Path,
    model_config: Mapping[str, Any],
    manifest_summary: Mapping[str, Any],
    pilot_config: e2e.PilotConfig,
    output_dir: pathlib.Path,
    batch_size: int,
    samples_per_epoch: int,
    data_workers: int,
    eval_samples_per_dataset: int,
    eval_batch_size: int,
    seed: int,
    ramp_up_epochs: int,
    train_config: Mapping[str, Any],
    devices: Sequence[str],
    resume: bool = False,
) -> dict[str, Any]:
    """并行运行五个独立前置方法，再单 GPU 运行 W3 并完成最终评价。"""
    import run_ws9_pilot as runner

    normalized_devices = _validate_parallel_devices(devices)
    destination = pathlib.Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result_dir = destination / "parallel_workers"
    result_dir.mkdir(parents=True, exist_ok=True)

    active_groups = _make_parallel_groups(normalized_devices)

    progress_context_base = {
        "format_version": TRAINING_PROGRESS_FORMAT,
        "weak_root": str(pathlib.Path(weak_root).resolve()),
        "haller_root": str(pathlib.Path(haller_root).resolve()),
        "model_config": e2e._jsonable(dict(model_config)),
        "artifact_manifest_hash": manifest_summary.get("manifest_hash"),
        "haller_manifest_hashes": e2e._jsonable(
            manifest_summary.get("haller_manifest_hashes")
        ),
        "pilot_config": _pilot_config_payload(pilot_config),
        "batch_size": int(batch_size),
        "samples_per_epoch": int(samples_per_epoch),
        "data_workers": int(data_workers),
        "eval_samples_per_dataset": int(eval_samples_per_dataset),
        "eval_batch_size": int(eval_batch_size),
        "seed": int(seed),
        "ramp_up_epochs": int(ramp_up_epochs),
        "train_config": e2e._jsonable(dict(train_config)),
        "checkpoint_interval_epochs": TRAINING_PROGRESS_INTERVAL_EPOCHS,
    }

    payload_common = {
        "weak_root": str(pathlib.Path(weak_root).resolve()),
        "haller_root": str(pathlib.Path(haller_root).resolve()),
        "model_config": e2e._jsonable(dict(model_config)),
        "manifest_summary": e2e._jsonable(dict(manifest_summary)),
        "pilot_config": _pilot_config_payload(pilot_config),
        "output_dir": str(destination),
        "batch_size": int(batch_size),
        "samples_per_epoch": int(samples_per_epoch),
        "data_workers": int(data_workers),
        "eval_samples_per_dataset": int(eval_samples_per_dataset),
        "eval_batch_size": int(eval_batch_size),
        "seed": int(seed),
        "ramp_up_epochs": int(ramp_up_epochs),
        "train_config": e2e._jsonable(dict(train_config)),
        "result_dir": str(result_dir),
        "progress_context": progress_context_base,
        "resume": bool(resume),
    }
    context = mp.get_context("spawn")
    processes: list[tuple[str, mp.Process]] = []
    for index, (device_group, modes) in enumerate(active_groups):
        device = device_group[0]
        worker_id = "worker{}_{}".format(
            index, "_".join(item.replace(":", "_") for item in device_group)
        )
        payload = dict(payload_common)
        payload.update({
            "worker_id": worker_id,
            "device": device,
            "device_group": list(device_group),
            "modes": list(modes),
        })
        process = context.Process(
            target=_parallel_worker_entry,
            args=(payload,),
            name=worker_id,
        )
        process.start()
        processes.append((worker_id, process))

    for _worker_id, process in processes:
        process.join()
    failed = [(worker_id, process.exitcode) for worker_id, process in processes
              if process.exitcode != 0]
    if failed:
        for _worker_id, process in processes:
            if process.is_alive():
                process.terminate()
        for _worker_id, process in processes:
            process.join()
        raise RuntimeError(f"parallel pilot worker 失败：{failed!r}")

    worker_results: dict[str, dict[str, Any]] = {}
    for index, (device_group, modes) in enumerate(active_groups):
        worker_id = "worker{}_{}".format(
            index, "_".join(item.replace(":", "_") for item in device_group)
        )
        status = e2e._read_json(result_dir / worker_id / "worker_status.json")
        if status.get("status") != "complete":
            raise RuntimeError(f"parallel worker status 非 complete：{status!r}")
        for mode in modes:
            result = e2e._read_json(result_dir / worker_id / f"{mode}.json")
            if result.get("status") != "complete" or result.get("mode") != mode:
                raise RuntimeError(f"parallel worker result 无效：{worker_id}/{mode}")
            worker_results[mode] = result
    if set(worker_results) != set(PRE_METHOD_ORDER):
        raise RuntimeError("parallel pilot 未覆盖完整五个前置 method")

    calibrations: dict[str, Any] = {
        mode: _typed_calibration(mode, worker_results[mode]["calibration"])
        for mode in PRE_METHOD_ORDER
    }
    thresholds = {
        mode: float(worker_results[mode]["prediction_threshold"])
        for mode in PRE_METHOD_ORDER
    }
    w2_selection = calibrations[contract.MODE_W2]
    w2_gate = float(w2_selection.variance_gate)
    baseline_candidates = (
        (float(calibrations[contract.MODE_W1_H].objective_value), contract.MODE_W1_H),
        (float(calibrations[contract.MODE_W2].objective_value), contract.MODE_W2),
    )
    best_baseline = min(baseline_candidates, key=lambda item: (-item[0], item[1]))[1]
    provisional_selection = e2e.PilotSelection(
        thresholds=thresholds,
        threshold_selections=calibrations,
        w2_selection=w2_selection,
        best_baseline=best_baseline,
    )

    # 父进程只负责 W3 和最终 test；前置 method checkpoint 在最终评价时逐个加载。
    # Each pre-method is rebuilt with the same two-device group that trained
    # it, so the canonical checkpoint state (including adapter.module keys)
    # round-trips without holding all five models in memory simultaneously.
    method_devices = {
        mode: device_group[0]
        for device_group, modes in active_groups
        for mode in modes
    }
    method_parallel_devices = {
        mode: device_group
        for device_group, modes in active_groups
        for mode in modes
    }
    data = runner.RealPilotData(
        weak_root=pathlib.Path(weak_root),
        haller_root=pathlib.Path(haller_root),
        model_config=dict(model_config),
        manifest_summary=dict(manifest_summary),
        batch_size=int(batch_size),
        samples_per_epoch=int(samples_per_epoch),
        data_workers=int(data_workers),
        eval_samples_per_dataset=int(eval_samples_per_dataset),
        eval_batch_size=int(eval_batch_size),
        seed=int(seed),
        device=normalized_devices[0],
        ramp_up_epochs=int(ramp_up_epochs),
        parallel_devices=normalized_devices,
    )
    e2e._seed_everything(pilot_config.seed)
    methods = data.build_methods(
        pilot_config=pilot_config,
        lr=float(train_config["lr"]),
        second_lr=float(train_config["second_lr"]),
        warmup_epochs=int(train_config["warmup_epochs"]),
        weight_decay=float(train_config["weight_decay"]),
        grad_clip=float(train_config["grad_clip"]),
        modes=(contract.MODE_W3,),
        method_devices={contract.MODE_W3: normalized_devices[0]},
        method_parallel_devices={contract.MODE_W3: (normalized_devices[0],)},
    )
    w3_method = e2e._call_method_factory(methods[contract.MODE_W3], provisional_selection)
    w3_result = _train_calibrate_checkpoint(
        w3_method,
        config=pilot_config,
        checkpoint_path=destination / _checkpoint_name(contract.MODE_W3),
        final_gate=w2_gate,
        progress_context={
            **progress_context_base,
            "mode": contract.MODE_W3,
            "device_group": [normalized_devices[0]],
        },
        progress_output_dir=destination,
        resume=bool(resume),
        progress_device=normalized_devices[0],
    )
    calibrations[contract.MODE_W3] = _typed_calibration(
        contract.MODE_W3, w3_result["calibration"]
    )
    thresholds[contract.MODE_W3] = float(w3_result["prediction_threshold"])
    selection = e2e.PilotSelection(
        thresholds=thresholds,
        threshold_selections=calibrations,
        w2_selection=w2_selection,
        best_baseline=best_baseline,
    )

    method_reports: dict[str, Any] = {}
    for mode in e2e.PILOT_METHOD_ORDER:
        gate = w2_gate if mode in {contract.MODE_W2, contract.MODE_W3} else None
        if mode == contract.MODE_W3:
            method = w3_method
            source_result = w3_result
            roundtrip = source_result["checkpoint_roundtrip"]
        else:
            mode_data_methods = data.build_methods(
                pilot_config=pilot_config,
                lr=float(train_config["lr"]),
                second_lr=float(train_config["second_lr"]),
                warmup_epochs=int(train_config["warmup_epochs"]),
                weight_decay=float(train_config["weight_decay"]),
                grad_clip=float(train_config["grad_clip"]),
                modes=(mode,),
                method_devices={mode: method_devices[mode]},
                method_parallel_devices={mode: method_parallel_devices[mode]},
            )
            method = mode_data_methods.get(mode)
            if not isinstance(method, e2e.PilotMethod):
                raise TypeError(f"父进程缺少待评价 method={mode}")
            checkpoint_path = pathlib.Path(worker_results[mode]["checkpoint"])
            loaded = method.load_checkpoint(checkpoint_path)
            roundtrip = e2e._validate_roundtrip(
                loaded,
                method=method,
                config=pilot_config,
                threshold=thresholds[mode],
                gate=gate,
            )
            source_result = worker_results[mode]
        test_records = method.evaluate_test(
            prediction_threshold=thresholds[mode],
            variance_gate=gate,
        )
        test_report = evaluation_report.build_evaluation_report(
            test_records,
            prediction_threshold=thresholds[mode],
            variance_gate=gate,
            method=mode,
            dataset_names=pilot_config.dataset_names,
            checkpoint_epoch=pilot_config.epochs,
        )
        method_reports[mode] = {
            "mode": mode,
            "role": source_result["role"],
            "headline_eligible": bool(source_result["headline_eligible"]),
            "warm_start_aux": bool(source_result["warm_start_aux"]),
            "training_variance_gate": source_result["training_variance_gate"],
            "epochs": pilot_config.epochs,
            "seed": pilot_config.seed,
            "history": source_result["history"],
            "final_metrics": source_result["final_metrics"],
            "calibration": source_result["calibration"],
            "calibration_policy": source_result["calibration_policy"],
            "checkpoint": source_result["checkpoint"],
            "checkpoint_roundtrip": roundtrip,
            "haller_anchor": source_result["haller_anchor"],
            "test": test_report,
            "parallel_device": source_result.get("parallel_device", normalized_devices[0]),
            "parallel_devices": source_result.get(
                "parallel_devices", [normalized_devices[0]
                ] if mode == contract.MODE_W3 else list(method_parallel_devices[mode])
            ),
            "parallel_worker": source_result.get("parallel_worker", "parent-w3"),
        }
        if mode != contract.MODE_W3:
            _release_method(data, mode_data_methods, mode)
            del method

    report = {
        "schema_version": "weak-supervision-pilot-report-v1",
        "pilot": {
            "epochs": pilot_config.epochs,
            "seeds": list(pilot_config.seeds),
            "ramp_up_epochs": pilot_config.ramp_up_epochs,
            "dataset_names": list(pilot_config.dataset_names),
            "device": normalized_devices[0],
            "from_scratch": True,
            "warm_start_aux": False,
            "dataset_config": e2e._jsonable(dict(pilot_config.dataset_config)),
            "split_config": e2e._jsonable(dict(pilot_config.split_config)),
            "sampling_config": e2e._jsonable(dict(pilot_config.sampling_config)),
            "parallel": {
                "enabled": True,
                "devices": list(normalized_devices),
                "groups": {
                    device_group[0]: list(modes)
                    for device_group, modes in active_groups
                },
                "device_groups": {
                    device_group[0]: list(device_group)
                    for device_group, _modes in active_groups
                },
                "pre_methods_parallel": list(PRE_METHOD_ORDER),
                "w3_device": normalized_devices[0],
                "worker_result_dir": str(result_dir),
                "resume": bool(resume),
                "training_progress_format": TRAINING_PROGRESS_FORMAT,
                "training_progress_interval_epochs": (
                    TRAINING_PROGRESS_INTERVAL_EPOCHS
                ),
                "training_progress_root": str(destination / "training_progress"),
            },
        },
        "selection": selection.as_dict(),
        "methods": method_reports,
        "test_label_source": evaluation_report.TEST_SOURCE,
        "haller_literature_status": "pending_verification",
    }
    e2e._write_json(destination / "pilot_report.json", report)
    return report


__all__ = ["PRE_METHOD_ORDER", "run_parallel_pilot"]
