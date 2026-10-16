"""P95/P80-based seven-source W3, reusing W1-P and W3 training seams."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict

import torch

import w1_p
import w3
import weak_supervision_contract as contract


@dataclass(frozen=True)
class SevenSourceW3Config:
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    max_embeddings: int = 512
    view_count: int = 2
    projection_dim: int = 64

    def __post_init__(self):
        if self.contrastive_weight < 0 or not torch.isfinite(torch.tensor(self.contrastive_weight)):
            raise ValueError("contrastive_weight must be finite and nonnegative")
        if self.temperature != .1:
            raise ValueError("This experiment uses the existing W3 temperature 0.1")
        if (self.max_embeddings, self.view_count, self.projection_dim) != (512, 2, 64):
            raise ValueError("This experiment uses 512 embeddings, two views, and dimension 64")


class SevenSourceW3Trainer(w1_p.W1PTrainer):
    """Add only a trajectory contrastive objective to the paired W1-P base.

    The student and EMA keys stay identical to the baseline. The reused W3
    adapter is a separate view of the same model, and projection state travels
    in the existing checkpoint extra_metadata alongside full optimizer/RNG state.
    """
    def __init__(self, *args, projection_head, w3_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.w3_config = w3_config or SevenSourceW3Config()
        self.embedding_adapter = w3.TrajectoryEmbeddingAdapter(
            self.student, classification_mode=contract.MODE_W1_P)
        self.projection_head = projection_head
        self.last_w3_metrics = None
        self.identity_epoch_history = []
        self.identity_partial_epoch = None

    def _move_models(self, device):
        super()._move_models(device)
        self.projection_head.to(device)

    def train_step(self, batch, *, epoch, device="cpu"):
        if not isinstance(batch, w1_p.W1PBatch):
            raise TypeError("Seven-source W3 requires an actual P95/P80 W1PBatch")
        epoch = w1_p._strict_nonnegative_int(epoch, name="epoch")
        if batch.sampling_source != self.sampling_source:
            raise ValueError("sampling_source differs from the paired baseline")
        self._move_models(device)
        model_batch = batch.to(device)
        self.student.train()
        self.projection_head.train()
        self.optimizer.zero_grad(set_to_none=True)
        # This first student and teacher forward consume the baseline RNG stream.
        predictions, first = self.embedding_adapter.forward_with_embedding(model_batch)
        with torch.no_grad():
            teacher = w1_p._forward_w1_p_model(self.teacher, model_batch)
        base_loss, stats = w1_p.compute_w1_p_loss(
            predictions, teacher, model_batch, config=self.config, epoch=epoch)
        base_loss.backward()
        del first, predictions, teacher
        # Only two real patches enter the contrastive views. This frees the core
        # full-batch activation graph before building the two small view graphs.
        patch_count = min(2, int(model_batch.labels.shape[0]))
        contrast_batch = w1_p.build_w1_p_batch(
            model_batch.pathlines[:patch_count], model_batch.labels[:patch_count],
            model_batch.label_mask[:patch_count], model_batch.unknown_mask[:patch_count],
            model_batch.solid_mask[:patch_count], invalid_mask=model_batch.invalid_mask[:patch_count],
            sampling_source=model_batch.sampling_source,
            provenance={**model_batch.contract_batch.provenance,
                        "contrastive_subset_patch_indices": list(range(patch_count))},
            dummy_field=model_batch.dummy_field[:patch_count])
        # Additional stochastic view cannot shift the paired next-update RNG.
        embeddings = []
        for view_index in range(2):
            view_seed = (self.seed + 1_000_003 * (self.global_step + 1)
                         + 40_021 * (view_index + 1)) % (2**63 - 1)
            with w3._seeded_view_rng(view_seed, model_batch.pathlines.device):
                _, embedding = self.embedding_adapter.forward_with_embedding(contrast_batch)
            embeddings.append(self.projection_head(embedding))
        reliable = (contrast_batch.label_mask.bool() & ~contrast_batch.solid_mask.bool()
                    & ~contrast_batch.invalid_mask.bool())
        contrast, pairs = w3.compute_trajectory_contrastive_loss(
            embeddings, reliable,
            temperature=self.w3_config.temperature,
            max_embeddings=self.w3_config.max_embeddings)
        ramp = w1_p.ramp_up_weight(epoch, self.config.ramp_up_epochs)
        weighted_contrast = ramp * self.w3_config.contrastive_weight * contrast
        weighted_contrast.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.projection_head.parameters()), self.grad_clip_norm)
        self.optimizer.step()
        w1_p.update_ema_teacher(self.student, self.teacher, decay=self.config.ema_decay)
        self.global_step += 1
        stats.update(pairs)
        stats.update(loss=float(base_loss.detach().cpu()) + float(weighted_contrast.detach().cpu()),
                     contrastive_loss=float(contrast.detach().cpu()), contrastive_patch_count=patch_count,
                     contrastive_ramp_weight=ramp, epoch=epoch, global_step=self.global_step,
                     trajectory_identity_source="known_p95p80_train",
                     haller_anchor_identity_count=0, haller_anchor_identity_source=None,
                     reliable_positive_count=int((reliable & (contrast_batch.labels > .5)).sum()),
                     reliable_negative_count=int((reliable & (contrast_batch.labels <= .5)).sum()),
                     view_count=2, memory_bank_used=False, cross_gpu_gather=False)
        self.last_w3_metrics = copy.deepcopy(stats)
        self._record_identity_stats(stats)
        return stats

    def _record_identity_stats(self, stats):
        epoch = stats["epoch"]
        if self.identity_partial_epoch is None:
            self.identity_partial_epoch = {"epoch": epoch, "updates": 0,
                "contrastive_loss_sum": 0., "max_effective_embedding_count": 0,
                "max_valid_identity_count": 0}
        row = self.identity_partial_epoch
        if row["epoch"] != epoch:
            self._finish_identity_epoch()
            return self._record_identity_stats(stats)
        row["updates"] += 1
        row["contrastive_loss_sum"] += stats["contrastive_loss"]
        for key in ("candidate_identity_count", "valid_identity_count", "effective_embedding_count",
                    "positive_pair_count", "negative_pair_count", "unknown_exclusion_count",
                    "cap_exclusion_count", "reliable_positive_count", "reliable_negative_count"):
            row[key] = row.get(key, 0) + stats[key]
        for key in ("effective_embedding_count", "valid_identity_count"):
            row["max_" + key] = max(row["max_" + key], stats[key])

    def _finish_identity_epoch(self):
        row = self.identity_partial_epoch
        row["contrastive_loss"] = row["contrastive_loss_sum"] / row["updates"]
        row["pair_stats_scope"] = "epoch"
        self.identity_epoch_history.append(row)
        self.identity_partial_epoch = None

    def _checkpoint_extra_metadata(self, extra_metadata):
        extra = super()._checkpoint_extra_metadata(extra_metadata)
        next_epoch = extra.get("mixed_training_progress", {}).get("next_epoch")
        if (self.identity_partial_epoch is not None and next_epoch is not None
                and next_epoch > self.identity_partial_epoch["epoch"]):
            self._finish_identity_epoch()
        extra["seven_source_w3"] = {
            "config": asdict(self.w3_config),
            "projection_state_dict": {key: tensor.detach().cpu().tolist()
                                      for key, tensor in self.projection_head.state_dict().items()},
            "last_metrics": copy.deepcopy(self.last_w3_metrics),
            "identity_epoch_history": copy.deepcopy(self.identity_epoch_history),
            "identity_partial_epoch": copy.deepcopy(self.identity_partial_epoch),
            "trajectory_identity_source": "known_p95p80_train",
            "haller_anchor_identity_source": None,
            "supervision_base": "seven_source_W1-P",
            "final_test_source_read": False,
        }
        return extra

    def load_checkpoint(self, path, **kwargs):
        # Read and validate the branch before the inherited resume changes state.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        branch = saved["extra_metadata"]["seven_source_w3"]
        if branch["config"] != asdict(self.w3_config):
            raise ValueError("Resume W3 objective differs from the saved experiment")
        result = super().load_checkpoint(path, **kwargs)
        projection_state = self.projection_head.state_dict()
        self.projection_head.load_state_dict({key: torch.as_tensor(value,
            dtype=projection_state[key].dtype, device=projection_state[key].device)
            for key, value in branch["projection_state_dict"].items()})
        self.last_w3_metrics = copy.deepcopy(branch["last_metrics"])
        self.identity_epoch_history = copy.deepcopy(branch["identity_epoch_history"])
        self.identity_partial_epoch = copy.deepcopy(branch["identity_partial_epoch"])
        return result


def paired_go_no_go(paired_calibration_rows):
    """Decide from two independent paired seeds on common known support."""
    rows = list(paired_calibration_rows)
    seeds = {r["seed"] for r in rows}
    if seeds != {0, 1} or len(rows) != 2:
        return {"decision": "pending", "reason": "awaiting_two_paired_seed_results", "rows": rows,
                "final_test_source_read": False}
    improved = all(r["w3_macro_f1"] > r["baseline_macro_f1"]
                   and r["w3_macro_iou"] > r["baseline_macro_iou"]
                   and r["w3_worst_dataset_f1"] >= r["baseline_worst_dataset_f1"] for r in rows)
    return {"decision": "go" if improved else "no-go",
            "reason": "repeatable_calibration_gain" if improved else "no_repeatable_gain",
            "w2_applicability": "NA: this branch is W3-on-seven-source-W1-P",
            "rows": rows, "final_test_source_read": False}
