"""
TargetSage — targetsage package
==============================
Public API for the TargetSage model and supporting utilities used in the
TargetSage NeurIPS 2026 paper.

Modules
-------
model   : TargetSage multi-modal neural architecture (M3)
loss    : nnPU (non-negative PU) loss function (Kiryo 2017)
metrics : Adjusted F1 — PU-safe evaluation metric
data    : Data loaders and feature matrix builders

Quick start
-----------
    from targetsage import TargetSage, nnpu_loss, adjusted_f1
"""

from .model import TargetSage, MLPHead
from .loss import nnpu_loss, logistic_loss
from .metrics import adjusted_f1
from .data import (
    TASKS, TASK_DISPLAY, load_bio_features, load_llm_scores,
    load_llm_embeddings, load_labels, load_prior_map,
    build_feature_matrix, get_feature_arrays,
)

__all__ = [
    "TargetSage", "MLPHead",
    "nnpu_loss", "logistic_loss",
    "adjusted_f1",
    "TASKS", "TASK_DISPLAY",
    "load_bio_features", "load_llm_scores", "load_llm_embeddings",
    "load_labels", "load_prior_map",
    "build_feature_matrix", "get_feature_arrays",
]
