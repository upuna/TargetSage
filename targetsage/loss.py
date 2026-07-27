"""
nnPU Loss — Non-Negative Positive-Unlabeled Learning
=====================================================
Implements the nnPU risk estimator from Kiryo et al. (2017):
"Positive-Unlabeled Learning with Non-Negative Risk Estimator"
NeurIPS 2017.  https://arxiv.org/abs/1703.00593

Background: PU Learning Setting
---------------------------------
In the TargetSage benchmark genes are labeled either as known positives
(experimentally validated drug targets) or as unlabeled (U).  Unlabeled
genes include both true negatives AND undiscovered positives.  Standard
supervised loss functions incorrectly treat all unlabeled genes as
negatives, which causes the model to suppress high-confidence predictions
on unlabeled genes — exactly the behavior we want to allow.

The unbiased PU risk estimator (uPU) decomposes the expected risk over
the full population into three computable terms using only P and U sets:

    R(f) = π · R_P^+  +  R_U^-  −  π · R_P^-

where:
    π      = P(Y=1), the class prior (fraction of true positives in genome)
    R_P^+  = E_{x~P}[ ℓ(f(x),  1) ]   — loss on positives labeled as positive
    R_U^-  = E_{x~U}[ ℓ(f(x),  0) ]   — loss on unlabeled labeled as negative
    R_P^-  = E_{x~P}[ ℓ(f(x),  0) ]   — loss on positives labeled as negative
              (this term is subtracted to de-bias the unlabeled risk)

Non-Negative Clamp (nnPU)
---------------------------
A well-known problem with uPU is that the term (R_U^- − π·R_P^-) can
become negative when the model overfits on positives, producing a
negative overall loss that triggers large negative gradient updates and
drives the model to collapse (outputting constant predictions).

The nnPU fix is a single non-negative clamp on the unlabeled risk term:

    risk_u = max(0,  R_U^- − π·R_P^-)

This prevents negative gradient propagation through the unlabeled risk
while leaving the positive risk term unaffected.  It is equivalent to
saying: "if the model is already doing too well on unlabeled data relative
to what the prior predicts, do not further penalize it".

Semantic Reweighting
----------------------
TargetSage extends nnPU with per-sample weights w_j on the unlabeled set.
The weighted unlabeled risk becomes:

    R_U^-(w) = Σ_j (1-w_j) · ℓ(f(x_j), 0) / Σ_j (1-w_j)

where w_j = β · sigmoid(D(z_j)) + (1-β) · cos(h_e_j, centroid_pos).
A gene resembling known positives has a high w_j and therefore a small (1-w_j)
weight in the unlabeled negative risk, i.e. it is spared harsh penalization.
See train.py::train_nnpu() for the full implementation.

Hybrid Class Prior
-------------------
The class prior π used in nnpu_loss is a convex combination of a
data-driven Elkan-Noto estimate and an LLM-derived estimate:

    π = α · π_data + (1-α) · π_llm    (α = 0.6 by default)

This is computed before calling nnpu_loss; see train.py::main() for details.
"""

from typing import Optional
import torch
import torch.nn.functional as F


def logistic_loss(logits, y01):
    """
    Binary cross-entropy written in terms of logits for numerical stability.

    Equivalent to BCEWithLogitsLoss but returns per-sample values.

    ℓ(logit, y) = log(1 + exp(logit)) - y * logit
                = softplus(logit) - y * logit

    Parameters
    ----------
    logits : [N] unnormalized model outputs
    y01    : [N] binary targets in {0, 1} (can be float tensors)

    Returns
    -------
    [N] per-sample loss values (always non-negative)
    """
    return F.softplus(logits) - y01 * logits


def nnpu_loss(
    logits_p: torch.Tensor,
    logits_u: torch.Tensor,
    pi: float,
    w_u: Optional[torch.Tensor] = None,
    non_negative: bool = True,
) -> torch.Tensor:
    """
    Non-negative PU risk estimator (Kiryo et al., 2017).

    Full formula:
        L = π · R_P^+  +  max(0,  R_U^-(w) − π · R_P^-)

    where the max(0, ...) clamp is applied only when non_negative=True
    (nnPU).  Setting non_negative=False recovers the original uPU estimator.

    Parameters
    ----------
    logits_p     : [n_p]   logits on labeled-positive samples
    logits_u     : [n_u]   logits on unlabeled samples
    pi           : scalar  class prior P(Y=1), estimated by hybrid method
    w_u          : [n_u]   optional per-sample weights for unlabeled samples;
                            if None, all unlabeled samples are weighted equally
    non_negative : bool    True → nnPU (non-negative clamp applied)
                           False → uPU (may produce negative loss)

    Returns
    -------
    scalar loss (single tensor, differentiable)

    Notes
    -----
    The three risk components:

    R_P^+  — mean logistic loss treating positives as positives (y=1).
              This term encourages the model to assign high scores to labeled
              positives.

    R_P^-  — mean logistic loss treating positives as negatives (y=0).
              Subtracted (scaled by π) to remove the contribution of the true
              positives hiding inside the unlabeled set from R_U^-.

    R_U^-  — (weighted) mean logistic loss treating all unlabeled as negatives.
              Contains both true negatives (we want this term to be small) and
              undiscovered positives (those should contribute zero after the
              π·R_P^- subtraction in expectation).
    """
    # Target tensors for the three loss terms
    y1   = torch.ones_like(logits_p)    # positives → treated as class 1
    y0_p = torch.zeros_like(logits_p)   # positives → treated as class 0 (de-bias)
    y0_u = torch.zeros_like(logits_u)   # unlabeled → treated as class 0

    # R_P^+ : encourage model to score labeled positives highly
    Rp_pos = logistic_loss(logits_p, y1).mean()

    # R_P^- : cost of mis-labeling known positives as negative;
    #         subtracted below to correct for positives hidden in U
    Rp_neg = logistic_loss(logits_p, y0_p).mean()

    # R_U^-(w) : (optionally weighted) loss on unlabeled samples
    Lu = logistic_loss(logits_u, y0_u)
    if w_u is not None:
        # Weighted mean with (1 - w_u): unlabeled genes that most resemble known
        # positives (high w_u) get the SMALLEST negative weight, so likely hidden
        # positives are not penalized hard for receiving a high score; genes
        # dissimilar to positives (low w_u) are treated as reliable negatives.
        Ru_neg = (Lu * (1.0 - w_u)).sum() / ((1.0 - w_u).sum() + 1e-8)
    else:
        Ru_neg = Lu.mean()

    # Unlabeled risk: subtract the de-bias term π·R_P^-
    risk_u = Ru_neg - float(pi) * Rp_neg

    if non_negative:
        # Non-negative clamp: prevents gradient collapse when the model
        # has already learned to discriminate better than the prior predicts.
        # Without this clamp (uPU), a negative risk_u creates a negative loss
        # whose gradient pushes the model toward a trivial constant solution.
        risk_u = torch.clamp(risk_u, min=0.0)

    # Full nnPU objective
    return float(pi) * Rp_pos + risk_u
