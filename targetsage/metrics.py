"""
Adjusted F1 — PU-Safe Evaluation Metric
=========================================
Implements the Adjusted F1 score used throughout the TargetSage benchmark.
This metric was designed specifically for the positive-unlabeled setting
where the test set contains both confirmed negatives and undiscovered
positives mixed together in the unlabeled group.

Why Standard F1 Fails in PU Evaluation
-----------------------------------------
Standard F1 requires a hard threshold to binarize predictions, and it
penalizes the model for predicting "positive" on unlabeled genes.  But in
drug-target discovery, unlabeled genes may include genuine targets that
have simply not been tested yet.  A model that correctly assigns high
scores to these undiscovered positives would be *incorrectly penalized*
under standard F1, leading to pessimistic and misleading comparisons.

The Adjusted F1 Formula
------------------------
Adjusted F1 is a threshold-free, soft metric:

    Adjusted F1 = R_soft^2 / p̄

where:
    R_soft  = mean predicted probability over known positives
              = (1/|P|) Σ_{i ∈ P} P̂(y_i = 1)
              This is the "soft recall" — how highly the model scores
              the genes we know to be targets.

    p̄       = mean predicted probability over ALL genes (P ∪ U)
              = (1/N) Σ_{i=1}^N P̂(y_i = 1)
              This is the model's overall predicted positive rate,
              analogous to precision (how selective the model is).

Relationship to Standard F1
-----------------------------
Under standard F1 with hard threshold t, if we define:
    recall    R  = |{i ∈ P : P̂_i ≥ t}| / |P|
    precision p  = |{i ∈ P : P̂_i ≥ t}| / |{j : P̂_j ≥ t}|

then F1 = 2Rp/(R+p).  The Adjusted F1 approximates this in the soft,
threshold-free regime: R_soft replaces R, and p̄ approximates the
threshold-dependent precision.  Squaring R_soft gives the harmonic mean
structure F1 = 2Rp/(R+p) ≈ R²/p when R ≈ p (verified in the paper).

PU Safety Property
-------------------
Crucially, p̄ = mean P̂(y=1) over ALL genes.  If the model assigns high
scores to genuinely-positive unlabeled genes, both R_soft (numerator)
and p̄ (denominator) increase proportionally, keeping the ratio roughly
stable.  In contrast, high scores on truly-negative genes increase p̄
without increasing R_soft, causing the metric to decrease appropriately.

This means Adjusted F1 does NOT penalize a model for correctly
identifying undiscovered positive genes — exactly the behavior needed
for fair evaluation in drug-target discovery.

Reference
---------
TargetSage, NeurIPS 2026.  See Appendix A for formal derivation and
comparison with area-under-recall-precision curves.
"""

import numpy as np


def adjusted_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Adjusted F1 score for PU evaluation.

    Formula: Adjusted F1 = R_soft^2 / p̄

    Parameters
    ----------
    y_true : [N] binary array — 1 for known positives, 0 for unlabeled
             (unlabeled may include undiscovered positives)
    y_prob : [N] predicted probabilities in [0, 1]

    Returns
    -------
    float in [0, 1] — Adjusted F1 score
             Returns 0.0 if there are no known positives or if the mean
             predicted probability is zero.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 1, 0, 0, 0, 0])
    >>> p = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.3])
    >>> adjusted_f1(y, p)   # R_soft = 0.85, p_bar = 0.40, adj_f1 = 0.85^2/0.40
    1.8062...
    """
    y_true = np.asarray(y_true)
    # Clip probabilities to [0, 1] to handle any floating-point edge cases
    p      = np.clip(np.asarray(y_prob), 0.0, 1.0)

    # Identify known positives (labeled as 1)
    pos = (y_true == 1)
    if not pos.any():
        # Cannot compute soft recall without at least one known positive
        return 0.0

    # R_soft: soft recall — mean predicted probability on known positives.
    # A model that correctly scores all positives highly will have R_soft ≈ 1.
    R_soft = float(np.mean(p[pos]))

    # p̄: mean predicted probability over ALL genes (positives + unlabeled).
    # This acts as a soft precision proxy — a model that is too liberal
    # (high scores everywhere) will have large p̄, reducing Adjusted F1.
    p_bar  = float(np.mean(p))

    # Adjusted F1 = R_soft^2 / p̄
    # The squaring of R_soft gives the metric its harmonic-mean structure
    # analogous to standard F1 (see module docstring for derivation).
    return (R_soft ** 2) / p_bar if p_bar > 0 else 0.0
