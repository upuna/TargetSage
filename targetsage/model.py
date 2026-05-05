"""
TargetSage Multi-Modal Neural Architecture (Module M3)
======================================================
This module implements the TargetSage discriminator used as the core scoring
model in the TargetSage framework (Module M3: PU-Aware Scoring).

Architecture Overview
---------------------
Three modality-specific MLP heads independently project each input space into
a shared latent dimension d_latent, then a gated fusion layer combines them
into a single fused representation that is passed to a linear discriminator:

    x_bio  (482-dim structured biology)
        └─► head_bio  (MLP)  ──► h_b  [B, d_latent]  ─┐
                                                        ├─► gated fusion ──► z ──► disc ──► logit
    x_attr (23-dim LLM attribute scores from M2)        │     (softmax)
        └─► head_attr (MLP)  ──► h_a  [B, d_latent]  ─┤
                                                        │
    x_emb  (256-dim PCA of 1536-dim LLM embeddings)    │
        └─► head_emb  (MLP)  ──► h_e  [B, d_latent]  ─┘
                                         └── also returned for semantic reweighting

Gated Fusion (default, fusion='gated')
---------------------------------------
Each head produces a scalar gate score via a learned linear projection.  The
three gate scores are passed through a joint softmax so they sum to 1:

    g = softmax( [W_b h_b, W_a h_a, W_e h_e] )   shape [B, 3]

The fused representation is then the weighted sum:

    h = g[:,0] * h_b + g[:,1] * h_a + g[:,2] * h_e   shape [B, d_latent]

This lets the model learn which modality is most informative for each gene and
task, rather than treating all modalities equally.

nnPU Discriminator
------------------
The final layer `disc` is a single linear unit producing an unnormalized logit.
During training this is fed to `nnpu_loss` (see loss.py).  At inference time
sigmoid(logit) is the TargetSage druggability score.

Semantic Reweighting Side-Channel
----------------------------------
`encode()` returns (z, h_e) where h_e is the embedding head's latent vector.
During nnPU training in train.py, h_e is used to compute the cosine similarity
of each unlabeled gene to the centroid of known positives, which forms one
component of the per-sample unlabeled weight w_j.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """
    Single modality encoder: Linear → ReLU → Dropout → Linear → ReLU.

    Parameters
    ----------
    d_in    : input feature dimension
    d_h     : hidden layer width
    d_out   : output latent dimension (= d_latent of parent TargetSage)
    dropout : dropout rate applied after the first ReLU
    """

    def __init__(self, d_in: int, d_h: int, d_out: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),   # project from input space to hidden space
            nn.ReLU(),
            nn.Dropout(dropout),    # regularize during training
            nn.Linear(d_h, d_out),  # project to shared latent space
            nn.ReLU(),              # non-negative latent activations
        )

    def forward(self, x):
        return self.net(x)


class TargetSage(nn.Module):
    """
    TargetSage: three-head gated fusion discriminator for PU learning.

    Parameters
    ----------
    d_bio   : dimension of structured biological features (typically 482)
    d_attr  : dimension of LLM explicit attribute scores (23 after M1/M2)
    d_emb   : dimension of LLM embeddings after PCA (typically 256)
    d_latent: shared latent dimension for all heads (default 256)
    head_h  : width of the hidden layer inside each MLPHead (default 512)
    dropout : dropout probability applied inside each head (default 0.2)
    fusion  : fusion strategy — one of:
                'gated'  (default) softmax-weighted sum of the three heads
                'concat' concatenate then project
                'sum'    elementwise sum then project
                'attn'   scaled dot-product attention (bio query, all keys)

    Notes
    -----
    The paper reports results with fusion='gated' (the default).  'concat' and
    'sum' are provided as ablations.
    """

    def __init__(
        self,
        d_bio: int,
        d_attr: int,
        d_emb: int,
        d_latent: int = 256,
        head_h: int = 512,
        dropout: float = 0.2,
        fusion: str = "gated",
    ):
        super().__init__()
        assert fusion in ("gated", "concat", "sum", "attn"), \
            f"fusion must be gated | concat | sum | attn, got '{fusion}'"
        self.fusion = fusion
        self.d_latent = d_latent

        # Three independent MLP encoders, one per modality
        self.head_bio  = MLPHead(d_bio,  head_h, d_latent, dropout)
        self.head_attr = MLPHead(d_attr, head_h, d_latent, dropout)
        self.head_emb  = MLPHead(d_emb,  head_h, d_latent, dropout)

        if fusion == "gated":
            # Each gate is a learned linear projection from the latent head
            # to a single scalar; the three scalars are jointly softmax-normalized.
            self.gate_b = nn.Linear(d_latent, 1)  # gate for bio head
            self.gate_a = nn.Linear(d_latent, 1)  # gate for attr head
            self.gate_e = nn.Linear(d_latent, 1)  # gate for emb head
            # Post-fusion projection to refine the weighted mixture
            self.fuse_proj = nn.Sequential(
                nn.Linear(d_latent, d_latent), nn.ReLU(), nn.Dropout(dropout)
            )
        elif fusion == "concat":
            # Concatenating three d_latent vectors → 3*d_latent; project back
            self.fuse_proj = nn.Sequential(
                nn.Linear(3 * d_latent, d_latent), nn.ReLU(), nn.Dropout(dropout)
            )
        elif fusion == "attn":
            # Cross-modal attention: bio head acts as query, all heads as keys
            self.attn_q = nn.Linear(d_latent, d_latent)
            self.attn_k = nn.Linear(d_latent, d_latent)
            self.fuse_proj = nn.Sequential(
                nn.Linear(d_latent, d_latent), nn.ReLU(), nn.Dropout(dropout)
            )
        else:  # sum — simplest baseline fusion
            self.fuse_proj = nn.Sequential(
                nn.Linear(d_latent, d_latent), nn.ReLU(), nn.Dropout(dropout)
            )

        # Final discriminator: scalar logit (positive = drug target)
        self.disc = nn.Linear(d_latent, 1)

    def encode(self, x_bio, x_attr, x_emb):
        """
        Encode three modality inputs into a fused latent representation.

        Parameters
        ----------
        x_bio  : [B, d_bio]  structured biological features
        x_attr : [B, d_attr] LLM attribute scores (from M1/M2)
        x_emb  : [B, d_emb]  PCA-compressed LLM embeddings

        Returns
        -------
        z   : [B, d_latent]  fused latent vector, input to discriminator
        h_e : [B, d_latent]  embedding head's latent vector, used for semantic
                              reweighting in train_nnpu() — see train.py
        """
        # Each head independently maps its modality to the shared latent space
        h_b = self.head_bio(x_bio)    # [B, d_latent]
        h_a = self.head_attr(x_attr)  # [B, d_latent]
        h_e = self.head_emb(x_emb)   # [B, d_latent]  (also returned for semantic reweighting)

        if self.fusion == "gated":
            # Compute one scalar gate per head, then normalize with softmax so
            # all three gates sum to 1 — this is a soft modality selector.
            # g[:,0] weights bio, g[:,1] weights attr, g[:,2] weights emb.
            g = torch.softmax(
                torch.cat([self.gate_b(h_b), self.gate_a(h_a), self.gate_e(h_e)], dim=1),
                dim=1,
            )  # shape [B, 3], each row sums to 1
            # Weighted sum of latent vectors
            h = g[:, [0]] * h_b + g[:, [1]] * h_a + g[:, [2]] * h_e  # [B, d_latent]

        elif self.fusion == "concat":
            # Straightforward concatenation; the fuse_proj below reduces dimensionality
            h = torch.cat([h_b, h_a, h_e], dim=1)  # [B, 3*d_latent]

        elif self.fusion == "attn":
            # Bio head queries; all three heads serve as keys and values
            V = torch.stack([h_b, h_a, h_e], dim=1)          # [B, 3, d_latent]
            Q = self.attn_q(h_b).unsqueeze(1)                  # [B, 1, d_latent]
            K = self.attn_k(V)                                  # [B, 3, d_latent]
            # Scaled dot-product attention
            scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) / (self.d_latent ** 0.5)
            w = torch.softmax(scores, dim=1)                   # [B, 3]
            h = (w.unsqueeze(2) * V).sum(dim=1)                # [B, d_latent]

        else:  # sum
            h = h_b + h_a + h_e  # [B, d_latent]

        # Post-fusion non-linear projection
        z = self.fuse_proj(h)  # [B, d_latent]

        # Return both the fused representation z and the embedding head latent h_e.
        # h_e is passed back to the training loop to compute cosine similarity
        # between unlabeled genes and the centroid of known positives (semantic reweighting).
        return z, h_e

    def forward_logits(self, x_bio, x_attr, x_emb):
        """
        Full forward pass returning the discriminator logit.

        Returns
        -------
        logit : [B]          unnormalized score; sigmoid(logit) = P(positive)
        z     : [B, d_latent] fused representation (for downstream use)
        h_e   : [B, d_latent] embedding head output (for semantic reweighting)
        """
        z, h_e = self.encode(x_bio, x_attr, x_emb)
        return self.disc(z).squeeze(1), z, h_e
