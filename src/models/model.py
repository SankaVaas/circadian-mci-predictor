"""
src/models/model.py
────────────────────────────────────────────────────────────────────────────────
CircadianMCIPredictor — Two-Stream Architecture
circadian-mci-predictor | PhD Research

Architecture overview:
    Stream A │ Behavioral GRU   → raw + rolling phone metadata
    Stream B │ Circadian MLP    → circadian phase features (NOVEL)
    Fusion   │ Concat → LayerNorm → Dropout → Linear → Sigmoid

~48,000 parameters — fully trainable on CPU or Colab free T4.

Usage:
    from src.models.model import CircadianMCIPredictor, BehavioralOnlyBaseline
    model = CircadianMCIPredictor(n_behavioral=15, n_circadian=7)
    logit = model(behavioral_seq, circadian_seq)   # (B,1)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CircadianMCIPredictor", "BehavioralOnlyBaseline"]


# ── Helper: weight init ───────────────────────────────────────────────────────
def _xavier_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)


# ── Primary model ─────────────────────────────────────────────────────────────
class CircadianMCIPredictor(nn.Module):
    """
    Two-stream model for MCI prediction from passive smartphone data.

    Stream A — Behavioral GRU
        Processes a sequence of daily behavioral feature vectors
        (screen usage, app patterns, location entropy, steps, calls)
        through a stacked GRU to capture temporal dependencies.

    Stream B — Circadian Encoder  [NOVEL COMPONENT]
        Processes per-window aggregated circadian features
        (phase estimate, amplitude, R², regularity score, drift rate)
        through an MLP. These features are derived from the cosine
        phase estimator and are not found in prior MCI prediction work.

    Fusion
        Concatenates both stream outputs and passes through a small
        classifier head with dropout regularisation.

    Parameters
    ----------
    n_behavioral : int    Number of behavioral input features.
    n_circadian  : int    Number of circadian input features.
    gru_hidden   : int    GRU hidden state size (default 64).
    gru_layers   : int    Number of stacked GRU layers (default 2).
    circ_hidden  : int    Circadian MLP hidden size (default 32).
    dropout      : float  Dropout probability (default 0.3).
    """

    def __init__(
        self,
        n_behavioral: int = 15,
        n_circadian:  int = 7,
        gru_hidden:   int = 64,
        gru_layers:   int = 2,
        circ_hidden:  int = 32,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()

        # ── Stream A: Behavioral GRU ──────────────────────────────────────────
        self.behavioral_norm = nn.LayerNorm(n_behavioral)
        self.gru = nn.GRU(
            input_size=n_behavioral,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Stream B: Circadian Encoder ───────────────────────────────────────
        self.circadian_encoder = nn.Sequential(
            nn.LayerNorm(n_circadian),
            nn.Linear(n_circadian, circ_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(circ_hidden, circ_hidden),
            nn.GELU(),
        )

        # ── Fusion classifier ─────────────────────────────────────────────────
        fusion_dim = gru_hidden + circ_hidden
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        _xavier_init(self)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        behavioral_seq: torch.Tensor,
        circadian_seq:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        behavioral_seq : (batch, seq_len, n_behavioral)
        circadian_seq  : (batch, seq_len, n_circadian)

        Returns
        -------
        probability : (batch, 1)  — P(MCI within prediction horizon)
        """
        # Stream A — use final hidden state of top GRU layer
        b_norm = self.behavioral_norm(behavioral_seq)
        _, h_n = self.gru(b_norm)
        h_behavioral = h_n[-1]                      # (batch, gru_hidden)

        # Stream B — mean-pool circadian features over the window
        circ_mean    = circadian_seq.mean(dim=1)    # (batch, n_circadian)
        h_circadian  = self.circadian_encoder(circ_mean)  # (batch, circ_hidden)

        # Fusion
        fused = torch.cat([h_behavioral, h_circadian], dim=1)
        return self.classifier(fused)

    # ── Utility ───────────────────────────────────────────────────────────────
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"CircadianMCIPredictor("
            f"params={self.n_params:,}, "
            f"gru={self.gru.input_size}→{self.gru.hidden_size}×{self.gru.num_layers}, "
            f"circ_encoder={self.circadian_encoder[1].in_features}→"
            f"{self.circadian_encoder[1].out_features})"
        )