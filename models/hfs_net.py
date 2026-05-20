import torch
import torch.nn as nn


class SelectiveSSMBlock(nn.Module):
    """A lightweight selective state-space style block with linear sequence scan."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.in_proj = nn.Linear(d_model, hidden)
        self.gate_proj = nn.Linear(d_model, hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.norm(x)
        v = torch.tanh(self.in_proj(x))
        g = torch.sigmoid(self.gate_proj(x))
        y = v * g
        y = self.out_proj(self.dropout(y))
        return residual + y


class SequenceEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, d_model: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(d_model, dropout=dropout) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, 2]
        x = self.input_proj(seq)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pool(x).squeeze(-1)
        return x


class StatsEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats)


class CSDGFusion(nn.Module):
    """Cross-space dynamic gating fusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_seq: torch.Tensor, h_stats: torch.Tensor):
        concat = torch.cat([h_seq, h_stats], dim=-1)
        alpha = self.gate(concat)  # [B, 1]
        fused = alpha * h_seq + (1.0 - alpha) * h_stats
        return fused, alpha


class HFSNet(nn.Module):
    def __init__(self, stats_dim: int, num_classes: int, d_model: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seq_encoder = SequenceEncoder(in_dim=2, d_model=d_model, depth=depth, dropout=dropout)
        self.stats_encoder = StatsEncoder(in_dim=stats_dim, out_dim=d_model, dropout=dropout)
        self.fusion = CSDGFusion(dim=d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, seq: torch.Tensor, stats: torch.Tensor):
        h_seq = self.seq_encoder(seq)
        h_stats = self.stats_encoder(stats)
        fused, alpha = self.fusion(h_seq, h_stats)
        logits = self.classifier(fused)
        return logits, alpha
