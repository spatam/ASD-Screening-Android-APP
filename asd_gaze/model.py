from __future__ import annotations

import torch
import torch.nn as nn
import timm


def _create_backbone(candidates: list[str], pretrained: bool) -> nn.Module:
    last_error: Exception | None = None
    for name in candidates:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0)
        except Exception as error:
            last_error = error
    raise RuntimeError(f'Could not create a timm backbone from {candidates}: {last_error}')


class TwoStreamTeacher(nn.Module):
    def __init__(
        self,
        seq_feature_dim: int = 3,
        temporal_dim: int = 64,
        temporal_layers: int = 3,
        temporal_heads: int = 4,
        temporal_ff_dim: int = 128,
        fusion_dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_backbone = _create_backbone(
            ['vit_base_patch16_224.augreg_in21k', 'vit_base_patch16_224'],
            pretrained=pretrained,
        )
        for parameter in self.spatial_backbone.parameters():
            parameter.requires_grad = False

        self.temporal_dim = temporal_dim
        self.temporal_proj = nn.Linear(seq_feature_dim, temporal_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=temporal_heads,
            dim_feedforward=temporal_ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

        self.fusion_in = nn.Linear(768 + temporal_dim, 64)
        self.fusion_norm = nn.LayerNorm(64)
        self.fusion_drop = nn.Dropout(fusion_dropout)
        self.fusion_act = nn.GELU()
        self.fusion_out = nn.Linear(64, 1)

    def forward(self, sequence: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_backbone(heatmap)
        temporal = self.temporal_proj(sequence)

        padding_mask = sequence.abs().sum(dim=-1) == 0
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        temporal = (temporal * valid).sum(dim=1) / denom

        fused = torch.cat([spatial, temporal], dim=1)
        fused = self.fusion_in(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_drop(fused)
        fused = self.fusion_act(fused)
        logits = self.fusion_out(fused)
        return logits.squeeze(1)


TwoStreamASD = TwoStreamTeacher
