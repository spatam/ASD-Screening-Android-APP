from __future__ import annotations

import torch
import torch.nn as nn
import timm


def _create_student_backbone(pretrained: bool) -> nn.Module:
    last_error: Exception | None = None
    for name in ['mobilenetv3_small_100', 'mobilenetv3_small_075']:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool='avg')
        except Exception as error:
            last_error = error
    raise RuntimeError(f'Could not create MobileNetV3-Small from timm: {last_error}')


class StudentASD(nn.Module):
    def __init__(self, seq_feature_dim: int = 3, hidden: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        self.spatial_backbone = _create_student_backbone(pretrained=pretrained)
        # timm exposes num_features=576 for MobileNetV3-Small, but the pooled
        # forward output used with num_classes=0 is 1024-dimensional.
        spatial_dim = 1024
        self.spatial_proj = nn.Linear(spatial_dim, hidden)

        self.temporal_proj = nn.Linear(seq_feature_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dim_feedforward=128,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.temporal_out = nn.Linear(32, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_proj(self.spatial_backbone(heatmap))

        temporal = self.temporal_proj(sequence)
        padding_mask = sequence.abs().sum(dim=-1) == 0
        temporal = self.temporal_encoder(temporal, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        temporal = (temporal * valid).sum(dim=1) / denom
        temporal = self.temporal_out(temporal)

        logits = self.head(torch.cat([spatial, temporal], dim=1))
        return logits.squeeze(1)


__all__ = ['StudentASD']
