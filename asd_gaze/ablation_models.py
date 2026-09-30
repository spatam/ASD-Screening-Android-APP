"""Model variants used by the paper's ablations.

The ablations need only one architectural variant, ``SpatialOnlyTeacher``. It
keeps the frozen ViT-B/16 spatial branch and replaces the temporal branch with
a zero vector (Section VI.A of the paper).

Passing a sequence of zeros to ``TwoStreamTeacher`` *cannot* produce this
replacement. The padding mask ``sequence.abs().sum(-1) == 0`` would be true for
every token, so the ``TransformerEncoder`` would get a fully masked row and
return NaN, which would then spread to the fusion head. In this class the
temporal branch therefore never reaches the encoder.
"""

from __future__ import annotations

import torch

from .model import TwoStreamTeacher


class SpatialOnlyTeacher(TwoStreamTeacher):
    """Teacher with only the spatial branch active.

    It shares modules and ``state_dict`` with ``TwoStreamTeacher``, so each
    class can load checkpoints saved by the other. The random-heatmap control
    relies on this, since by design it reuses the spatial-only checkpoints
    without retraining anything.
    """

    def forward(self, sequence: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_backbone(heatmap)

        # A zero vector replaces the temporal branch and bypasses the encoder.
        # This method never reads the input sequence.
        temporal = torch.zeros(
            spatial.shape[0],
            self.temporal_dim,
            device=spatial.device,
            dtype=spatial.dtype,
        )

        fused = torch.cat([spatial, temporal], dim=1)
        fused = self.fusion_in(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_drop(fused)
        fused = self.fusion_act(fused)
        logits = self.fusion_out(fused)
        return logits.squeeze(1)


__all__ = ['SpatialOnlyTeacher']
