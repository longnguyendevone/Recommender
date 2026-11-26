"""Contrastive Learning Sequential Recommender (CL4SRec) on SpatialTemporalSASRec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sequential import SequentialRecommender, SpatialTemporalSASRec


class CL4SRecRecommender(SequentialRecommender):
    """Extends SpatialTemporalSASRec with crop/mask/reorder trajectory contrastive learning."""

    name = "sequential_cl"

    def __init__(
        self,
        cl_weight: float = 0.1,
        temperature: float = 0.2,
        crop_ratio: float = 0.6,
        mask_ratio: float = 0.2,
        reorder_ratio: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cl_weight = float(cl_weight)
        self.temperature = float(temperature)
        self.crop_ratio = float(crop_ratio)
        self.mask_ratio = float(mask_ratio)
        self.reorder_ratio = float(reorder_ratio)

    def _build_network(self, num_pois: int, num_cats: int) -> SpatialTemporalSASRec:
        net = super()._build_network(num_pois, num_cats)
        net.cl_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        ).to(self.device)
        return net

    def _augment_view(
        self,
        poi_seq: torch.Tensor,
        cat_seq: torch.Tensor,
        time_feats: torch.Tensor,
        delta_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized batch trajectory augmentations (crop, mask, local reorder) on GPU."""
        bsz, seq_len = poi_seq.shape
        device = poi_seq.device

        valid_mask = poi_seq != 0
        valid_len = valid_mask.sum(dim=1)  # (B,)
        ops = torch.randint(0, 3, (bsz,), device=device)
        # Require valid_len >= 3 for reorder, otherwise fall back to mask.
        ops = torch.where((ops == 2) & (valid_len < 3), torch.ones_like(ops), ops)
        can_aug = valid_len >= 2

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        # 1) Crop view (op == 0)
        keep_len = torch.clamp((valid_len.float() * self.crop_ratio).round().long(), min=1)
        max_off = torch.clamp(valid_len - keep_len, min=0)
        off = (torch.rand(bsz, device=device) * (max_off.float() + 1.0)).long()
        off = torch.minimum(off, max_off)
        start_base = seq_len - valid_len
        dst_start = seq_len - keep_len.unsqueeze(1)
        crop_keep_mask = pos >= dst_start
        src_idx = torch.clamp(start_base.unsqueeze(1) + off.unsqueeze(1) + (pos - dst_start), 0, seq_len - 1)

        p_crop = torch.where(crop_keep_mask, poi_seq.gather(1, src_idx), torch.zeros_like(poi_seq))
        c_crop = torch.where(crop_keep_mask, cat_seq.gather(1, src_idx), torch.zeros_like(cat_seq))
        t_crop = torch.where(
            crop_keep_mask.unsqueeze(-1),
            time_feats.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, 4)),
            torch.zeros_like(time_feats),
        )
        d_crop = torch.where(
            crop_keep_mask.unsqueeze(-1),
            delta_feats.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, 2)),
            torch.zeros_like(delta_feats),
        )

        # 2) Mask view (op == 1)
        is_last = pos == (seq_len - 1)
        mask_tokens = valid_mask & (~is_last) & (torch.rand((bsz, seq_len), device=device) < self.mask_ratio)
        p_mask = torch.where(mask_tokens, torch.ones_like(poi_seq), poi_seq)
        c_mask = torch.where(mask_tokens, torch.ones_like(cat_seq), cat_seq)

        # 3) Local reorder view (op == 2)
        pos_f = pos.float()
        jitter = torch.randn((bsz, seq_len), device=device) * max(self.reorder_ratio * 3.0, 0.5)
        keys = torch.where(valid_mask, pos_f + jitter, pos_f - 10000.0)
        perm_idx = torch.argsort(keys, dim=1)
        p_reord = poi_seq.gather(1, perm_idx)
        c_reord = cat_seq.gather(1, perm_idx)
        t_reord = time_feats.gather(1, perm_idx.unsqueeze(-1).expand(-1, -1, 4))
        d_reord = delta_feats.gather(1, perm_idx.unsqueeze(-1).expand(-1, -1, 2))

        use_crop = (can_aug & (ops == 0)).unsqueeze(1)
        use_mask = (can_aug & (ops == 1)).unsqueeze(1)
        use_reord = (can_aug & (ops == 2)).unsqueeze(1)

        p_out = torch.where(use_crop, p_crop, torch.where(use_mask, p_mask, torch.where(use_reord, p_reord, poi_seq)))
        c_out = torch.where(use_crop, c_crop, torch.where(use_mask, c_mask, torch.where(use_reord, c_reord, cat_seq)))
        t_out = torch.where(use_crop.unsqueeze(-1), t_crop, torch.where(use_reord.unsqueeze(-1), t_reord, time_feats))
        d_out = torch.where(use_crop.unsqueeze(-1), d_crop, torch.where(use_reord.unsqueeze(-1), d_reord, delta_feats))
        return p_out, c_out, t_out, d_out

    def _compute_step_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.net is not None
        poi_seq = batch["poi_seq"].to(self.device, non_blocking=True)
        cat_seq = batch["cat_seq"].to(self.device, non_blocking=True)
        time_feats = batch["time_feats"].to(self.device, non_blocking=True)
        delta_feats = batch["delta_feats"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        negatives = batch["negatives"].to(self.device, non_blocking=True)

        h_seq = self.net.encode_sequence(poi_seq, cat_seq, time_feats, delta_feats)
        rec_loss = self._compute_rec_loss(h_seq, targets, negatives)

        v1 = self._augment_view(poi_seq, cat_seq, time_feats, delta_feats)
        v2 = self._augment_view(poi_seq, cat_seq, time_feats, delta_feats)
        h1 = self.net.encode_last(*v1)
        h2 = self.net.encode_last(*v2)

        z1 = F.normalize(self.net.cl_proj(h1).float(), dim=-1)
        z2 = F.normalize(self.net.cl_proj(h2).float(), dim=-1)
        sim = torch.matmul(z1, z2.T) / max(self.temperature, 1e-4)
        labels = torch.arange(sim.size(0), device=sim.device)
        cl_loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))

        total_loss = rec_loss + self.cl_weight * cl_loss
        return total_loss, {
            "rec_loss": float(rec_loss.detach().item()),
            "cl_loss": float(cl_loss.detach().item()),
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._checkpoint_payload()
        payload["cl_config"] = {
            "cl_weight": self.cl_weight,
            "temperature": self.temperature,
            "crop_ratio": self.crop_ratio,
            "mask_ratio": self.mask_ratio,
            "reorder_ratio": self.reorder_ratio,
        }
        return payload

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "CL4SRecRecommender":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        cl_cfg = ckpt.get("cl_config", {})
        obj = cls(**cl_cfg, **cfg, device=device)
        obj.num_pois = int(ckpt["num_pois"])
        obj.num_cats = int(ckpt["num_cats"])
        obj.poi_indices = np.asarray(ckpt["poi_indices"], dtype=np.int64)
        obj.train_history = ckpt.get("train_history", [])
        obj.peak_vram_mb = float(ckpt.get("peak_vram_mb", 0.0))
        obj.total_train_time_s = float(ckpt.get("total_train_time_s", 0.0))
        obj.net = obj._build_network(obj.num_pois, obj.num_cats)
        obj.net.load_state_dict({k: v.to(obj.device) for k, v in ckpt["state_dict"].items()})
        obj.net.eval()
        return obj
