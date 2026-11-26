"""Spatial-Temporal SASRec sequential Next-POI recommender (PyTorch GPU/CPU)."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import NextPOIDataset, build_eval_queries
from src.evaluation.metrics import compute_ranking_metrics
from src.models.base import BaseRecommender
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


class CausalSelfAttentionBlock(nn.Module):
    """Pre-norm causal multi-head self-attention block with pad-safe softmax."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        causal_ok = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)
        key_ok = valid_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        diag_eye = torch.eye(seq_len, device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        # Keep diagonal unmasked when query is padding so softmax never receives all -inf.
        attn_ok = (causal_ok & key_ok) | diag_eye

        scores = scores.masked_fill(~attn_ok, -1e4)
        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_weights = self.attn_drop(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = x + self.resid_drop(self.out_proj(context))
        x = x + self.ffn(self.norm2(x))
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class SpatialTemporalSASRec(nn.Module):
    """Causal Transformer encoder over POI, category, cyclical time, and spatial deltas."""

    def __init__(
        self,
        num_pois: int,
        num_cats: int,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_pois = int(num_pois)
        self.num_cats = int(num_cats)
        self.max_seq_len = int(max_seq_len)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)

        self.poi_emb = nn.Embedding(self.num_pois, self.embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(self.num_cats, self.embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.embed_dim)
        self.time_proj = nn.Linear(4, self.embed_dim)
        self.delta_proj = nn.Linear(2, self.embed_dim)

        self.emb_drop = nn.Dropout(self.dropout_p)
        self.emb_norm = nn.LayerNorm(self.embed_dim)
        self.blocks = nn.ModuleList(
            [
                CausalSelfAttentionBlock(self.embed_dim, self.num_heads, self.dropout_p)
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.padding_idx is not None:
                    with torch.no_grad():
                        mod.weight[mod.padding_idx].zero_()
            elif isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def encode_sequence(
        self,
        poi_seq: torch.Tensor,
        cat_seq: torch.Tensor,
        time_feats: torch.Tensor,
        delta_feats: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = poi_seq.shape
        valid_mask = poi_seq != 0
        positions = torch.arange(seq_len, device=poi_seq.device).unsqueeze(0).expand(bsz, -1)

        x = (
            self.poi_emb(poi_seq)
            + 0.5 * self.cat_emb(cat_seq)
            + 0.25 * self.time_proj(time_feats)
            + 0.25 * self.delta_proj(delta_feats)
            + self.pos_emb(positions)
        )
        x = self.emb_norm(self.emb_drop(x)) * valid_mask.unsqueeze(-1).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, valid_mask)
        return self.final_norm(x) * valid_mask.unsqueeze(-1).to(x.dtype)

    def encode_last(
        self,
        poi_seq: torch.Tensor,
        cat_seq: torch.Tensor,
        time_feats: torch.Tensor,
        delta_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_sequence(poi_seq, cat_seq, time_feats, delta_feats)[:, -1, :]


class SequentialRecommender(BaseRecommender):
    """Wrapper around SpatialTemporalSASRec with AMP training and full-ranking inference."""

    name = "sequential"

    def __init__(
        self,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_negatives: int = 64,
        device: str = "cuda",
        use_amp: bool = True,
        batch_size: int = 128,
        grad_accum_steps: int = 2,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 4,
        seed: int = 42,
    ) -> None:
        self.max_seq_len = int(max_seq_len)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_negatives = int(num_negatives)
        self.use_amp = bool(use_amp)
        self.batch_size = int(batch_size)
        self.grad_accum_steps = int(grad_accum_steps)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.seed = int(seed)

        if device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.net: SpatialTemporalSASRec | None = None
        self.poi_indices: np.ndarray | None = None
        self.num_pois: int = 0
        self.num_cats: int = 0
        self.train_history: list[dict[str, float]] = []
        self.peak_vram_mb: float = 0.0
        self.total_train_time_s: float = 0.0

    def _build_network(self, num_pois: int, num_cats: int) -> SpatialTemporalSASRec:
        return SpatialTemporalSASRec(
            num_pois=num_pois,
            num_cats=num_cats,
            max_seq_len=self.max_seq_len,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _compute_rec_loss(
        self,
        h_seq: torch.Tensor,
        targets: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        assert self.net is not None
        valid_mask = targets >= 2
        if not valid_mask.any():
            return h_seq.sum() * 0.0

        h_flat = h_seq[valid_mask].float()  # (M, D)
        pos_ids = targets[valid_mask]  # (M,)
        neg_ids = negatives[valid_mask]  # (M, K_neg)

        pos_emb = self.net.poi_emb(pos_ids).float()
        neg_emb = self.net.poi_emb(neg_ids).float()

        pos_logits = (h_flat * pos_emb).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_emb, h_flat.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def _compute_step_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.net is not None
        poi_seq = batch["poi_seq"].to(self.device, non_blocking=True)
        cat_seq = batch["cat_seq"].to(self.device, non_blocking=True)
        time_feats = batch["time_feats"].to(self.device, non_blocking=True)
        delta_feats = batch["delta_feats"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        negatives = batch["negatives"].to(self.device, non_blocking=True)

        h_seq = self.net.encode_sequence(poi_seq, cat_seq, time_feats, delta_feats)
        loss = self._compute_rec_loss(h_seq, targets, negatives)
        return loss, {"rec_loss": float(loss.detach().item())}

    def fit(self, train_data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        set_seed(self.seed)
        self.num_pois = int(train_data["num_pois"])
        self.num_cats = int(train_data["num_cats"])
        self.poi_indices = train_data["poi_metadata"]["poi_idx"].to_numpy(dtype=np.int64)

        train_seq_path = train_data["train_sequences_path"]
        val_seq_df = train_data["val_sequences"]
        val_queries = build_eval_queries(val_seq_df)
        val_targets = np.array([q["target_poi_idx"] for q in val_queries], dtype=np.int64)

        train_ds = NextPOIDataset(
            sequences_path=train_seq_path,
            num_pois=self.num_pois,
            max_seq_len=self.max_seq_len,
            num_negatives=self.num_negatives,
            mode="train",
            seed=self.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        self.net = self._build_network(self.num_pois, self.num_cats)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.epochs, 1), eta_min=self.lr * 0.1
        )
        amp_enabled = self.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_ndcg = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0
        self.train_history = []
        self.peak_vram_mb = 0.0
        t_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            t_ep = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            self.net.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            steps = 0

            for step_idx, batch in enumerate(train_loader):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, _ = self._compute_step_loss(batch)
                    loss_scaled = loss / self.grad_accum_steps

                scaler.scale(loss_scaled).backward()
                if (step_idx + 1) % self.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.detach().item())
                steps += 1

            scheduler.step()
            epoch_vram_mb = (
                float(torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2))
                if self.device.type == "cuda"
                else 0.0
            )
            self.peak_vram_mb = max(self.peak_vram_mb, epoch_vram_mb)

            preds, _ = self.predict_topk(val_queries, k=10, batch_size=256)
            val_metrics = compute_ranking_metrics(preds, val_targets, len(self.poi_indices))
            epoch_s = time.perf_counter() - t_ep
            avg_loss = running_loss / max(steps, 1)

            record = {
                "epoch": float(epoch),
                "train_loss": round(avg_loss, 5),
                "val_Recall@10": round(val_metrics["Recall@10"], 5),
                "val_NDCG@10": round(val_metrics["NDCG@10"], 5),
                "val_MRR@10": round(val_metrics["MRR@10"], 5),
                "peak_vram_mb": round(epoch_vram_mb, 2),
                "epoch_time_s": round(epoch_s, 2),
            }
            self.train_history.append(record)
            print(
                f"[{self.name}] Epoch {epoch:02d}/{self.epochs:02d} | "
                f"loss={avg_loss:.4f} | val_R@10={val_metrics['Recall@10']:.4f} | "
                f"val_NDCG@10={val_metrics['NDCG@10']:.4f} | "
                f"vram={epoch_vram_mb:.1f}MB | time={epoch_s:.1f}s"
            )

            if val_metrics["NDCG@10"] > best_ndcg:
                best_ndcg = val_metrics["NDCG@10"]
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"[{self.name}] Early stopping at epoch {epoch} (best NDCG@10={best_ndcg:.4f})")
                    break

        if best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.total_train_time_s = time.perf_counter() - t_start
        return {
            "model": self.name,
            "best_val_NDCG@10": round(best_ndcg, 5),
            "peak_vram_mb": round(self.peak_vram_mb, 2),
            "total_train_time_s": round(self.total_train_time_s, 2),
            "epochs_run": len(self.train_history),
        }

    def _queries_to_tensors(
        self, queries: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(queries)
        L = self.max_seq_len
        poi_arr = np.zeros((n, L), dtype=np.int64)
        cat_arr = np.zeros((n, L), dtype=np.int64)
        time_arr = np.zeros((n, L, 4), dtype=np.float32)
        delta_arr = np.zeros((n, L, 2), dtype=np.float32)

        for i, q in enumerate(queries):
            p_seq = q.get("poi_seq", [])
            if not p_seq:
                continue
            c_seq = q.get("cat_seq", [1] * len(p_seq))
            used = min(len(p_seq), L)
            offset = L - used

            poi_arr[i, offset:] = [
                int(p) if 0 <= int(p) < self.num_pois else 1 for p in p_seq[-used:]
            ]
            cat_arr[i, offset:] = [
                int(c) if 0 <= int(c) < self.num_cats else 1 for c in c_seq[-used:]
            ]

            if "hour_sin_seq" in q and len(q["hour_sin_seq"]) >= used:
                time_arr[i, offset:, 0] = q["hour_sin_seq"][-used:]
                time_arr[i, offset:, 1] = q["hour_cos_seq"][-used:]
                time_arr[i, offset:, 2] = q["dow_sin_seq"][-used:]
                time_arr[i, offset:, 3] = q["dow_cos_seq"][-used:]
            else:
                time_arr[i, -1, 0] = float(q.get("hour_sin", 0.0))
                time_arr[i, -1, 1] = float(q.get("hour_cos", 1.0))
                time_arr[i, -1, 2] = float(q.get("dow_sin", 0.0))
                time_arr[i, -1, 3] = float(q.get("dow_cos", 1.0))

            if "delta_dist_seq" in q and len(q["delta_dist_seq"]) >= used:
                delta_arr[i, offset:, 0] = np.log1p(
                    np.maximum(0.0, np.asarray(q["delta_dist_seq"][-used:], dtype=np.float32))
                )
                delta_arr[i, offset:, 1] = np.log1p(
                    np.maximum(0.0, np.asarray(q["delta_time_seq"][-used:], dtype=np.float32))
                )

        return (
            torch.from_numpy(poi_arr).to(self.device),
            torch.from_numpy(cat_arr).to(self.device),
            torch.from_numpy(time_arr).to(self.device),
            torch.from_numpy(delta_arr).to(self.device),
        )

    @torch.no_grad()
    def encode_queries(self, queries: list[dict[str, Any]], batch_size: int = 256) -> np.ndarray:
        """Encode queries into (N, embed_dim) float32 embeddings."""
        assert self.net is not None
        self.net.eval()
        out = np.empty((len(queries), self.embed_dim), dtype=np.float32)
        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            p_t, c_t, t_t, d_t = self._queries_to_tensors(queries[start:end])
            h = self.net.encode_last(p_t, c_t, t_t, d_t)
            out[start:end] = h.detach().cpu().numpy().astype(np.float32)
        return out

    @torch.no_grad()
    def score_candidates(
        self, query: dict[str, Any], candidate_poi_indices: np.ndarray
    ) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        cand = np.asarray(candidate_poi_indices, dtype=np.int64)
        p_t, c_t, t_t, d_t = self._queries_to_tensors([query])
        h = self.net.encode_last(p_t, c_t, t_t, d_t).squeeze(0)
        cand_t = torch.from_numpy(cand).to(self.device)
        emb = self.net.poi_emb(cand_t)
        scores = torch.matmul(emb, h)
        return scores.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_topk(
        self, queries: list[dict[str, Any]], k: int = 10, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.net is not None and self.poi_indices is not None
        self.net.eval()
        n = len(queries)
        out_idx = np.zeros((n, k), dtype=np.int64)
        out_scores = np.zeros((n, k), dtype=np.float32)
        if n == 0:
            return out_idx, out_scores

        valid_ids = torch.from_numpy(np.array(self.poi_indices, copy=True)).to(self.device)
        valid_item_emb = self.net.poi_emb.weight[valid_ids]  # (39113, D)
        k_eff = min(k, len(self.poi_indices))

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            p_t, c_t, t_t, d_t = self._queries_to_tensors(queries[start:end])
            h = self.net.encode_last(p_t, c_t, t_t, d_t)  # (B, D)
            logits = torch.matmul(h, valid_item_emb.T)  # (B, 39113)
            top_scores, top_pos = torch.topk(logits, k=k_eff, dim=-1)
            pos_np = top_pos.detach().cpu().numpy()
            out_idx[start:end, :k_eff] = self.poi_indices[pos_np]
            out_scores[start:end, :k_eff] = top_scores.detach().cpu().numpy().astype(np.float32)

        return out_idx, out_scores

    def _checkpoint_payload(self) -> dict[str, Any]:
        assert self.net is not None
        return {
            "model_class": self.name,
            "config": {
                "max_seq_len": self.max_seq_len,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "num_negatives": self.num_negatives,
                "use_amp": self.use_amp,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "epochs": self.epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
                "seed": self.seed,
            },
            "num_pois": self.num_pois,
            "num_cats": self.num_cats,
            "poi_indices": self.poi_indices,
            "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "train_history": self.train_history,
            "peak_vram_mb": self.peak_vram_mb,
            "total_train_time_s": self.total_train_time_s,
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        ensure_dir(out_path.parent)
        torch.save(self._checkpoint_payload(), out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SequentialRecommender":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        obj = cls(**cfg, device=device)
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
