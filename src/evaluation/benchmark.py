"""Offline benchmark evaluating all 6 recommender models on validation or test split."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import build_eval_queries
from src.evaluation.metrics import compute_ranking_metrics
from src.models.cf_bpr import CFBPRRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.models.sequential import SequentialRecommender
from src.models.sequential_cl import CL4SRecRecommender
from src.utils.io import ensure_dir, load_parquet


def _format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):.4f}" if h != "eval_time_s" else f"{float(v):.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    metrics_to_plot = ["Recall@1", "Recall@5", "Recall@10", "NDCG@10", "MRR@10"]
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#64748b", "#0284c7", "#2563eb", "#0d9488", "#ea580c"]
    for j, m_col in enumerate(metrics_to_plot):
        offset = (j - (len(metrics_to_plot) - 1) / 2.0) * width
        ax.bar(x + offset, df[m_col].to_numpy(dtype=float), width=width, label=m_col, color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("NYC Next-POI Offline Benchmark Comparison")
    ax.legend(frameon=False, ncol=5, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_benchmark(
    split: str = "test",
    processed_dir: str | Path = "data/processed",
    models_dir: str | Path = "artifacts/models",
    out_dir: str | Path = "artifacts",
) -> pd.DataFrame:
    proc_path = Path(processed_dir)
    ckpt_dir = Path(models_dir)
    art_dir = ensure_dir(out_dir)

    poi_meta = load_parquet(proc_path / "poi_metadata.parquet")
    seq_df = load_parquet(proc_path / f"{split}_sequences.parquet")
    with open(proc_path / "poi_balltree.pkl", "rb") as f:
        ball_tree = pickle.load(f)["tree"]

    queries = build_eval_queries(seq_df)
    targets = np.array([q["target_poi_idx"] for q in queries], dtype=np.int64)
    num_catalog = len(poi_meta)

    loaders: list[tuple[str, Any]] = [
        ("popularity", lambda: PopularityRecommender.load(ckpt_dir / "popularity.pkl")),
        ("cf_bpr", lambda: CFBPRRecommender.load(ckpt_dir / "cf_bpr.pkl")),
        ("content_based", lambda: ContentBasedRecommender.load(ckpt_dir / "content_based.pkl")),
        ("sequential", lambda: SequentialRecommender.load(ckpt_dir / "sequential.pt", device="cpu")),
        ("sequential_cl", lambda: CL4SRecRecommender.load(ckpt_dir / "sequential_cl.pt", device="cpu")),
        ("hybrid", lambda: HybridRecommender.load(ckpt_dir / "hybrid.pkl", ball_tree=ball_tree)),
    ]

    rows: list[dict[str, Any]] = []
    for name, loader_fn in loaders:
        model = loader_fn()
        t0 = time.perf_counter()
        preds, _ = model.predict_topk(queries, k=10)
        eval_s = time.perf_counter() - t0
        m = compute_ranking_metrics(preds, targets, num_catalog)
        row = {
            "model": name,
            "Recall@1": m["Recall@1"],
            "Recall@5": m["Recall@5"],
            "Recall@10": m["Recall@10"],
            "NDCG@5": m["NDCG@5"],
            "NDCG@10": m["NDCG@10"],
            "MRR@10": m["MRR@10"],
            "Coverage@10": m["Coverage@10"],
            "in_vocab_Recall@10": m["in_vocab_Recall@10"],
            "eval_time_s": round(eval_s, 2),
        }
        rows.append(row)
        print(
            f"[{split}] {name:14s} | R@1={m['Recall@1']:.4f} | R@5={m['Recall@5']:.4f} | "
            f"R@10={m['Recall@10']:.4f} | NDCG@10={m['NDCG@10']:.4f} | "
            f"MRR@10={m['MRR@10']:.4f} | Cov@10={m['Coverage@10']:.4f} | time={eval_s:.2f}s"
        )

    df = pd.DataFrame(rows)
    csv_path = art_dir / "benchmark_results.csv"
    md_path = art_dir / "benchmark_results.md"
    plot_path = art_dir / "benchmark_comparison.png"

    df.to_csv(csv_path, index=False, float_format="%.6f")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# NYC Next-POI Benchmark ({split} split, {len(queries)} trails)\n\n")
        f.write(_format_markdown_table(df))
    _plot_comparison(df, plot_path)

    print(f"Saved benchmark CSV -> {csv_path}")
    print(f"Saved benchmark MD  -> {md_path}")
    print(f"Saved benchmark PNG -> {plot_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all 6 recommender models.")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    args = parser.parse_args()
    run_benchmark(split=args.split)


if __name__ == "__main__":
    main()
