# 15-Minute Presentation & Live Demo Script (HUST IT4613)

Target duration: **15 minutes** (2 to 4 team members).

---

## 00:00 - 02:30 | Problem Motivation & NYC Dataset (Speaker 1)
- **Hook:** Tourists in New York City face 49,000+ venues across Manhattan, Brooklyn, and Queens. Generic "Top 10 NYC attractions" lists ignore where the tourist just visited, how far they are willing to travel, and what category they want next.
- **Problem Formulation:** Given a sequence of visited POIs (coordinates, categories, timestamps) plus real-time constraints (`max_distance_km`, `categories`, `hour_of_day`), rank the Top-K next POIs.
- **Dataset Integrity:** `CRUISEResearchGroup/Massive-STEPS-New-York` (`tabular` config):
  - `train`: 190,740 check-ins (64,428 trails, 39,113 POIs, 6,917 users, 569 categories).
  - `validation`: 27,201 check-ins (9,204 trails).
  - `test`: 54,427 check-ins (18,409 trails).
  - Highlight train-only ID mapping (`0=PAD`, `1=OOV`), deterministic imputation of ~7.2% missing venue coords/names using city coords/categories, and zero data leakage.

---

## 02:30 - 06:30 | Modeling Progression (Speaker 2)
Walk through the progression of all 6 paradigms taught in IT4613:
1. **Local & Global Popularity:** Fast geospatial baseline using a Haversine `BallTree`.
2. **Collaborative Filtering (BPR-MF):** Matrix factorization on implicit check-ins with item-averaging fold-in for cold-start trails.
3. **Content-Based (TF-IDF + Spatial):** Bigram TF-IDF over POI categories and names combined with recency decay and exponential distance decay.
4. **Spatial-Temporal SASRec (CUDA):** Causal self-attention fusing POI IDs, categories, cyclical hour/day-of-week, and `log1p(delta_dist)` / `log1p(delta_time)`, trained in mixed precision (`AMP`) with 64-negative sampled softmax (peak VRAM: **130.59 MB**).
5. **CL4SRec Contrastive Learning (CUDA):** Self-supervised trajectory augmentations (crop, mask, local reorder) with NT-Xent loss (peak VRAM: **219.53 MB**).
6. **Context-Aware Hybrid Reranker (CPU):** Two-stage `BallTree` retrieval + validation-tuned fusion (`0.65 * Seq + 0.05 * Content + 0.15 * CF + 0.15 * Spatial`) + hard radius/category filtering.

---

## 06:30 - 09:30 | Offline Benchmark Results (Speaker 2 / 3)
Show `artifacts/benchmark_comparison.png` and the test table (18,409 trails, full-ranking over 39,113 POIs):
- `popularity`: `Recall@10 = 0.0648`, `NDCG@10 = 0.0355`, `Coverage@10 = 0.0184` (severe popularity bias).
- `cf_bpr`: `Recall@10 = 0.0859`, `NDCG@10 = 0.0457`, `Coverage@10 = 0.2135`.
- `content_based`: `Recall@10 = 0.0294`, `Coverage@10 = 0.8190` (high diversity, low precision alone).
- `sequential` (SASRec): `Recall@10 = 0.1626`, `NDCG@10 = 0.0901`, `MRR@10 = 0.0674` (nearly 2x CF-BPR).
- `sequential_cl` (CL4SRec): `Recall@10 = 0.1600`, `NDCG@10 = 0.0890`.
- **`hybrid`:** **`Recall@10 = 0.1989`**, **`NDCG@10 = 0.1006`**, **`MRR@10 = 0.0697`**, **`Coverage@10 = 0.4297`**, **`in_vocab Recall@10 = 0.2183`**.

---

## 09:30 - 13:30 | Live Web Prototype Demo (Speaker 3)
Open FastAPI (`:8000`) + React-Leaflet UI (`:5173`):
1. **Demo 1 - Returning Tourist Mode:**
   - Select a Manhattan trajectory from the 20 test trails; toggle **Show Ground-Truth Next POI** (green marker).
   - Switch model from `Local & Global Popularity` to `Context-Aware Hybrid` and highlight how recommendations shift to match the user's trajectory context and hit near the ground-truth stop in `< 50 ms` on CPU.
   - Click a recommendation card to smoothly `flyTo` the venue on the Leaflet map and inspect the score breakdown bars (`Seq`, `Cnt`, `CF`, `Geo`) + human-readable explanation.
2. **Demo 2 - Cold-Start Tourist Mode:**
   - Switch to `Cold-Start Tourist`.
   - Search and add stops (e.g., `Central Park`, a museum, or click an anchor directly on Midtown Manhattan).
   - Tighten `Max Radius` slider to `1.5 km`, select category chips (e.g., `Coffee Shop`, `Italian Restaurant`), adjust `Time of Day` to `18:00`, and show real-time constrained recommendations + `"Cold-Start"` badge.

---

## 13:30 - 15:00 | Limitations, Engineering Takeaways & Q&A (Speaker 1 / All)
- **Key Takeaways:** Sequence + spatial-temporal context beats static CF by >2x; hybrid reranking boosts both accuracy (`0.1989 R@10`) and catalog coverage (`43.0%`).
- **Limitations & Future Work:** ~8.9% of test targets are unseen POIs (cold items); future integration of opening hours, weather, and multi-stop route optimization.
- Invite questions from the grading panel.
