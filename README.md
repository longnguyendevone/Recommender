# NYC Context-Aware Next-POI Tourist Recommender (HUST IT4613)

End-to-end prototypical recommender system for Next-POI tourist recommendation in New York City using `CRUISEResearchGroup/Massive-STEPS-New-York` (`tabular` config). Implements and benchmarks six recommender paradigms taught in HUST IT4613 alongside a CPU-served FastAPI backend and an interactive React + Leaflet map UI.

---

## 1. Repository Structure

```text
Recommender/
+-- configs/
|   +-- data.yaml                # Dataset and EDA/preprocessing paths
|   +-- baselines.yaml           # Popularity, BPR-MF, Content-Based configs
|   +-- sequential.yaml          # SASRec & CL4SRec CUDA training configs
|   \-- hybrid.yaml              # Candidate retrieval & fusion weights
+-- data/                        # Raw and processed Parquet/BallTree artifacts
+-- artifacts/                   # Checkpoints, EDA plots, benchmark tables
+-- src/
|   +-- utils/                   # seed.py, geo.py (Haversine + BallTree), io.py
|   +-- data/                    # download_eda.py, preprocess.py, dataset.py
|   +-- models/                  # base.py, popularity.py, cf_bpr.py, content_based.py,
|   |                            # sequential.py, sequential_cl.py, hybrid.py
|   +-- evaluation/              # metrics.py, benchmark.py
|   \-- train.py                 # Unified training CLI
+-- app/
|   +-- backend/                 # FastAPI CPU inference API (main.py, schemas.py, recommender_service.py)
|   \-- frontend/                # Vite + React + Leaflet interactive NYC map UI
+-- docs/                        # architecture.mmd, report_template.tex, DEMO_SCRIPT.md
+-- scripts/                     # package_submission.py
+-- tests/                       # test_api.py
+-- requirements.txt
\-- README.md
```

---

## 2. Environment Setup

### Python Environment (Recommended: Python 3.12 via `uv`)
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

### Alternative: Miniconda (Python 3.12)
```powershell
conda create -n nyc-poi python=3.12 -y
conda activate nyc-poi
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Node.js Frontend Setup (Node 18+ / 22+)
```powershell
cd app/frontend
npm install
```

---

## 3. Data Pipeline & Model Training

1. **Download Dataset & Run EDA**
   ```powershell
   .venv\Scripts\python.exe -m src.data.download_eda --config configs/data.yaml
   ```
2. **Preprocess Splits, Mappings, Metadata & BallTree**
   ```powershell
   .venv\Scripts\python.exe -m src.data.preprocess --config configs/data.yaml
   ```
3. **Train All 6 Models (CPU Baselines + CUDA Sequential/CL4SRec + CPU Hybrid)**
   ```powershell
   .venv\Scripts\python.exe -m src.train --model all
   ```
   Or train individually:
   ```powershell
   .venv\Scripts\python.exe -m src.train --model popularity --config configs/baselines.yaml
   .venv\Scripts\python.exe -m src.train --model cf_bpr --config configs/baselines.yaml
   .venv\Scripts\python.exe -m src.train --model content_based --config configs/baselines.yaml
   .venv\Scripts\python.exe -m src.train --model sequential --config configs/sequential.yaml
   .venv\Scripts\python.exe -m src.train --model sequential_cl --config configs/sequential.yaml
   .venv\Scripts\python.exe -m src.train --model hybrid --config configs/hybrid.yaml
   ```
4. **Run Offline Test Benchmark (18,409 Trajectories, Full-Ranking over 39,113 POIs)**
   ```powershell
   .venv\Scripts\python.exe -m src.evaluation.benchmark --split test
   ```

---

## 4. Offline Test Benchmark Results

| Model | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 | Coverage@10 | in_vocab Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `popularity` | 0.0138 | 0.0425 | 0.0648 | 0.0283 | 0.0355 | 0.0266 | 0.0184 | 0.0711 |
| `cf_bpr` | 0.0169 | 0.0545 | 0.0859 | 0.0356 | 0.0457 | 0.0335 | 0.2135 | 0.0943 |
| `content_based` | 0.0022 | 0.0183 | 0.0294 | 0.0106 | 0.0142 | 0.0095 | 0.8190 | 0.0323 |
| `sequential` | 0.0259 | 0.1225 | 0.1626 | 0.0771 | 0.0901 | 0.0674 | 0.3616 | 0.1785 |
| `sequential_cl` | 0.0253 | 0.1225 | 0.1600 | 0.0768 | 0.0890 | 0.0666 | 0.3537 | 0.1756 |
| `hybrid` | 0.0121 | 0.1467 | 0.1989 | 0.0836 | 0.1006 | 0.0697 | 0.4297 | 0.2183 |

---

## 5. Launching the Web Prototype

1. **Start FastAPI CPU Backend (Port 8000):**
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Start Vite + React + Leaflet Frontend (Port 5173):**
   ```powershell
   cd app/frontend
   npm run dev
   ```
   Open `http://127.0.0.1:5173` in your browser.

---

## 6. Verification & Packaging

- **Run API Unit & Latency Tests:**
  ```powershell
  .venv\Scripts\python.exe -m pytest tests/test_api.py -v
  ```
- **Build Production Frontend Bundle:**
  ```powershell
  cd app/frontend && npm run build
  ```
- **Create Clean Submission Zip:**
  ```powershell
  .venv\Scripts\python.exe scripts/package_submission.py
  ```
