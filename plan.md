# Project Specification: Context-Aware Next-POI Tourist Recommender (New York City)

## 1. Objective & Academic Context
Build a complete, modular, end-to-end prototypical recommender system for the HUST IT4613 (Recommender Systems) course project.
- Problem: Next-POI Tourist Recommendation in New York City. Given a tourist's visit trajectory (sequence of visited POIs with timestamps and coordinates) plus optional real-time constraints (max travel radius, preferred categories, time of day), predict and rank the Top-K next places they should visit.
- Deliverables Required:
  1. Clean, reproducible source code with a detailed `README.md` (setup, training, evaluation, and app launch instructions).
  2. Offline benchmark results (CSV + Markdown tables + plots) comparing all recommender paradigms taught in IT4613.
  3. An interactive web prototype (FastAPI + ReactJS + Leaflet map) ready for a 15-minute live presentation.

---

## 2. Dataset Specification
- Source: Hugging Face `CRUISEResearchGroup/Massive-STEPS-New-York` (272,368 check-ins, 92,041 trajectories, 6,929 users, 49,218 POIs).
- Config: Load the `tabular` configuration (do NOT use the `default` LLM-prompt text config unless needed for verification).
- Splits: Strictly respect the official `train` (64,428 trails), `validation` (9,204 trails), and `test` (18,409 trails) splits to prevent data leakage.
- Preprocessing Rules:
  - Inspect exact column names and missing values first in Phase 1.
  - Build deterministic ID mappings (`user2idx`, `poi2idx`, `cat2idx`) fit ONLY on the training set (reserve index `0` for padding and `1` for unseen/OOV items).
  - Precompute a POI metadata lookup table (`poi_id`, `name`, `category`, `lat`, `lon`, `address`, `train_popularity`).
  - Extract cyclical temporal features (`sin/cos` of hour-of-day and day-of-week), step-to-step Haversine distance (`delta_dist_km`), and time gap (`delta_time_min`).

---

## 3. Hardware, Device Scoping & Coding Standards
- Target Machine: Windows 11, NVIDIA RTX 3060 Laptop GPU (6 GB VRAM), 16 GB DDR5 RAM, Python 3.12 `.venv` via `uv` (or Miniconda).
- Strict Device Scoping:
  - CPU Only: Data downloading, EDA, preprocessing, Haversine `BallTree` indexing, `popularity.py`, `content_based.py`, `cf_bpr.py` (use CPU-optimized `implicit` or `scipy.sparse`), `hybrid.py`, offline evaluation, and FastAPI live inference.
  - GPU (CUDA) Only: Deep learning sequential models (`sequential.py` and `sequential_cl.py`).
- 6 GB VRAM Safeguards (for PyTorch models):
  - Use `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`.
  - Use sampled softmax or negative sampling during training (avoid full 49,218-class dense cross-entropy if it causes OOM, or use chunked/micro-batched evaluation).
  - Support configurable `batch_size` and `grad_accum_steps` in YAML configs.
  - Log peak VRAM usage (`torch.cuda.max_memory_allocated`) and training wall-clock time per epoch.
- Code Style:
  - Pure ASCII only across all `.py`, `.yaml`, `.js`, and `.md` files (no emojis or decorative Unicode).
  - Strictly modular, single-purpose files. No monolithic scripts.
  - Deterministic seeding (`seed=42`) across Python, NumPy, and PyTorch.

---

## 4. Repository Architecture
Scaffold and maintain this exact directory structure directly at the repository root (`d:\Repo\Recommender`):

```text
Recommender/
+-- configs/
|   +-- data.yaml
|   +-- baselines.yaml
|   +-- sequential.yaml
|   \-- hybrid.yaml
+-- data/                        # Git-ignored raw & processed parquet/pkl artifacts
+-- artifacts/                   # Saved model checkpoints, metrics CSVs, and plots
+-- src/
|   +-- utils/
|   |   +-- seed.py              # Reproducibility helpers
|   |   +-- geo.py               # Vectorized Haversine distance & BallTree radius queries
|   |   \-- io.py                # YAML, JSON, Parquet loaders
|   +-- data/
|   |   +-- download_eda.py      # Download HF dataset, print schema & EDA stats, save plots
|   |   +-- preprocess.py        # Clean, map IDs, build POI metadata table & sequences
|   |   \-- dataset.py           # PyTorch Dataset/DataLoader with padding & negative sampling
|   +-- models/
|   |   +-- base.py              # Abstract BaseRecommender interface: fit(), predict_topk()
|   |   +-- popularity.py        # Global Popularity + Distance-Aware Local Popularity (CPU)
|   |   +-- cf_bpr.py            # Collaborative Filtering: BPR-MF via implicit (CPU)
|   |   +-- content_based.py     # Category TF-IDF/embedding + Spatial Proximity (CPU)
|   |   +-- sequential.py        # Spatial-Temporal SASRec or GRU4Rec (PyTorch GPU)
|   |   +-- sequential_cl.py     # Contrastive Learning Sequential Rec: CL4SRec augmentations (PyTorch GPU)
|   |   \-- hybrid.py            # Context-Aware Hybrid Reranker + Knowledge-Based Constraints (CPU)
|   +-- evaluation/
|   |   +-- metrics.py           # Recall@K, NDCG@K, MRR, HitRate@K (K=1,5,10), Catalog Coverage
|   |   \-- benchmark.py         # Run all models on test set, export Markdown/CSV comparison table
|   \-- train.py                 # Unified CLI: python -m src.train --model <name> --config <path>
+-- app/
|   +-- backend/
|   |   +-- main.py              # FastAPI app (CPU inference only)
|   |   +-- schemas.py           # Pydantic request/response models
|   |   \-- recommender_service.py # Loads artifacts, serves Top-K + human-readable explanations
|   \-- frontend/                # ReactJS + Leaflet interactive NYC map UI
+-- scripts/
+-- tests/
+-- docs/
+-- requirements.txt
\-- README.md
```

---

# Execution Specification: Phases 1 to 4 (Data, Baselines, Deep Sequential & Benchmark)

## Phase 1: Scaffolding, Environment Setup, Data Download & EDA
- Create `.gitignore`, `requirements.txt`, `configs/*.yaml`, `src/utils/{seed,geo,io}.py`, and `src/data/download_eda.py`.
- Provision a Python 3.12 virtual environment (`.venv`) via `uv` with CUDA-enabled PyTorch and project dependencies.
- Download `CRUISEResearchGroup/Massive-STEPS-New-York` (`tabular` config), save `train`, `validation`, and `test` splits to `data/raw/`, inspect exact column names/dtypes/missing counts, compute OOV overlap, and export EDA plots + `artifacts/eda/eda_summary.json`.

## Phase 2: Preprocessing & PyTorch Dataset Pipeline
- Implement `src/data/preprocess.py` using the exact schema verified in Phase 1.
- Build deterministic ID mappings (`user2idx`, `poi2idx`, `cat2idx`) fit ONLY on `train` (reserving `0` for PAD and `1` for OOV/unseen).
- Precompute `poi_metadata.parquet`, cyclical temporal features (`sin/cos` of hour-of-day and day-of-week), step-to-step Haversine distance (`delta_dist_km`), time gap (`delta_time_min`), and a Haversine `BallTree` index.
- Implement `src/data/dataset.py` with sequence truncation/padding and negative sampling for PyTorch GPU training.

## Phase 3: CPU Baselines & Offline Evaluation Framework
- Implement `src/models/base.py` (`BaseRecommender`: `fit()`, `predict_topk()`).
- Implement CPU models: `popularity.py`, `cf_bpr.py`, and `content_based.py`.
- Implement `src/evaluation/metrics.py` (`Recall@K`, `NDCG@K`, `MRR`, `HitRate@K` for `K=1,5,10`, and `Catalog Coverage`) using chunked full-ranking evaluation and explicit OOV target reporting.

## Phase 4: GPU Sequential Models, Contrastive Learning, Hybrid Reranker & Benchmark
- Implement `src/models/sequential.py` (Spatial-Temporal SASRec/GRU4Rec on CUDA with AMP + gradient accumulation) and `src/models/sequential_cl.py` (CL4SRec with trajectory augmentations).
- Implement `src/models/hybrid.py` (validation-tuned fusion weights + two-stage `BallTree` retrieve-and-rerank with knowledge-based constraints).
- Implement `src/train.py` and `src/evaluation/benchmark.py` to train all 6 models, log peak VRAM/epoch times, and export `artifacts/benchmark_results.csv` and Markdown tables/plots.

---

# Execution Specification: Phase 5 and Forward (Prototype, Packaging & Deliverables)

## Phase 5: FastAPI Inference Backend (`app/backend/`)
Build a lightweight, CPU-only REST API that loads trained artifacts from `artifacts/` and `data/` to serve low-latency Next-POI recommendations.

### 5.1 Backend Module Breakdown
- `app/backend/schemas.py`:
  - Define Pydantic v2 models:
    - `RecommendRequest`: `user_id` (optional `str | int` matching Phase 1 schema), `trajectory` (list of `str | int` `poi_id`s matching Phase 1 schema), `current_lat` (optional float), `current_lon` (optional float), `hour_of_day` (optional int, `0..23`), `max_distance_km` (float, default `5.0`), `categories` (list of str, default `[]`), `model` (str, default `"hybrid"`), `top_k` (int, default `10`, range `1..30`).
    - `POIRecommendation`: `rank` (int), `poi_id` (int), `name` (str), `category` (str), `lat` (float), `lon` (float), `distance_km` (float), `total_score` (float), `score_breakdown` (dict with `sequential`, `content`, `cf`, `spatial`), `explanation` (str).
    - `RecommendResponse`: `model_used` (str), `inference_time_ms` (float), `is_cold_start` (bool), `recommendations` (list of `POIRecommendation`).
- `app/backend/recommender_service.py`:
  - Load `poi_metadata.parquet`, ID mappings (`user2idx`, `poi2idx`, `cat2idx`), and the prebuilt Haversine `BallTree` index once on startup.
  - Load all trained model checkpoints (`popularity`, `cf_bpr`, `content_based`, `sequential`, `sequential_cl`, `hybrid`) strictly onto **CPU** (`map_location="cpu"`, `model.eval()` with `torch.no_grad()`).
  - Resolve current anchor coordinates from the last POI in `trajectory` (or explicit `current_lat`/`current_lon` if provided).
  - Generate human-readable explanation strings per item (e.g., `"Matches your Art Museum & Cafe trail | 0.9 km away | Top sequential pick"`).
- `app/backend/main.py`:
  - Initialize FastAPI app with CORS middleware enabled for local frontend development (`http://localhost:5173`).
  - Endpoints:
    - `GET /health`: Returns loaded models, total NYC POIs indexed, and device (`cpu`).
    - `GET /pois/search?q=&category=&limit=20`: Case-insensitive prefix/substring search over NYC POI names and categories.
    - `GET /pois/categories`: Returns sorted list of unique POI categories with counts.
    - `GET /users/sample?limit=20`: Returns 20 diverse test-set tourist trajectories (including full stop coordinates and ground-truth next POI) for instant demo playback.
    - `POST /recommend`: Runs inference and returns `RecommendResponse`.

### 5.2 Phase 5 Verification
- Write `tests/test_api.py` using FastAPI `TestClient` to verify:
  1. `/health` returns HTTP 200 with all 6 models registered.
  2. `/recommend` succeeds for both a known test user (`is_cold_start=False`) and an anonymous cold-start trail (`user_id=None`, `is_cold_start=True`).
  3. End-to-end CPU inference latency per request stays under `150 ms`.

---

## Phase 6: Interactive NYC Map Frontend (`app/frontend/`)
Build a responsive, split-screen web interface using **Vite + React + Leaflet** (`leaflet` and `react-leaflet`) with OpenStreetMap tiles (zero API keys required).

### 6.1 Frontend Component Breakdown
- `app/frontend/src/api/client.js`:
  - Centralized Axios/Fetch wrapper calling `GET /pois/search`, `GET /pois/categories`, `GET /users/sample`, and `POST /recommend`.
- `app/frontend/src/components/ControlSidebar.jsx`:
  - **Mode Switcher:**
    - `Returning Tourist`: Dropdown to load any of the 20 sample test trajectories and reveal/hide the ground-truth next POI on the map.
    - `Cold-Start Tourist`: Autocomplete search bar + map-click mode allowing the presenter to build a custom 1-to-5 stop NYC itinerary (e.g., `Central Park -> The Met -> Times Square`).
  - **Model Selector:** Dropdown to switch live between `Hybrid`, `Sequential+CL`, `Sequential`, `Content-Based`, `CF-BPR`, and `Popularity`.
  - **Knowledge-Based Constraints:** Slider for `max_distance_km` (`0.5 km` to `15.0 km`), multi-select category filter chips, and `Top-K` selector (`5`, `10`, `15`).
- `app/frontend/src/components/MapCanvas.jsx`:
  - Import `leaflet/dist/leaflet.css` explicitly and enforce an explicit viewport height (`100vh`) so tiles never collapse.
  - Center default view on Midtown Manhattan (`[40.7580, -73.9855]`, zoom `13`).
  - Render:
    - Past trajectory stops as numbered blue markers connected by a dashed polyline.
    - Active travel radius around the last stop as a translucent `<Circle />`.
    - Top-K recommended POIs as numbered red markers with interactive popups.
    - Optional ground-truth target marker (green star/pin) when inspecting test trajectories.
- `app/frontend/src/components/RecommendationList.jsx`:
  - Scrollable ranked cards displaying POI name, category badge, distance in km, explanation string, and horizontal mini-bars showing the contribution of Sequential, Content, and Spatial scores.
  - Clicking a card smoothly pans/zooms the Leaflet map (`map.flyTo`) to that POI and opens its popup.

### 6.2 Phase 6 Verification
- Run `npm install` and `npm run build` inside `app/frontend/` to verify zero bundler or missing-asset errors.
- Verify live end-to-end flow: start FastAPI on port `8000` and Vite on port `5173`, execute both demo modes, and confirm map markers update cleanly when switching models or adjusting radius sliders.

---

## Phase 7: Documentation, Submission Packaging & Academic Report Prep
Prepare all artifacts required for the HUST IT4613 course evaluation (source archive, setup guide, formal PDF report, and 15-minute team presentation).

### 7.1 Repository Documentation & Packaging
- `README.md`:
  - Complete step-by-step instructions covering Miniconda environment creation, `pip install -r requirements.txt`, Hugging Face dataset download (`CRUISEResearchGroup/Massive-STEPS-New-York`, `tabular` config), training commands for all 6 models, benchmark table generation, and launching backend + frontend servers.
- `scripts/package_submission.py`:
  - Utility script that validates required files exist and packages clean source code (`src/`, `app/`, `configs/`, `tests/`, `scripts/`, `requirements.txt`, `README.md`) into `artifacts/IT4613_NYC_POI_Recommender_Source.zip`, excluding large raw parquet caches, PyTorch checkpoints (unless requested), `.git/`, and `node_modules/`.

### 7.2 Academic Report Template (`docs/report_template.tex`)
Generate a clean LaTeX report skeleton using **Computer Modern** fonts and **Mermaid/PlantUML** architecture diagrams (saved as `.mmd` sources alongside exported PNG/SVG or TikZ equivalents for `pdflatex`) covering all 6 required IT4613 report sections:
1. **Introduction & Problem Formulation:** Real-world tourist Next-POI recommendation challenges (sequential dynamics, spatial constraints, cold-start tourists, data sparsity).
2. **Methodology:** Mathematical formulation and design of all 6 implemented approaches (Popularity, BPR-MF, Content-Based TF-IDF/Spatial, Spatial-Temporal SASRec/GRU4Rec, CL4SRec Contrastive Learning, and Context-Aware Hybrid Reranker).
3. **System Architecture & Prototype Features:** Data pipeline, CPU/GPU workload split, FastAPI + React-Leaflet architecture, and user guide screenshots.
4. **Third-Party Datasets, Libraries & Tools Attribution:** Explicit citation of `CRUISEResearchGroup/Massive-STEPS-New-York` (Foursquare Open Source Places), `PyTorch`, `implicit`, `scikit-learn`, `FastAPI`, `React`, and `Leaflet`.
5. **Challenges & Engineering Solutions:**
   - Handling 49,218-class softmax memory bottlenecks on a 6 GB VRAM GPU via mixed precision (`AMP`) and negative sampling.
   - Mitigating short-trajectory sparsity using self-supervised trajectory augmentations (`crop`, `mask`, `reorder`) in `sequential_cl.py`.
   - Balancing accuracy and diversity in `hybrid.py` so recommendations are geographically reachable and non-redundant.
6. **Experimental Evaluation, Discussion & Future Work:** Auto-populated LaTeX table from `artifacts/benchmark_results.csv` (`Recall@1/5/10`, `NDCG@5/10`, `MRR`, `Coverage`), qualitative case studies, and future extensions (weather/opening-hours API integration, multi-day itinerary routing).

### 7.3 15-Minute Presentation & Live Demo Script (`docs/DEMO_SCRIPT.md`)
Create a timed 15-minute presentation outline structured for a 2-to-4 student team:
- **00:00 - 02:30 | Problem & Dataset:** Motivation, NYC Massive-STEPS statistics (272k check-ins, 49k POIs), and train/val/test split integrity.
- **02:30 - 06:30 | Modeling Progression:** Progression from Collaborative Filtering & Content-Based baselines to Spatial-Temporal Sequential DL, Contrastive Learning, and the Knowledge-Based Hybrid Reranker.
- **06:30 - 09:30 | Offline Benchmark Results:** Quantitative comparison across `Recall@K`, `NDCG@K`, `MRR`, and `Coverage`, highlighting gains from spatial-temporal encoding and contrastive augmentation.
- **09:30 - 13:30 | Live Web Prototype Demo:**
  - Demo 1: Replay a returning test tourist trajectory in Manhattan and compare `Popularity` vs. `Hybrid` predictions on the map.
  - Demo 2: Build a cold-start tourist path from scratch, tighten the radius filter to `1.5 km`, filter by `Museum` / `Food`, and inspect explanation breakdowns.
- **13:30 - 15:00 | Limitations & Future Improvements:** Summary of key findings and Q&A readiness.

---

## Execution Instruction
Proceed sequentially through **Phases 1 through 7**. After each phase, run its verification checks and summarize created files and test outputs before moving forward.