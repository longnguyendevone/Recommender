"""Validate required project files and package clean source code for submission."""

from __future__ import annotations

import zipfile
from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "configs/data.yaml",
    "configs/baselines.yaml",
    "configs/sequential.yaml",
    "configs/hybrid.yaml",
    "src/train.py",
    "src/data/download_eda.py",
    "src/data/preprocess.py",
    "src/data/dataset.py",
    "src/models/base.py",
    "src/models/popularity.py",
    "src/models/cf_bpr.py",
    "src/models/content_based.py",
    "src/models/sequential.py",
    "src/models/sequential_cl.py",
    "src/models/hybrid.py",
    "src/evaluation/metrics.py",
    "src/evaluation/benchmark.py",
    "app/backend/main.py",
    "app/backend/schemas.py",
    "app/backend/recommender_service.py",
    "app/frontend/package.json",
    "app/frontend/src/App.jsx",
    "docs/report_template.tex",
    "docs/DEMO_SCRIPT.md",
    "docs/architecture.mmd",
    "tests/test_api.py",
]

INCLUDE_ROOTS = [
    "configs",
    "src",
    "app",
    "docs",
    "scripts",
    "tests",
]

INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    "plan.md",
    ".gitignore",
]

EXCLUDE_PARTS = {
    ".venv",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".git",
}


def package_submission(
    repo_root: Path = Path("."),
    out_zip: Path = Path("artifacts/IT4613_NYC_POI_Recommender_Source.zip"),
) -> Path:
    missing = [rel for rel in REQUIRED_FILES if not (repo_root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    added: list[str] = []

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_FILES:
            p = repo_root / rel
            if p.exists():
                zf.write(p, arcname=rel)
                added.append(rel)

        for root_name in INCLUDE_ROOTS:
            root_path = repo_root / root_name
            if not root_path.exists():
                continue
            for file_path in sorted(root_path.rglob("*")):
                if not file_path.is_file():
                    continue
                if any(part in EXCLUDE_PARTS for part in file_path.parts):
                    continue
                if file_path.suffix in {".pyc", ".pyo"}:
                    continue
                rel = file_path.relative_to(repo_root).as_posix()
                zf.write(file_path, arcname=rel)
                added.append(rel)

        # Include benchmark summaries & plots (small artifacts) without heavy checkpoints.
        for extra in [
            "artifacts/benchmark_results.csv",
            "artifacts/benchmark_results.md",
            "artifacts/benchmark_comparison.png",
            "artifacts/eda/eda_summary.json",
        ]:
            ep = repo_root / extra
            if ep.exists():
                zf.write(ep, arcname=extra)
                added.append(extra)

    size_kb = out_zip.stat().st_size / 1024.0
    print(f"Packaged {len(added)} files -> {out_zip} ({size_kb:.1f} KB)")
    return out_zip


if __name__ == "__main__":
    package_submission()
