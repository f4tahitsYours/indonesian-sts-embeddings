
"""
project_config.py
─────────────────
Centralized path configuration untuk seluruh notebook.
Cara penggunaan di notebook lain:

    import sys
    sys.path.insert(0, "/content/drive/MyDrive/AI-Projects/sts-indonesian-embeddings")
    from project_config import PATHS, PROJECT_ROOT, HF_CACHE_DIR
"""

from pathlib import Path

GDRIVE_ROOT   = Path("/content/drive/MyDrive")
PROJECT_ROOT  = GDRIVE_ROOT / "AI-Projects" / "sts-indonesian-embeddings"
HF_CACHE_DIR  = str(PROJECT_ROOT / "hf_cache")

PATHS = {
    "datasets"        : PROJECT_ROOT / "datasets",
    "indosts"         : PROJECT_ROOT / "datasets" / "indosts",
    "stsb_id"         : PROJECT_ROOT / "datasets" / "stsb-id",
    "mining_corpus"   : PROJECT_ROOT / "datasets" / "mining-corpus",
    "splits"          : PROJECT_ROOT / "datasets" / "splits",
    "experiments"     : PROJECT_ROOT / "experiments",
    "baseline_exp"    : PROJECT_ROOT / "experiments" / "baseline",
    "simcse_exp"      : PROJECT_ROOT / "experiments" / "simcse",
    "sbert_exp"       : PROJECT_ROOT / "experiments" / "sbert-hard-negatives",
    "models"          : PROJECT_ROOT / "models",
    "baseline_model"  : PROJECT_ROOT / "models" / "baseline",
    "simcse_model"    : PROJECT_ROOT / "models" / "simcse",
    "sbert_model"     : PROJECT_ROOT / "models" / "sbert",
    "hf_cache"        : PROJECT_ROOT / "hf_cache",
    "logs"            : PROJECT_ROOT / "logs",
    "embeddings"      : PROJECT_ROOT / "embeddings",
    "evaluation"      : PROJECT_ROOT / "evaluation",
    "demo"            : PROJECT_ROOT / "demo",
}

# Pastikan semua dir ada saat di-import
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)
