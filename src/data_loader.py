"""
src/data_loader.py
──────────────────
Reusable data loading utilities untuk seluruh pipeline.
Di-import oleh notebook Stage 3–10.

Cara penggunaan:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/AI-Projects/sts-indonesian-embeddings")
    from src.data_loader import load_splits, load_simcse_corpus, STSPairDataset
"""

import os
import re
import json
import random
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sentence_transformers import InputExample


# ──────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Minimal cleaning untuk Indonesian STS.
    Tidak melakukan lowercase/stemming (IndoBERT tokenizer handles it).
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# LOAD CSV SPLITS
# ──────────────────────────────────────────────────────────────

def load_splits(
    splits_dir: Union[str, Path],
    splits: Tuple[str, ...] = ('train', 'val', 'test'),
    score_col: str = 'score'          # 'score' (0–1) atau 'score_raw' (0–5)
) -> Dict[str, pd.DataFrame]:
    """
    Load train/val/test splits dari CSV.

    Returns:
        Dict dengan key 'train', 'val', 'test' (sesuai yang diminta).

    Example:
        data = load_splits(PATHS['splits'])
        df_train = data['train']
    """
    splits_dir = Path(splits_dir)
    result = {}
    for split in splits:
        path = splits_dir / f"{split}_pairs.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file tidak ditemukan: {path}\n"
                f"Pastikan Stage 2 (Dataset Setup) sudah dijalankan."
            )
        df = pd.read_csv(path)
        # Validasi kolom wajib
        required_cols = {'sentence1', 'sentence2', score_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Kolom tidak ditemukan di {path.name}: {missing}")
        result[split] = df
        print(f"[OK] Loaded {split}: {len(df):,} pairs  |  "
              f"score [{df[score_col].min():.2f}–{df[score_col].max():.2f}]")
    return result


# ──────────────────────────────────────────────────────────────
# SENTENCE TRANSFORMERS INPUT EXAMPLES
# ──────────────────────────────────────────────────────────────

def df_to_input_examples(
    df: pd.DataFrame,
    score_col: str = 'score'
) -> List[InputExample]:
    """
    Konversi DataFrame ke list InputExample untuk sentence-transformers.
    Digunakan oleh: Baseline evaluation, SBERT training.

    InputExample.texts = [sentence1, sentence2]
    InputExample.label = score (float 0–1)
    """
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[str(row['sentence1']), str(row['sentence2'])],
            label=float(row[score_col])
        ))
    return examples


def df_to_simcse_examples(sentences: List[str]) -> List[InputExample]:
    """
    Buat InputExample untuk SimCSE unsupervised training.
    Setiap kalimat dipasangkan dengan dirinya sendiri (dropout = augmentasi).

    InputExample.texts = [sent, sent]  ← same sentence twice
    """
    return [
        InputExample(texts=[s, s])
        for s in sentences
        if len(s.strip()) >= 10
    ]


def build_triplet_examples(
    triplets_csv: Union[str, Path]
) -> List[InputExample]:
    """
    Load hard negative triplets dari CSV dan konversi ke InputExample.
    Format CSV: anchor, positive, hard_negative

    InputExample.texts = [anchor, positive, hard_negative]
    Digunakan dengan MultipleNegativesRankingLoss.
    """
    df = pd.read_csv(triplets_csv)
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[
                str(row['anchor']),
                str(row['positive']),
                str(row['hard_negative'])
            ]
        ))
    print(f"[OK] Loaded {len(examples):,} hard negative triplets.")
    return examples


# ──────────────────────────────────────────────────────────────
# SIMCSE CORPUS
# ──────────────────────────────────────────────────────────────

def load_simcse_corpus(
    txt_path: Union[str, Path],
    max_sentences: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42
) -> List[str]:
    """
    Load SimCSE sentence corpus dari .txt file.

    Args:
        txt_path       : Path ke simcse_sentences.txt
        max_sentences  : Batasi jumlah kalimat (None = pakai semua)
        shuffle        : Acak urutan

    Returns:
        List[str] kalimat
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"SimCSE corpus tidak ditemukan: {txt_path}")

    with open(txt_path, encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    if shuffle:
        random.seed(seed)
        random.shuffle(sentences)

    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    print(f"[OK] Loaded {len(sentences):,} sentences for SimCSE.")
    return sentences


# ──────────────────────────────────────────────────────────────
# MINING CORPUS
# ──────────────────────────────────────────────────────────────

def load_mining_corpus(
    txt_path: Union[str, Path]
) -> List[str]:
    """Load mining corpus untuk BM25 hard negative retrieval."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Mining corpus tidak ditemukan: {txt_path}")

    with open(txt_path, encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"[OK] Loaded {len(sentences):,} sentences for BM25 mining corpus.")
    return sentences


# ──────────────────────────────────────────────────────────────
# DATASET STATS
# ──────────────────────────────────────────────────────────────

def load_dataset_stats(stats_json: Union[str, Path]) -> Dict:
    """Load dataset statistics JSON dari Stage 2."""
    with open(stats_json, encoding='utf-8') as f:
        return json.load(f)


def print_split_summary(data: Dict[str, pd.DataFrame], score_col: str = 'score') -> None:
    """Print ringkasan splits yang sudah diload."""
    print("\n" + "=" * 55)
    print("  DATASET SPLIT SUMMARY")
    print("=" * 55)
    total = 0
    for split_name, df in data.items():
        n = len(df)
        total += n
        score = df[score_col]
        n_low  = (score < 0.4).sum()
        n_mid  = ((score >= 0.4) & (score < 0.8)).sum()
        n_high = (score >= 0.8).sum()
        print(f"\n  {split_name.upper():6} : {n:,} pairs")
        print(f"    Score  : mean={score.mean():.3f}  std={score.std():.3f}")
        print(f"    Low    : {n_low:,} ({n_low/n*100:.1f}%)  "
              f"Mid: {n_mid:,} ({n_mid/n*100:.1f}%)  "
              f"High: {n_high:,} ({n_high/n*100:.1f}%)")
    print(f"\n  TOTAL  : {total:,} pairs")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────
# QUICK TEST (jalankan langsung sebagai script)
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from project_config import PATHS

    print("[TEST] data_loader.py\n")

    data = load_splits(PATHS['splits'])
    print_split_summary(data)

    simcse = load_simcse_corpus(PATHS['splits'] / 'simcse_sentences.txt', max_sentences=5)
    print(f"\n[TEST] SimCSE sample: {simcse[:2]}")

    mining = load_mining_corpus(PATHS['mining_corpus'] / 'mining_corpus.txt')
    print(f"[TEST] Mining corpus size: {len(mining)}")

    examples = df_to_input_examples(data['train'].head(3))
    print(f"[TEST] InputExamples: {examples[0].texts}  label={examples[0].label}")

    print("\n[OK] data_loader.py — semua test passed.")
