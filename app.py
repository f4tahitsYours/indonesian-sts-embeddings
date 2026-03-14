"""
app.py — Indonesian STS Embeddings Demo
Streamlit application untuk demo model IndoBERT STS Indonesian
Model: Fatahillah01/indobert-sts-indonesian (HuggingFace Hub)
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IndoBERT STS Indonesian",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.02em;
}

code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Main background */
.main {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1e1e3f 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #818cf8;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}

/* Similarity gauge */
.similarity-display {
    background: linear-gradient(135deg, #1e1e3f, #2d2d5e);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
}

.similarity-score {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    line-height: 1;
    margin-bottom: 8px;
}

.similarity-label {
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-bottom: 16px;
}

.similarity-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin: 12px 0;
}

.similarity-bar {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Result cards for NN search */
.result-card {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.15);
    border-left: 4px solid;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: all 0.2s ease;
}

.result-card:hover {
    background: rgba(99,102,241,0.1);
    transform: translateX(4px);
}

/* Pipeline step */
.pipeline-step {
    background: linear-gradient(135deg, rgba(15,15,26,0.8), rgba(30,30,63,0.8));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    position: relative;
}

.step-number {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 10px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-pass { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-fail { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-best { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

/* Table styling */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.styled-table th {
    background: rgba(99,102,241,0.15);
    color: #c7d2fe;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.styled-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #e2e8f0;
}

.styled-table tr:hover td {
    background: rgba(99,102,241,0.05);
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Fatahillah01/indobert-sts-indonesian")
    return model


@st.cache_data(show_spinner=False)
def load_evaluation_results():
    """Load evaluation JSON files dari folder evaluation/."""
    eval_dir = Path("evaluation")
    results  = {}
    for fname in ["baseline_results.json", "simcse_results.json",
                  "sbert_results.json", "final_report.json",
                  "hard_negative_stats.json", "dataset_stats.json"]:
        path = eval_dir / fname
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[fname.replace(".json", "")] = json.load(f)
    return results


@st.cache_data(show_spinner=False)
def get_corpus_embeddings(_model):
    """Pre-encode corpus untuk semantic search."""
    corpus_emb = _model.encode(
        CORPUS,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return corpus_emb


# ── Corpus ────────────────────────────────────────────────────
CORPUS = [
    "Cara membuat nasi goreng spesial yang enak dan lezat.",
    "Resep rendang daging sapi khas Padang yang otentik.",
    "Soto ayam kuah bening dengan bumbu rempah pilihan.",
    "Gado-gado sayuran segar dengan bumbu kacang kental.",
    "Timnas Indonesia berhasil lolos ke final Piala AFF 2024.",
    "Pemain sepak bola mencetak gol di menit terakhir pertandingan.",
    "Klub olahraga nasional meraih medali emas di kejuaraan dunia.",
    "Presiden mengumumkan kenaikan upah minimum nasional 2025.",
    "Bank Indonesia mempertahankan suku bunga acuan di level 6 persen.",
    "Pemerintah meluncurkan program subsidi untuk UMKM terdampak.",
    "Gempa bumi berkekuatan 6.2 SR mengguncang wilayah Sulawesi Tengah.",
    "Kebakaran hutan di Kalimantan Selatan belum sepenuhnya padam.",
    "Banjir merendam ratusan rumah warga di pesisir Jakarta Utara.",
    "Mahasiswa ITB raih juara pertama olimpiade matematika internasional.",
    "Pemerintah meluncurkan program beasiswa untuk siswa berprestasi.",
    "Peluncuran satelit komunikasi Merah Putih 2 berhasil dilakukan.",
    "Wisata Bali kembali ramai dikunjungi wisatawan mancanegara.",
    "Festival Budaya Nusantara digelar meriah di Jakarta selama tiga hari.",
    "Peneliti BRIN berhasil temukan spesies baru endemik Papua.",
    "Proyek kereta cepat Jakarta-Bandung resmi beroperasi penuh.",
]

CORPUS_CATEGORIES = [
    "🍜 Kuliner", "🍜 Kuliner", "🍜 Kuliner", "🍜 Kuliner",
    "⚽ Olahraga", "⚽ Olahraga", "⚽ Olahraga",
    "💰 Ekonomi", "💰 Ekonomi", "💰 Ekonomi",
    "🌊 Bencana", "🌊 Bencana", "🌊 Bencana",
    "🎓 Pendidikan", "🎓 Pendidikan", "🔬 Teknologi",
    "✈️ Pariwisata", "🎭 Budaya", "🔬 Teknologi", "🚄 Infrastruktur",
]


# ── Helper functions ──────────────────────────────────────────
def get_similarity_color(sim):
    if sim >= 0.75:   return "#10b981", "Sangat Mirip"
    elif sim >= 0.55: return "#6366f1", "Mirip"
    elif sim >= 0.35: return "#f59e0b", "Agak Mirip"
    elif sim >= 0.15: return "#f97316", "Sedikit Mirip"
    else:              return "#ef4444", "Tidak Mirip"


def render_similarity_display(sim):
    color, label = get_similarity_color(sim)
    score_5 = sim * 5.0
    pct     = sim * 100
    st.markdown(f"""
    <div class="similarity-display">
        <div class="similarity-score" style="color:{color};">{sim:.4f}</div>
        <div class="similarity-label" style="color:{color};">{label}</div>
        <div style="color:#64748b; font-size:0.85rem; margin-bottom:16px;">
            Skala 0–5: <strong style="color:#94a3b8;">{score_5:.2f}</strong>
        </div>
        <div class="similarity-bar-container">
            <div class="similarity-bar" style="width:{pct}%;
                 background:linear-gradient(90deg, {color}99, {color});"></div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-size:0.72rem; color:#475569; margin-top:4px;">
            <span>0.0 — Tidak Mirip</span>
            <span>1.0 — Identik</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 24px;">
        <div style="font-size:2.5rem; margin-bottom:8px;">🔍</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;
                    color:#c7d2fe; line-height:1.2;">IndoBERT STS<br>Indonesian</div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:6px;">
            Semantic Textual Similarity
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📐 STS Score", "🔎 Semantic Search",
         "📊 Hasil Evaluasi", "ℹ️ Tentang Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; line-height:1.8;">
        <div style="color:#818cf8; font-weight:600; margin-bottom:6px;">Model Info</div>
        <div>🤗 Fatahillah01/indobert-sts-indonesian</div>
        <div>📐 768-dim embeddings</div>
        <div>📏 Max 128 tokens</div>
        <div>🌐 Bahasa Indonesia</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569;">
        <a href="https://huggingface.co/Fatahillah01/indobert-sts-indonesian"
           style="color:#818cf8; text-decoration:none;">🤗 HuggingFace Hub</a><br>
        <a href="https://github.com/f4tahitsYours/indonesian-sts-embeddings"
           style="color:#818cf8; text-decoration:none;">💻 GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)


# ── Load model & data ─────────────────────────────────────────
with st.spinner("⚡ Memuat model dari HuggingFace Hub..."):
    model    = load_model()
    eval_res = load_evaluation_results()
    corp_emb = get_corpus_embeddings(model)


# ══════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""
    <h1 style="font-size:2.8rem; color:#e2e8f0; margin-bottom:4px;">
        Optimizing Indonesian<br>
        <span style="color:#818cf8;">Sentence Embeddings</span>
    </h1>
    <p style="color:#64748b; font-size:1.05rem; margin-bottom:32px;">
        Semantic Textual Similarity menggunakan Contrastive Learning & Hard Negative Mining
    </p>
    """, unsafe_allow_html=True)

    # -- Metric cards
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("0.7091", "Test Spearman", "Model terbaik (SBERT)"),
        ("0.4653", "Baseline Spearman", "Zero-shot IndoBERT"),
        ("+0.2438", "Improvement", "SBERT vs Baseline"),
        ("2,968", "Hard Negatives", "BM25 triplets mined"),
    ]
    for col, (val, label, sub) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div style="font-size:0.75rem; color:#475569; margin-top:4px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Pipeline overview
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Pipeline Penelitian")
        steps = [
            ("1", "Environment Setup", "Google Colab · T4 GPU · HuggingFace cache"),
            ("2", "Dataset Preparation", "English STSB → Helsinki-NLP translation → 10,062 pairs"),
            ("3", "Zero-Shot Baseline", "IndoBERT + Mean Pooling → Spearman 0.4653"),
            ("4", "SimCSE Training", "Dropout augmentation · 1 epoch · Spearman 0.5885"),
            ("5", "Hard Negative Mining", "BM25 retrieval · 2,968 triplets · 99.9% coverage"),
            ("6", "SBERT Fine-Tuning", "CosineSimilarityLoss · 4 epochs → Spearman 0.7091"),
            ("7", "Evaluation", "t-SNE · Nearest Neighbor · Error Analysis"),
            ("8", "Deployment", "HuggingFace Hub + Streamlit Demo"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="step-number">{num}</span>
                <strong style="color:#c7d2fe;">{title}</strong>
                <div style="color:#64748b; font-size:0.82rem; margin-top:6px; padding-left:38px;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Model Comparison")
        model_data = {
            "Model": ["Baseline", "SimCSE", "SBERT"],
            "Spearman": [0.4653, 0.5885, 0.7091],
            "Pearson": [0.4597, 0.5929, 0.7225],
            "MAE": [0.3012, 0.1940, 0.1644],
            "Delta": ["—", "+0.1232", "+0.2438"],
        }
        df_model = pd.DataFrame(model_data)
        st.dataframe(
            df_model,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Dataset Statistics")
        ds_data = {
            "Split": ["Train", "Val", "Test"],
            "Pairs": ["5,696", "2,994", "1,372"],
            "Mean Score": ["0.540", "0.472", "0.521"],
        }
        st.dataframe(pd.DataFrame(ds_data), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Gunakan menu di sidebar untuk mengeksplorasi fitur demo secara interaktif.")


# ══════════════════════════════════════════════════════════════
# PAGE: STS SCORE
# ══════════════════════════════════════════════════════════════
elif page == "📐 STS Score":
    st.markdown("""
    <h2 style="color:#e2e8f0;">Semantic Textual Similarity Score</h2>
    <p style="color:#64748b;">
        Hitung seberapa mirip dua kalimat Bahasa Indonesia secara semantik.
        Model menggunakan cosine similarity antara 768-dimensi sentence embeddings.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        s1 = st.text_area(
            "Kalimat Pertama",
            placeholder="Contoh: Seorang pria bermain gitar di jalanan.",
            height=100,
        )
    with col2:
        s2 = st.text_area(
            "Kalimat Kedua",
            placeholder="Contoh: Ada seseorang yang memainkan alat musik.",
            height=100,
        )

    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        compute_btn = st.button("⚡ Hitung Similarity", type="primary", use_container_width=True)

    # -- Example pairs
    st.markdown("**Contoh pasangan kalimat:**")
    examples = [
        ("Seorang anak bermain bola di lapangan.", "Bocah itu sedang menendang bola di taman."),
        ("Harga BBM naik signifikan bulan ini.", "Pemerintah mengumumkan kenaikan harga bahan bakar."),
        ("Presiden berpidato di Istana Negara.", "Kucing tidur di atas sofa yang nyaman."),
        ("Timnas Indonesia menang di babak semifinal.", "Indonesia berhasil masuk final turnamen sepak bola."),
    ]
    ex_cols = st.columns(len(examples))
    for i, (ex_col, (ex1, ex2)) in enumerate(zip(ex_cols, examples)):
        with ex_col:
            if st.button(f"Contoh {i+1}", key=f"ex_{i}", use_container_width=True):
                s1 = ex1
                s2 = ex2

    if compute_btn or (s1 and s2):
        if not s1.strip() or not s2.strip():
            st.warning("Masukkan kedua kalimat terlebih dahulu.")
        else:
            with st.spinner("Menghitung similarity..."):
                e1  = model.encode(s1.strip(), normalize_embeddings=True)
                e2  = model.encode(s2.strip(), normalize_embeddings=True)
                sim = float(np.dot(e1, e2))

            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                render_similarity_display(sim)

            with col_res2:
                st.markdown("#### Interpretasi")
                color, label = get_similarity_color(sim)

                interp_data = {
                    "Metrik": ["Cosine Similarity", "Skala 0–5", "Kategori", "Percentile"],
                    "Nilai": [
                        f"{sim:.6f}",
                        f"{sim*5:.4f} / 5.0",
                        label,
                        f"~{int(sim*100)}th percentile",
                    ]
                }
                st.dataframe(
                    pd.DataFrame(interp_data),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Panduan Interpretasi")
                ranges = [
                    ("≥ 0.75", "Sangat Mirip", "#10b981", "Hampir identik atau parafrase"),
                    ("0.55–0.75", "Mirip", "#6366f1", "Makna serupa, kata berbeda"),
                    ("0.35–0.55", "Agak Mirip", "#f59e0b", "Ada overlap makna"),
                    ("0.15–0.35", "Sedikit Mirip", "#f97316", "Sedikit keterkaitan"),
                    ("< 0.15", "Tidak Mirip", "#ef4444", "Tidak berkaitan"),
                ]
                for rng, lbl, clr, desc in ranges:
                    active = "→ " if lbl == label else "  "
                    st.markdown(
                        f"`{rng}` &nbsp; "
                        f"<span style='color:{clr};font-weight:600;'>{active}{lbl}</span>"
                        f" &nbsp;—&nbsp; <span style='color:#64748b;font-size:0.85rem;'>{desc}</span>",
                        unsafe_allow_html=True
                    )

                st.markdown("#### Kalimat yang dianalisis")
                st.markdown(f"**S1:** {s1}")
                st.markdown(f"**S2:** {s2}")


# ══════════════════════════════════════════════════════════════
# PAGE: SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════
elif page == "🔎 Semantic Search":
    st.markdown("""
    <h2 style="color:#e2e8f0;">Semantic Search</h2>
    <p style="color:#64748b;">
        Temukan kalimat paling relevan secara semantik dari corpus berita Indonesia.
        Pencarian berbasis makna — bukan keyword matching.
    </p>
    """, unsafe_allow_html=True)

    col_q, col_k = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Query",
            placeholder="Contoh: makanan tradisional Indonesia",
            label_visibility="collapsed",
        )
    with col_k:
        top_k = st.selectbox("Top-K", [3, 5, 10, 15], index=1)

    col_search, _ = st.columns([1, 4])
    with col_search:
        search_btn = st.button("🔎 Cari", type="primary", use_container_width=True)

    # Example queries
    st.markdown("**Query cepat:**")
    quick_queries = [
        "makanan khas Indonesia", "bencana alam",
        "prestasi olahraga", "kebijakan ekonomi pemerintah",
        "teknologi dan inovasi", "pariwisata Indonesia",
    ]
    q_cols = st.columns(len(quick_queries))
    for q_col, qq in zip(q_cols, quick_queries):
        with q_col:
            if st.button(qq, key=f"qq_{qq}", use_container_width=True):
                query = qq

    if (search_btn or query) and query.strip():
        with st.spinner("Mencari..."):
            q_emb = model.encode(query.strip(), normalize_embeddings=True)
            sims  = np.dot(corp_emb, q_emb)
            top_idx = np.argsort(sims)[::-1][:int(top_k)]

        st.markdown(f"---\n**Hasil untuk:** *\"{query}\"* &nbsp;·&nbsp; {top_k} hasil teratas")

        for rank, idx in enumerate(top_idx, 1):
            sim   = sims[idx]
            color, label = get_similarity_color(sim)
            cat   = CORPUS_CATEGORIES[idx]
            bar_w = int(sim * 100)

            st.markdown(f"""
            <div class="result-card" style="border-left-color:{color};">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="display:flex; align-items:center; gap:10px; flex:1;">
                        <span style="color:{color}; font-family:'DM Serif Display',serif;
                                     font-size:1.5rem; min-width:28px;">#{rank}</span>
                        <div style="flex:1;">
                            <div style="color:#e2e8f0; font-size:0.95rem; line-height:1.5;">
                                {CORPUS[idx]}
                            </div>
                            <div style="margin-top:6px;">
                                <span style="color:#64748b; font-size:0.75rem;">{cat}</span>
                            </div>
                        </div>
                    </div>
                    <div style="text-align:right; min-width:90px; padding-left:16px;">
                        <div style="color:{color}; font-family:'DM Serif Display',serif;
                                    font-size:1.3rem;">{sim:.4f}</div>
                        <div style="color:{color}; font-size:0.72rem;">{label}</div>
                    </div>
                </div>
                <div style="margin-top:10px; background:rgba(255,255,255,0.04);
                            border-radius:100px; height:4px; overflow:hidden;">
                    <div style="width:{bar_w}%; height:100%; border-radius:100px;
                                background:linear-gradient(90deg,{color}66,{color});"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Corpus yang Digunakan")
        corpus_df = pd.DataFrame({
            "#": range(1, len(CORPUS)+1),
            "Kategori": CORPUS_CATEGORIES,
            "Kalimat": CORPUS,
        })
        st.dataframe(corpus_df, use_container_width=True, hide_index=True, height=300)


# ══════════════════════════════════════════════════════════════
# PAGE: HASIL EVALUASI
# ══════════════════════════════════════════════════════════════
elif page == "📊 Hasil Evaluasi":
    st.markdown("""
    <h2 style="color:#e2e8f0;">Hasil Evaluasi Model</h2>
    <p style="color:#64748b;">
        Perbandingan komprehensif tiga strategi training pada Indonesian STS benchmark.
    </p>
    """, unsafe_allow_html=True)

    # -- Main comparison table
    st.markdown("### Perbandingan Model — Test Set")

    comp_data = {
        "Model": ["Zero-Shot Baseline", "SimCSE (Unsupervised)", "SBERT + Hard Neg"],
        "Strategy": ["No fine-tuning", "Contrastive dropout", "CosineSimilarityLoss"],
        "Val Spearman": [0.5625, 0.6783, 0.7688],
        "Test Spearman": [0.4653, 0.5885, 0.7091],
        "Test Pearson": [0.4597, 0.5929, 0.7225],
        "Test MAE": [0.3012, 0.1940, 0.1644],
        "Δ Baseline": ["—", "+0.1232", "+0.2438"],
    }
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # -- Success criteria
    st.markdown("### Success Criteria")
    criteria = [
        ("SBERT Spearman ≥ 0.80", False, "0.7091", "Performance ceiling akibat machine-translated data"),
        ("SBERT beats baseline ≥ +0.10", True, "+0.2438", "Melampaui target +0.10"),
        ("SimCSE beats baseline (tanpa label)", True, "+0.1232", "Unsupervised contrastive berhasil"),
    ]
    for desc, passed, val, note in criteria:
        badge = '<span class="badge badge-pass">PASS</span>' if passed else '<span class="badge badge-fail">FAIL</span>'
        st.markdown(
            f"{badge} &nbsp; **{desc}** &nbsp; `{val}` &nbsp;"
            f"<span style='color:#64748b; font-size:0.85rem;'>{note}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # -- Per-category
    st.markdown("### Per-Kategori Spearman — Test Set")
    cat_data = {
        "Kategori": ["Low (0–2)", "Mid (2–4)", "High (4–5)"],
        "N Pairs": [439, 599, 334],
        "Baseline": [0.3848, 0.1733, 0.2367],
        "SimCSE": [0.4194, 0.2814, 0.2418],
        "SBERT": [0.4887, 0.4080, 0.3069],
        "Best": ["SBERT", "SBERT", "SBERT"],
    }
    df_cat = pd.DataFrame(cat_data)
    st.dataframe(df_cat, use_container_width=True, hide_index=True)

    st.info(
        "📌 **Insight:** Semua model paling kesulitan di kategori High (0.8–1.0). "
        "Pasangan kalimat yang hampir identik (near-paraphrase) membutuhkan pemahaman "
        "semantik yang sangat dalam untuk dibedakan secara presisi."
    )

    st.markdown("---")

    # -- Training details
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### SimCSE Training Details")
        simcse_data = {
            "Parameter": ["Corpus size", "Epochs", "Batch size", "Learning rate",
                          "Loss function", "Temperature (1/scale)", "Training time"],
            "Value": ["10,373 sentences", "1", "64", "3e-5",
                      "MultipleNegativesRankingLoss", "0.05 (scale=20)", "~6 menit (T4)"],
        }
        st.dataframe(pd.DataFrame(simcse_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### SBERT Training Details")
        sbert_data = {
            "Parameter": ["Training pairs", "Epochs", "Batch size", "Learning rate",
                          "Loss function", "Hard negatives", "Training time"],
            "Value": ["5,696 pairs", "4", "32", "2e-5",
                      "CosineSimilarityLoss", "2,968 triplets (BM25)", "~7 menit (T4)"],
        }
        st.dataframe(pd.DataFrame(sbert_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # -- Hard negative stats
    st.markdown("### Hard Negative Mining Statistics")
    col1, col2, col3, col4 = st.columns(4)
    hn_metrics = [
        ("2,968", "Total Triplets"),
        ("99.9%", "Coverage"),
        ("19.77", "BM25 Score Mean"),
        ("0.778", "A-P Score Mean"),
    ]
    for col, (val, lbl) in zip([col1,col2,col3,col4], hn_metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.8rem;">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # -- Error analysis
    st.markdown("### Error Analysis — SBERT Worst Predictions")
    error_data = {
        "Type": ["FP", "FP", "FP", "FN", "FN"],
        "Gold": [0.000, 0.040, 0.160, 0.880, 1.000],
        "Predicted": [0.994, 0.951, 0.913, 0.182, 0.335],
        "Error": [0.994, 0.911, 0.753, 0.698, 0.665],
        "Pattern": [
            "Same-topic → high sim",
            "Named entity overlap",
            "Topic sim ≠ semantic sim",
            "English sentence (OOD)",
            "Paraphrase diff structure",
        ],
    }
    st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.2);
                border-radius:12px; padding:16px; margin-top:8px;">
        <strong style="color:#818cf8;">📌 Temuan Error Analysis:</strong>
        <ul style="color:#94a3b8; margin-top:8px; font-size:0.88rem;">
            <li><strong>False Positives mendominasi</strong> — model salah mengira kalimat
                yang topiknya sama sebagai semantically similar</li>
            <li><strong>English OOD</strong> — 1 kalimat English lolos ke test set,
                IndoBERT tidak bisa handle dengan baik</li>
            <li><strong>Near-paraphrase (FN)</strong> — model kesulitan mengenali
                parafrase dengan struktur sintaksis yang sangat berbeda</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # -- Evaluation plots
    st.markdown("---")
    st.markdown("### Visualisasi Evaluasi")
    eval_dir = Path("evaluation")
    plot_files = {
        "t-SNE Embedding Space": "tsne_comparison.png",
        "Full Model Comparison": "full_comparison.png",
        "Error Distribution": "error_analysis.png",
        "Hard Negative Analysis": "hard_negative_analysis.png",
    }
    tabs = st.tabs(list(plot_files.keys()))
    for tab, (title, fname) in zip(tabs, plot_files.items()):
        with tab:
            path = eval_dir / fname
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Plot tidak ditemukan: `evaluation/{fname}`")


# ══════════════════════════════════════════════════════════════
# PAGE: TENTANG MODEL
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Model":
    st.markdown("""
    <h2 style="color:#e2e8f0;">Tentang Model & Project</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Deskripsi Project")
        st.markdown("""
        Project ini mengembangkan dan membandingkan tiga strategi sentence embedding
        untuk **Semantic Textual Similarity (STS)** Bahasa Indonesia menggunakan
        IndoBERT sebagai base encoder.

        **Masalah yang diselesaikan:**
        Bahasa Indonesia yang dituturkan 270+ juta orang belum memiliki sentence
        embedding model yang teroptimasi khusus untuk STS task. Model multilingual
        yang ada tidak dioptimalkan untuk monolingual Indonesian similarity.

        **Kontribusi teknis:**
        - Implementasi SimCSE untuk Indonesian tanpa labeled data
        - Pipeline BM25 hard negative mining yang efisien (99.9% coverage, <1 detik indexing)
        - Dataset Indonesian STS dari machine translation EN→ID via Helsinki-NLP
        """)

        st.markdown("### Arsitektur Model")
        st.markdown("""
        ```
        Input Sentence (max 128 tokens)
              ↓
        IndoBERT Tokenizer
              ↓
        IndoBERT Encoder (12 layers, 768-dim hidden)
        [indobenchmark/indobert-base-p1]
              ↓
        Mean Pooling over token embeddings
        (weighted by attention mask)
              ↓
        L2 Normalization
              ↓
        768-dimensional Sentence Embedding
        ```
        Similarity = Cosine(emb_A, emb_B) = dot(emb_A, emb_B)
        [karena sudah L2-normalized]
        """)

        st.markdown("### Cara Penggunaan")
        st.code("""
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer(
    "Fatahillah01/indobert-sts-indonesian"
)

# Encode sentences
sentences = [
    "Seorang pria bermain gitar.",
    "Ada seseorang memainkan alat musik.",
    "Presiden berpidato di istana.",
]
embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

# Compute similarity
sim = float(np.dot(embeddings[0], embeddings[1]))
print(f"Similarity: {sim:.4f}")
        """, language="python")

    with col2:
        st.markdown("### Model Information")
        info_data = {
            "Property": [
                "Model ID", "Base Model", "Pooling",
                "Embedding Dim", "Max Seq Length", "Language",
                "Training Loss", "Fine-tuning Epochs",
                "Training Pairs", "Test Spearman",
            ],
            "Value": [
                "Fatahillah01/indobert-sts-indonesian",
                "indobert-base-p1",
                "Mean Pooling",
                "768",
                "128 tokens",
                "Bahasa Indonesia",
                "CosineSimilarityLoss",
                "4 epochs",
                "5,696",
                "0.7091",
            ],
        }
        st.dataframe(
            pd.DataFrame(info_data),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Dataset")
        st.markdown("""
        **Source:** STS Benchmark (English)  
        **Translation:** Helsinki-NLP/opus-mt-en-id  
        **Method:** Greedy decoding, batch=128  

        | Split | Pairs |
        |-------|-------|
        | Train | 5,696 |
        | Val   | 2,994 |
        | Test  | 1,372 |
        | Total | 10,062 |
        """)

        st.markdown("### Links")
        st.markdown("""
        - [🤗 HuggingFace Hub](https://huggingface.co/Fatahillah01/indobert-sts-indonesian)
        - [💻 GitHub Repository](https://github.com/f4tahitsYours/indonesian-sts-embeddings)
        - [📄 SimCSE Paper](https://arxiv.org/abs/2104.08821)
        - [📄 SBERT Paper](https://arxiv.org/abs/1908.10084)
        """)

        st.markdown("### Key Findings")
        findings = [
            ("SimCSE +0.12 tanpa label", "Dropout augmentation efektif untuk Indonesian"),
            ("SBERT +0.24 vs baseline", "Supervised STS training signifikan"),
            ("Ceiling ~0.71", "Akibat noise dari machine translation"),
            ("High sim paling sulit", "Near-paraphrase butuh model yang lebih besar"),
        ]
        for title, desc in findings:
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.06); border-left:3px solid #6366f1;
                        border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0;">
                <strong style="color:#c7d2fe; font-size:0.88rem;">{title}</strong>
                <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)