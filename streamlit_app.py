"""
streamlit_app.py — VocalCanvas Professional Dashboard
======================================================
Run with:
    streamlit run streamlit_app.py
"""

import io
import os
import tempfile

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Import shared state from the inference pipeline ───────────────────────────
# predict.py is the single source of truth for constants and loaders.
# We do NOT duplicate any DSP or model-loading logic here.
from predict import (
    PROC,
    SPEAKERS,
    CNN_MODEL_PATH,
    KMEANS_PKL_PATH,
    SAMPLE_RATE,
    CHUNK_DURATION,
    convert_to_wav,
    load_cnn_model      as _load_cnn,
    load_kmeans_bundle  as _load_kmeans,
    cluster_confidence,
)

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="VocalCanvas",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Global CSS — dark, premium aesthetic
# =============================================================================
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero header ── */
    .vc-hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .vc-hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .vc-hero p {
        color: #94a3b8;
        margin-top: 0.4rem;
        font-size: 1rem;
    }

    /* ── Card panels ── */
    .vc-card {
        background: #1e1e2e;
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
    }
    .vc-card h3 {
        color: #a78bfa;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Winner badge ── */
    .winner-badge {
        background: linear-gradient(135deg, #1a472a, #166534);
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .winner-badge h2 {
        color: #4ade80;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    .winner-badge p {
        color: #86efac;
        margin: 0.3rem 0 0 0;
    }

    /* ── Chunk table ── */
    .stDataFrame { border-radius: 8px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #13131f !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #a78bfa !important; }

    /* ── Divider ── */
    hr { border-color: #2d2d44; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached model loaders  (run once per process, persisted across reruns)
# =============================================================================
@st.cache_resource(show_spinner="Loading CNN model…")
def get_cnn_model():
    if not os.path.exists(CNN_MODEL_PATH):
        return None
    return _load_cnn()


@st.cache_resource(show_spinner="Loading K-Means bundle…")
def get_kmeans_bundle():
    if not os.path.exists(KMEANS_PKL_PATH):
        return None
    return _load_kmeans()


# =============================================================================
# Inference helpers — return structured dicts instead of printing
# =============================================================================
def _run_cnn(wav_path: str, model) -> dict:
    """CNN chunk-by-chunk inference.  Returns structured result dict."""
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    chunks   = PROC.split_audio(audio)

    chunk_details = []
    votes         = {s: 0   for s in SPEAKERS}
    conf_lists    = {s: []  for s in SPEAKERS}

    progress = st.progress(0, text="Analysing chunks…")

    for idx, chunk in enumerate(chunks):
        progress.progress((idx + 1) / max(len(chunks), 1),
                          text=f"Chunk {idx+1} / {len(chunks)}")

        if PROC.is_silent(chunk):
            chunk_details.append({
                "Chunk": idx, "Speaker": "—",
                "Confidence": "—", "Status": "Silent"
            })
            continue

        spec  = np.expand_dims(PROC.chunk_to_spectrogram(chunk), 0)  # (1,128,128,1)
        probs = model.predict(spec, verbose=0)[0]
        pidx  = int(np.argmax(probs))
        spk   = SPEAKERS[pidx]
        conf  = float(probs[pidx])

        votes[spk] += 1
        conf_lists[spk].append(conf)

        chunk_details.append({
            "Chunk": idx, "Speaker": spk,
            "Confidence": f"{conf*100:.1f}%", "Status": "✓"
        })

    progress.empty()

    total = sum(votes.values())
    if total == 0:
        return None

    avg_conf = {
        s: (float(np.mean(conf_lists[s])) if conf_lists[s] else 0.0)
        for s in SPEAKERS
    }

    return {
        "winner":        max(votes, key=votes.get),
        "votes":         votes,
        "vote_pcts":     {s: votes[s] / total * 100 for s in SPEAKERS},
        "avg_conf":      avg_conf,
        "chunk_details": chunk_details,
        "total":         total,
        "model_type":    "CNN",
    }


def _run_kmeans(wav_path: str, bundle: dict) -> dict:
    """K-Means chunk-by-chunk inference.  Returns structured result dict."""
    kmeans      = bundle["kmeans"]
    scaler      = bundle["scaler"]
    cluster_map = bundle["cluster_map"]

    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    chunks   = PROC.split_audio(audio)

    chunk_details = []
    votes         = {s: 0  for s in SPEAKERS}
    conf_lists    = {s: [] for s in SPEAKERS}

    progress = st.progress(0, text="Analysing chunks…")

    for idx, chunk in enumerate(chunks):
        progress.progress((idx + 1) / max(len(chunks), 1),
                          text=f"Chunk {idx+1} / {len(chunks)}")

        if PROC.is_silent(chunk):
            chunk_details.append({
                "Chunk": idx, "Speaker": "—",
                "Confidence": "—", "Status": "Silent"
            })
            continue

        feat       = PROC.chunk_to_kmeans_features(chunk)
        feat_sc    = scaler.transform(feat.reshape(1, -1))
        cluster_id = int(kmeans.predict(feat_sc)[0])
        spk        = cluster_map[cluster_id]
        conf       = cluster_confidence(kmeans, scaler, feat)

        votes[spk] += 1
        conf_lists[spk].append(conf)

        chunk_details.append({
            "Chunk":      idx,
            "Speaker":    spk,
            "Confidence": f"{conf*100:.1f}%",
            "Status":     f"Cluster {cluster_id}",
        })

    progress.empty()

    total = sum(votes.values())
    if total == 0:
        return None

    avg_conf = {
        s: (float(np.mean(conf_lists[s])) if conf_lists[s] else 0.0)
        for s in SPEAKERS
    }

    return {
        "winner":        max(votes, key=votes.get),
        "votes":         votes,
        "vote_pcts":     {s: votes[s] / total * 100 for s in SPEAKERS},
        "avg_conf":      avg_conf,
        "chunk_details": chunk_details,
        "total":         total,
        "model_type":    "K-Means",
    }


# =============================================================================
# Visualisation helpers
# =============================================================================
def plot_mel_spectrogram(audio: np.ndarray, sr: int) -> plt.Figure:
    """Render a publication-quality mel spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    ax.set_title("Mel Spectrogram", color="#a78bfa", fontsize=12, pad=8)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Frequency (mel)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d44")

    fig.tight_layout()
    return fig


def plot_results_chart(result: dict) -> plt.Figure:
    """Horizontal bar chart — vote % + average confidence per speaker."""
    speakers  = SPEAKERS
    vote_pcts = [result["vote_pcts"].get(s, 0) for s in speakers]
    avg_confs = [result["avg_conf"].get(s, 0) * 100 for s in speakers]
    winner    = result["winner"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(2.5, len(speakers) * 0.9)))
    fig.patch.set_facecolor("#1e1e2e")

    colors = ["#4ade80" if s == winner else "#60a5fa" for s in speakers]

    for ax, values, title, xlabel, fmt in [
        (ax1, vote_pcts, "Vote Share (%)",       "% of chunks", "{:.0f}%"),
        (ax2, avg_confs, "Avg Confidence (%)",   "confidence",  "{:.1f}%"),
    ]:
        ax.set_facecolor("#13131f")
        bars = ax.barh(speakers, values, color=colors,
                       height=0.55, edgecolor="none")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                va="center", ha="left",
                color="white", fontsize=10, fontweight="600",
            )

        ax.set_xlim(0, max(values) * 1.25 + 5)
        ax.set_title(title, color="#a78bfa", fontsize=11, pad=8)
        ax.set_xlabel(xlabel, color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#13131f")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d2d44")

    fig.tight_layout(pad=2)
    return fig


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:

    st.markdown("## 🎙️ VocalCanvas")
    st.markdown("---")

    # ── Model selector ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model Configuration")

    model_choice = st.radio(
        "Select inference engine",
        options=["CNN (Deep Learning)", "K-Means (Statistical Clustering)"],
        index=0,
        help=(
            "CNN: Trained on mel spectrogram images. High accuracy.\n\n"
            "K-Means: Trained on 144-dim DSP features (MFCCs, spectral "
            "contrast, deltas). Fully unsupervised."
        ),
    )

    use_cnn = model_choice.startswith("CNN")

    st.markdown("---")

    # ── Audio settings display ───────────────────────────────────────────────
    st.markdown("### 📊 DSP Settings")
    col_a, col_b = st.columns(2)
    col_a.metric("Sample Rate",   f"{SAMPLE_RATE:,} Hz")
    col_b.metric("Chunk Length",  f"{CHUNK_DURATION} s")

    chunk_samples = PROC.chunk_samples
    st.caption(f"Chunk size: {chunk_samples:,} samples")
    st.caption(f"Mel bands: {PROC.N_MELS}   •   MFCCs: {PROC.N_MFCC}")

    st.markdown("---")

    # ── Model status ─────────────────────────────────────────────────────────
    st.markdown("### 🗂️ Model Files")

    cnn_exists    = os.path.exists(CNN_MODEL_PATH)
    kmeans_exists = os.path.exists(KMEANS_PKL_PATH)

    st.markdown(
        f"{'✅' if cnn_exists    else '❌'} CNN model "
        f"(`{os.path.basename(CNN_MODEL_PATH)}`)"
    )
    st.markdown(
        f"{'✅' if kmeans_exists else '❌'} K-Means bundle "
        f"(`{os.path.basename(KMEANS_PKL_PATH)}`)"
    )

    if not cnn_exists and not kmeans_exists:
        st.warning("No models found. Run `train.py` first.")

    st.markdown("---")

    # ── Registered speakers ──────────────────────────────────────────────────
    st.markdown("### 👥 Registered Speakers")
    for spk in SPEAKERS:
        st.markdown(f"- **{spk.capitalize()}**")


# =============================================================================
# Main content
# =============================================================================
st.markdown("""
<div class="vc-hero">
    <h1>🎙️ VocalCanvas</h1>
    <p>Speaker Identification · CNN &amp; K-Means · DSP-Powered</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# File upload
# =============================================================================
st.markdown("### 📂 Upload Audio")

uploaded = st.file_uploader(
    "Drag and drop or click to browse",
    type=["wav", "mp3", "mp4", "ogg", "flac", "m4a"],
    help="Supported: WAV, MP3, MP4, OGG, FLAC, M4A",
)

if uploaded is None:
    st.info(
        "⬆️  Upload an audio file to get started.  "
        "Files longer than 3 s will be split into chunks automatically."
    )
    st.stop()

# ── Save upload to a temp file ───────────────────────────────────────────────
suffix = os.path.splitext(uploaded.name)[1].lower()

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

# ── Convert to WAV if needed ─────────────────────────────────────────────────
with st.spinner("Converting to WAV…"):
    try:
        wav_path = convert_to_wav(tmp_path) if suffix != ".wav" else tmp_path
    except Exception as e:
        st.error(f"❌ FFmpeg conversion failed: {e}")
        st.stop()

# ── Load audio for visualisation ─────────────────────────────────────────────
audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
duration  = len(audio) / sr
n_chunks  = len(PROC.split_audio(audio))

st.success(
    f"✅  **{uploaded.name}** loaded  "
    f"· Duration: **{duration:.1f} s**  "
    f"· Chunks: **{n_chunks}**"
)

# =============================================================================
# Visualisation — spectrogram + waveform + player
# =============================================================================
st.markdown("---")
st.markdown("### 🔊 Audio Analysis")

vis_col, play_col = st.columns([3, 1])

with vis_col:
    tab_spec, tab_wave = st.tabs(["Mel Spectrogram", "Waveform"])

    with tab_spec:
        with st.spinner("Rendering spectrogram…"):
            fig_spec = plot_mel_spectrogram(audio, sr)
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)

    with tab_wave:
        fig_wave, ax_w = plt.subplots(figsize=(10, 2.5))
        fig_wave.patch.set_facecolor("#1e1e2e")
        ax_w.set_facecolor("#1e1e2e")
        times = np.linspace(0, duration, len(audio))
        ax_w.plot(times, audio, color="#60a5fa", linewidth=0.5, alpha=0.9)
        ax_w.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
        ax_w.set_ylabel("Amplitude", color="#94a3b8", fontsize=9)
        ax_w.tick_params(colors="#94a3b8")
        for spine in ax_w.spines.values():
            spine.set_edgecolor("#2d2d44")
        ax_w.set_title("Waveform", color="#a78bfa", fontsize=11, pad=8)
        fig_wave.tight_layout()
        st.pyplot(fig_wave, use_container_width=True)
        plt.close(fig_wave)

with play_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**▶ Listen**")
    # Re-read the original uploaded bytes for the player
    uploaded.seek(0)
    st.audio(uploaded.read(), format=f"audio/{suffix.lstrip('.')}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Duration",  f"{duration:.1f} s")
    st.metric("Chunks",    str(n_chunks))
    st.metric("Model",     "CNN" if use_cnn else "K-Means")

# =============================================================================
# Inference
# =============================================================================
st.markdown("---")
st.markdown("### 🔍 Speaker Identification")

model_label = "CNN (Deep Learning)" if use_cnn else "K-Means (Statistical Clustering)"

if st.button(
    f"🚀  Identify Speaker  —  {model_label}",
    type="primary",
    use_container_width=True,
):

    # ── Load model ────────────────────────────────────────────────────────
    if use_cnn:
        model = get_cnn_model()
        if model is None:
            st.error(
                f"❌ CNN model not found at `{CNN_MODEL_PATH}`. "
                "Run `train.py` first."
            )
            st.stop()

        with st.spinner("Running CNN inference…"):
            result = _run_cnn(wav_path, model)

    else:
        bundle = get_kmeans_bundle()
        if bundle is None:
            st.error(
                f"❌ K-Means bundle not found at `{KMEANS_PKL_PATH}`. "
                "Run `train.py` first."
            )
            st.stop()

        # Sanity check feature dimension
        if bundle["n_features"] != PROC.n_features:
            st.error(
                f"❌ Feature dimension mismatch: bundle has "
                f"{bundle['n_features']} dims but processor expects "
                f"{PROC.n_features}. Re-run `train.py`."
            )
            st.stop()

        with st.spinner("Running K-Means inference…"):
            result = _run_kmeans(wav_path, bundle)

    # ── No valid chunks ───────────────────────────────────────────────────
    if result is None:
        st.warning(
            "⚠️ No valid (non-silent) chunks detected. "
            "Try a louder or longer recording."
        )
        st.stop()

    # ── Store in session state so result persists on rerun ────────────────
    st.session_state["result"] = result


# =============================================================================
# Results display  (shown whenever session_state has a result)
# =============================================================================
if "result" in st.session_state:

    result = st.session_state["result"]
    winner = result["winner"]

    st.markdown("---")
    st.markdown("### 🏆 Results")

    # ── Winner banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="winner-badge">
        <h2>🎤 {winner.upper()}</h2>
        <p>Predicted speaker · {result['model_type']} model
        · {result['votes'][winner]}/{result['total']} chunks
        · avg confidence {result['avg_conf'][winner]*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────────────────
    metric_cols = st.columns(len(SPEAKERS) + 1)

    metric_cols[0].metric(
        "Total Chunks",
        result["total"],
        help="Non-silent chunks used for voting"
    )

    for i, spk in enumerate(SPEAKERS):
        metric_cols[i + 1].metric(
            spk.capitalize(),
            f"{result['vote_pcts'][spk]:.0f}%",
            f"avg {result['avg_conf'][spk]*100:.1f}% conf",
            delta_color="normal" if spk == winner else "off",
        )

    # ── Bar chart ─────────────────────────────────────────────────────────
    fig_res = plot_results_chart(result)
    st.pyplot(fig_res, use_container_width=True)
    plt.close(fig_res)

    # ── Chunk detail table ────────────────────────────────────────────────
    with st.expander("📋 Chunk-by-chunk breakdown", expanded=False):

        df = pd.DataFrame(result["chunk_details"])

        # Colour the winner rows
        def highlight_winner(row):
            if row["Speaker"] == winner:
                return ["background-color: #1a472a; color: #4ade80"] * len(row)
            elif row["Speaker"] == "—":
                return ["color: #4b5563"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(highlight_winner, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # ── Raw vote data as JSON (collapsible) ───────────────────────────────
    with st.expander("🔬 Raw vote data", expanded=False):
        st.json({
            "winner":     winner,
            "model_type": result["model_type"],
            "votes":      result["votes"],
            "vote_pcts":  {k: round(v, 1) for k, v in result["vote_pcts"].items()},
            "avg_conf":   {k: round(v * 100, 2) for k, v in result["avg_conf"].items()},
        })

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption(
    "VocalCanvas · CNN + K-Means Speaker Identification  "
    "· Powered by librosa, TensorFlow, scikit-learn & Streamlit"
)
