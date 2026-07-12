import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter
from scipy.fft import rfft, rfftfreq

# ── Path setup ────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loader import load_cwru_mat
from filters import full_preprocessing_pipeline, segment_signal
from features import extract_features, get_feature_names

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Bearing Fault Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f11; }
    .stApp { background-color: #0f0f11; color: #e8e8f0; }

    .metric-card {
        background: #17171a;
        border: 1px solid #2a2a30;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .metric-label {
        font-size: 13px;
        color: #7a7a8e;
    }

    .result-card {
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
        text-align: center;
    }
    .result-normal  { background: rgba(79,190,151,.12); border: 2px solid #4fbe97; }
    .result-fault   { background: rgba(255,107,122,.12); border: 2px solid #ff6b7a; }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-sub {
        font-size: 14px;
        color: #aaaacc;
    }

    .info-box {
        background: #12151c;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 13px;
        color: #7a7a8e;
        margin: 10px 0;
    }

    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #1e2330;
        font-size: 12px;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d0f14;
        border-right: 1px solid #1e2330;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #3d6aff, #4fbe97);
    }

    h1, h2, h3 { color: #e8e8f0 !important; }

    .upload-section {
        background: #12151c;
        border: 2px dashed #2a2a30;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────
CLASS_INFO = {
    0: {
        'name'       : 'Normal',
        'icon'       : '✅',
        'color'      : '#4fbe97',
        'description': 'Bearing is operating normally. No fault detected.',
        'action'     : 'No maintenance required. Continue regular monitoring.',
        'severity'   : 'Healthy',
    },
    1: {
        'name'       : 'Inner Race Fault',
        'icon'       : '⚠️',
        'color'      : '#ff6b7a',
        'description': 'Fault detected on the inner raceway of the bearing.',
        'action'     : 'Schedule maintenance inspection within 48–72 hours.',
        'severity'   : 'High',
    },
    2: {
        'name'       : 'Ball Fault',
        'icon'       : '⚠️',
        'color'      : '#f0a147',
        'description': 'Fault detected on a rolling element (ball) of the bearing.',
        'action'     : 'Monitor closely. Plan maintenance within 1 week.',
        'severity'   : 'Medium',
    },
    3: {
        'name'       : 'Outer Race Fault',
        'icon'       : '🚨',
        'color'      : '#ff6b7a',
        'description': 'Fault detected on the outer raceway of the bearing.',
        'action'     : 'Immediate inspection recommended. Risk of rapid deterioration.',
        'severity'   : 'Critical',
    },
}

FS          = 12000
WINDOW_SIZE = 1024
OVERLAP     = 0.5

# ── Load pipeline ─────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join(
        os.path.dirname(__file__), 'models', 'inference_pipeline.pkl'
    )
    if not os.path.exists(pipeline_path):
        return None
    return joblib.load(pipeline_path)

# ── Feature names in EXACT training order ─────────────────
# This must match the order used in notebook 03 exactly.
# get_feature_names() returns them in the same order as features.py
FEATURE_NAMES = get_feature_names()

# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Bearing Fault Detector")
    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This system analyzes vibration signals from rotating machinery
    and classifies the bearing health condition using machine learning.

    **Classes detected:**
    - ✅ Normal
    - ⚠️ Inner Race Fault
    - ⚠️ Ball Fault
    - 🚨 Outer Race Fault
    """)

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("""
    - **Model:** Random Forest (300 trees)
    - **Features:** 16 (9 time + 7 frequency domain)
    - **CV F1-Score:** 0.9977
    - **Dataset:** CWRU Bearing Dataset
    - **Sampling rate:** 12,000 Hz
    - **Window size:** 1,024 samples (~85ms)
    """)

    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
    [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)

    *Cited in 3,000+ research papers*
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Advanced Settings")
    show_features  = st.checkbox("Show extracted features", value=False)
    show_fft       = st.checkbox("Show FFT spectrum",       value=True)
    show_signal    = st.checkbox("Show raw signal",         value=True)
    n_signal_ms    = st.slider("Signal preview (ms)", 50, 500, 100, step=50)

# ─────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────
st.markdown("# ⚙️ Industrial Bearing Fault Detection")
st.markdown(
    "Upload a CWRU `.mat` vibration signal file to classify bearing health condition "
    "using a trained Random Forest model with signal processing and feature engineering."
)

# ── Load model ────────────────────────────────────────────
pipeline = load_pipeline()
if pipeline is None:
    st.error(
        "⚠️ Model file not found. Make sure `models/inference_pipeline.pkl` "
        "exists in the project directory."
    )
    st.stop()

st.success("✓ Model loaded successfully")

st.markdown("---")

# ── File upload ───────────────────────────────────────────
st.markdown("### 📂 Upload Vibration Signal")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose a CWRU .mat file",
        type=["mat"],
        help="Upload a .mat file from the CWRU Bearing Dataset. "
             "Files should contain a drive-end accelerometer signal sampled at 12,000 Hz."
    )

with col_info:
    st.markdown("""
    <div class="info-box">
    <strong>Expected file format</strong><br><br>
    CWRU .mat files containing a key ending in <code>_DE_time</code>
    (drive-end accelerometer signal).<br><br>
    Example filenames:<br>
    • normal_1.mat<br>
    • IR_007_1.mat<br>
    • ball_014_1.mat<br>
    • OR_021_1.mat
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  PROCESSING
# ─────────────────────────────────────────────────────────
if uploaded_file is not None:

    # ── Save temp file ────────────────────────────────────
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # ── Progress bar ──────────────────────────────────
        progress = st.progress(0, text="Loading signal...")

        # Step 1: Load signal
        raw_signal = load_cwru_mat(tmp_path)
        progress.progress(20, text="Signal loaded — applying preprocessing...")

        # Step 2: Preprocessing pipeline
        # DC removal → Butterworth low-pass (5000 Hz) → no normalization
        clean_signal = full_preprocessing_pipeline(raw_signal, fs=FS)
        progress.progress(40, text="Filtering complete — segmenting into windows...")

        # Step 3: Windowing
        windows = segment_signal(
            clean_signal,
            window_size=WINDOW_SIZE,
            overlap=OVERLAP
        )
        progress.progress(60, text=f"Created {len(windows)} windows — extracting features...")

        # Step 4: Feature extraction
        # CRITICAL: features must be extracted in the same order as training
        # get_feature_names() returns the canonical order from features.py
        # which matches exactly how notebook 03 built the DataFrame
        rows = []
        for w in windows:
            feats = extract_features(w, fs=FS)
            rows.append(feats)

        # Build DataFrame with columns in exact training order
        df_features = pd.DataFrame(rows)[FEATURE_NAMES]

        progress.progress(80, text="Running model inference...")

        # Step 5: Predict using pipeline
        # pipeline = scaler → (optional selector) → classifier
        # Feed raw unscaled features — pipeline handles scaling internally
        X_input = df_features.values
        
        
        
        preds   = pipeline.predict(X_input)
        probas  = pipeline.predict_proba(X_input)

        progress.progress(100, text="Done!")
        progress.empty()

        # ── Aggregate predictions ─────────────────────────
        vote_counts  = Counter(preds)
        final_label  = vote_counts.most_common(1)[0][0]
        confidence   = probas[:, final_label].mean()
        info         = CLASS_INFO[final_label]

        os.unlink(tmp_path)

    except Exception as e:
        os.unlink(tmp_path)
        st.error(f"Error processing file: {str(e)}")
        st.stop()

    # ─────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    card_class = "result-normal" if final_label == 0 else "result-fault"
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-title" style="color:{info['color']}">
            {info['icon']} {info['name']}
        </div>
        <div class="result-sub">{info['description']}</div>
        <br>
        <div style="color:{info['color']};font-weight:500;">
            🔧 {info['action']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{info['color']}">
                {confidence*100:.1f}%
            </div>
            <div class="metric-label">Confidence</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#3d6aff">
                {len(windows):,}
            </div>
            <div class="metric-label">Windows Analyzed</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        duration = len(raw_signal) / FS
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#f0a147">
                {duration:.1f}s
            </div>
            <div class="metric-label">Signal Duration</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        severity_colors = {
            'Healthy':'#4fbe97','Medium':'#f0a147',
            'High':'#ff6b7a','Critical':'#ff6b7a'
        }
        sev_color = severity_colors.get(info['severity'], '#7a7a8e')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{sev_color}">
                {info['severity']}
            </div>
            <div class="metric-label">Severity Level</div>
        </div>""", unsafe_allow_html=True)

    # ── Per-class probability breakdown ───────────────────
    st.markdown("#### Class Probability Breakdown")
    for label, cls_info in CLASS_INFO.items():
        prob = probas[:, label].mean()
        col_a, col_b, col_c = st.columns([2, 5, 1])
        with col_a:
            st.markdown(
                f"<span style='color:{cls_info['color']}'>"
                f"{cls_info['icon']} {cls_info['name']}</span>",
                unsafe_allow_html=True
            )
        with col_b:
            st.progress(float(prob))
        with col_c:
            st.markdown(f"`{prob*100:.1f}%`")

    # ── Window vote distribution ──────────────────────────
    st.markdown("#### Window-Level Vote Distribution")
    vote_cols = st.columns(len(CLASS_INFO))
    for i, (label, cls_info) in enumerate(CLASS_INFO.items()):
        count = vote_counts.get(label, 0)
        pct   = count / len(preds) * 100
        with vote_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:20px;margin-bottom:4px;">
                    {cls_info['icon']}
                </div>
                <div style="font-size:18px;font-weight:600;
                            color:{cls_info['color']}">
                    {count}
                </div>
                <div style="font-size:11px;color:#7a7a8e;">
                    {cls_info['name']}<br>{pct:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    #  VISUALIZATIONS
    # ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Signal Analysis")

    n_plot = int(FS * n_signal_ms / 1000)
    t_ms   = np.arange(n_plot) / FS * 1000

    plot_col1, plot_col2 = st.columns(2)

    # ── Raw signal plot ───────────────────────────────────
    if show_signal:
        with plot_col1:
            st.markdown("**Raw Vibration Signal**")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0f0f11')
            ax.set_facecolor('#0f0f11')

            ax.plot(t_ms, raw_signal[:n_plot],
                    color=info['color'], linewidth=0.8, alpha=0.9)
            ax.set_xlabel('Time (ms)', color='#7a7a8e', fontsize=9)
            ax.set_ylabel('Amplitude', color='#7a7a8e', fontsize=9)
            ax.set_title(f'Raw Signal — {n_signal_ms}ms preview',
                         color='#e8e8f0', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#7a7a8e', labelsize=8)
            ax.spines[['top','right']].set_visible(False)
            for sp in ['bottom','left']:
                ax.spines[sp].set_color('#2a2a30')
            ax.grid(True, color='#1e2330', linewidth=0.5, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── FFT spectrum ──────────────────────────────────────
    if show_fft:
        with plot_col2:
            st.markdown("**Frequency Spectrum (FFT)**")
            n_fft    = 4096
            seg      = clean_signal[:n_fft] * np.hanning(n_fft)
            fft_mag  = np.abs(rfft(seg))
            freqs    = rfftfreq(n_fft, d=1.0/FS)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor('#0f0f11')
            ax.set_facecolor('#0f0f11')

            ax.plot(freqs, fft_mag,
                    color=info['color'], linewidth=0.7, alpha=0.85)
            ax.axvspan(100, 4000, color=info['color'],
                       alpha=0.06, label='Fault band (100–4000 Hz)')
            ax.axvline(4000, color='#7a7a8e',
                       linewidth=0.8, linestyle='--', alpha=0.5)

            dom_freq = freqs[np.argmax(fft_mag)]
            dom_mag  = np.max(fft_mag)
            ax.annotate(f'{dom_freq:.0f} Hz',
                        xy=(dom_freq, dom_mag),
                        xytext=(dom_freq + 200, dom_mag * 0.85),
                        fontsize=8, color='#ffffff',
                        arrowprops=dict(
                            arrowstyle='->', color='#ffffff', lw=0.8
                        ))

            ax.set_xlabel('Frequency (Hz)', color='#7a7a8e', fontsize=9)
            ax.set_ylabel('Magnitude', color='#7a7a8e', fontsize=9)
            ax.set_title('FFT Spectrum — Filtered Signal',
                         color='#e8e8f0', fontsize=10, fontweight='bold')
            ax.set_xlim(0, FS/2)
            ax.tick_params(colors='#7a7a8e', labelsize=8)
            ax.spines[['top','right']].set_visible(False)
            for sp in ['bottom','left']:
                ax.spines[sp].set_color('#2a2a30')
            ax.grid(True, color='#1e2330', linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=8, framealpha=0.2,
                      labelcolor='#aaaacc')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Feature table ─────────────────────────────────────
    if show_features:
        st.markdown("---")
        st.markdown("### 🔬 Extracted Features (Mean across all windows)")
        st.markdown(
            "*These are the 16 features fed to the model in the exact same "
            "order used during training.*"
        )

        feat_means = df_features.mean()
        feat_stds  = df_features.std()

        feat_display = pd.DataFrame({
            'Feature'    : FEATURE_NAMES,
            'Mean Value' : feat_means.values.round(4),
            'Std Dev'    : feat_stds.values.round(4),
            'Domain'     : [
                'Time','Time','Time','Time','Time',
                'Time','Time','Time','Time',
                'Frequency','Frequency','Frequency',
                'Frequency','Frequency','Frequency','Frequency'
            ]
        })

        st.dataframe(
            feat_display.style.background_gradient(
                subset=['Mean Value'], cmap='Blues'
            ),
            use_container_width=True,
            hide_index=True
        )

    # ─────────────────────────────────────────────────────
    #  PROCESSING SUMMARY
    # ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔍 Processing Pipeline Summary"):
        st.markdown(f"""
        | Step | Detail |
        |---|---|
        | **File uploaded** | `{uploaded_file.name}` |
        | **Signal length** | `{len(raw_signal):,}` samples |
        | **Duration** | `{len(raw_signal)/FS:.2f}` seconds at {FS:,} Hz |
        | **DC removal** | Subtracted signal mean |
        | **Filter** | Butterworth low-pass — 5000 Hz cutoff, order 5 |
        | **Window size** | {WINDOW_SIZE} samples ({WINDOW_SIZE/FS*1000:.1f} ms) |
        | **Overlap** | {int(OVERLAP*100)}% → step = {int(WINDOW_SIZE*(1-OVERLAP))} samples |
        | **Windows created** | {len(windows):,} |
        | **Features extracted** | {len(FEATURE_NAMES)} per window ({FEATURE_NAMES}) |
        | **Scaling** | StandardScaler (fitted on training data) |
        | **Model** | Random Forest — 300 trees |
        | **Final prediction** | Majority vote across {len(preds):,} window predictions |
        """)

# ── Empty state ───────────────────────────────────────────
else:
    st.markdown("""
    <div class="info-box" style="text-align:center; padding:40px;">
        <div style="font-size:48px; margin-bottom:16px;">📂</div>
        <div style="font-size:16px; color:#e8e8f0; margin-bottom:8px;">
            Upload a .mat file to get started
        </div>
        <div style="font-size:13px; color:#7a7a8e;">
            Download sample files from the
            <a href="https://engineering.case.edu/bearingdatacenter/download-data-file"
               target="_blank" style="color:#3d6aff;">
               CWRU Bearing Data Center
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How It Works")
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "Upload a .mat vibration signal file from CWRU dataset"),
        ("2️⃣", "Filter", "Butterworth low-pass filter removes noise above 5000 Hz"),
        ("3️⃣", "Features", "16 time & frequency domain features extracted per window"),
        ("4️⃣", "Predict", "Random Forest classifies bearing health condition"),
    ]
    for col, (icon, title, desc) in zip([col1,col2,col3,col4], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:28px">{icon}</div>
                <div style="font-size:14px;font-weight:600;
                            color:#e8e8f0;margin:8px 0">{title}</div>
                <div style="font-size:12px;color:#7a7a8e">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#3a3a46;font-size:12px;'>"
    "Industrial Predictive Maintenance · CWRU Bearing Dataset · "
    "Random Forest · CV F1 = 0.9977"
    "</div>",
    unsafe_allow_html=True
)
