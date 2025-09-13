import time
import io
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

# =====================
# KONFIGURASI FITUR
# =====================
MOTION_FEATURES = [
    "body_roll_deg", "pitch_deg", "yaw_deg", "stroke_rate_spm", "lap_speed_mps"
]

TAPPER_CLS_FEATURES = [
    "distance_cm", "time_to_wall_s", "speed_mps", "pace_s_per_25m"
]

TAPPER_REG_FEATURES = [
    "distance_cm", "speed_mps", "pace_s_per_25m", "th_early_cm",
    "th_urgent_cm", "th_safety_cm", "time_to_wall_s"
]

TS_COL = "timestamp_s"

# =====================
# STATE & HELPERS
# =====================
@st.cache_resource
def load_models():
    """Memuat semua model Joblib dari file lokal."""
    try:
        models = {
            "fatigue": joblib.load("fatigue_model.pkl"),
            "stroke": joblib.load("stroke_model.pkl"),
            "movement_quality": joblib.load("movement_quality_model.pkl"),
            "safe": joblib.load("safe_model.pkl"),
            "threshold": joblib.load("threshold_model.pkl")
        }
        return models
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}. Pastikan semua file .pkl berada di direktori yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

def _init_state():
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("df_motion", None)
    st.session_state.setdefault("df_tapper", None)
    st.session_state.setdefault("history", [])

def _row_to_vector(row: Dict[str, Any], feats) -> np.ndarray:
    vals = []
    for f in feats:
        v = row.get(f, 0.0)
        if pd.isna(v):
            v = 0.0
        vals.append(float(v))
    return np.array(vals, dtype=float).reshape(1, -1)

def _can_start(models):
    # Semua model & kedua CSV harus tersedia
    models_ready = models is not None
    data_ready = (st.session_state.df_motion is not None) and (st.session_state.df_tapper is not None)
    return models_ready and data_ready

def _process_tick(models, rows_per_tick: int):
    dm = st.session_state.df_motion
    dt = st.session_state.df_tapper
    i0 = st.session_state.idx

    if dm is None or dt is None:
        return
    n = min(len(dm), len(dt))
    if i0 >= n:
        st.session_state.running = False
        return

    i1 = min(i0 + rows_per_tick, n)

    # Model
    mdl = models

    for i in range(i0, i1):
        row_m = dm.iloc[i].to_dict()
        row_t = dt.iloc[i].to_dict()

        # --- PREDIKSI MOTION (scikit-learn pipelines dari joblib) ---
        Xm = _row_to_vector(row_m, MOTION_FEATURES)
        stroke_pred = mdl["stroke"].predict(Xm)[0]
        fatigue_pred = mdl["fatigue"].predict(Xm)[0]
        moveq_pred = mdl["movement_quality"].predict(Xm)[0]

        # (opsional) confidence jika model mendukung predict_proba
        def _prob_or_none(model, X):
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                return float(np.max(proba))
            return None

        stroke_prob = _prob_or_none(mdl["stroke"], Xm)
        fatigue_prob = _prob_or_none(mdl["fatigue"], Xm)
        moveq_prob = _prob_or_none(mdl["movement_quality"], Xm)

        # --- PREDIKSI TAPPER ---
        Xc = _row_to_vector(row_t, TAPPER_CLS_FEATURES)
        Xr = _row_to_vector(row_t, TAPPER_REG_FEATURES)

        safe_pred = mdl["safe"].predict(Xc)[0]
        threshold_pred = float(mdl["threshold"].predict(Xr)[0])

        # Jika CSV memiliki ground-truth (opsional), ambil untuk perbandingan grafis
        threshold_true = row_t.get("optimal_threshold_cm", None)
        if pd.isna(threshold_true):
            threshold_true = None

        # Simpan riwayat + kolom prediksi ringkas
        st.session_state.history.append({
            TS_COL: row_m.get(TS_COL, i),

            # Prediksi motion
            "stroke_pred": stroke_pred,
            "stroke_prob": stroke_prob,
            "fatigue_pred": fatigue_pred,
            "fatigue_prob": fatigue_prob,
            "movement_quality_pred": moveq_pred,
            "movement_quality_prob": moveq_prob,

            # Prediksi tapper
            "safe_pred": safe_pred,
            "threshold_pred": threshold_pred,
            "threshold_true": threshold_true,

            # Kolom prediksi yang diminta
            "prediksi_gerakan": moveq_pred,
            "prediksi_fatigue": fatigue_pred,
        })

    st.session_state.idx = i1

# =====================
# UI
# =====================
st.set_page_config(page_title="AI Swim Assistant — Live Dashboard (Joblib)", layout="wide")
_init_state()

st.title("🏊‍♂️ AI Swim Assistant — Live Dashboard")

# Load models outside of the main loop so they are cached
models = load_models()

with st.sidebar:
    st.header("Controls")
    rows_per_tick = st.number_input("Rows per tick", min_value=1, max_value=500, value=5, step=1)
    secs_per_tick = st.number_input("Seconds per tick", min_value=0.2, max_value=10.0, value=1.0, step=0.2)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("▶️ Start", use_container_width=True, disabled=not _can_start(models)):
            st.session_state.running = True
    with b2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.running = False
    with b3:
        if st.button("🔁 Reset", use_container_width=True):
            st.session_state.running = False
            st.session_state.idx = 0
            st.session_state.history = []
    
    motion_file = "ai_motion_sample_session.csv"
    tapper_file = "ai_tapper_sample_session.csv"

    if motion_file is not None:
        st.session_state.df_motion = pd.read_csv(motion_file)
    if tapper_file is not None:
        st.session_state.df_tapper = pd.read_csv(tapper_file)

# Live Status
status = st.container()
chart_box = st.container()
table_box = st.container()

# ============ RUNTIME ============
if st.session_state.running and models:
    _process_tick(models, int(rows_per_tick))

# Render selalu agar visual langsung muncul saat Start
hist_df = pd.DataFrame(st.session_state.history)

with status:
    st.subheader("🔴 Live Status")
    if not hist_df.empty:
        last = hist_df.iloc[-1]
        colA, colB, colC = st.columns(3)
        colA.markdown(f"**Stroke (pred):** {last['stroke_pred']}")
        colB.markdown(f"**Movement (pred):** {last['prediksi_gerakan']}")
        colC.markdown(f"**Fatigue (pred):** {last['prediksi_fatigue']}")
    else:
        if models is None:
            st.warning("Failed to load model. Please double check your .pkl file.")
        else:
            st.info("Click Start to Begin")

with chart_box:
    st.subheader("Threshold Pred")
    if not hist_df.empty:
        plot_df = hist_df[[TS_COL, "threshold_pred", "threshold_true"]].tail(200)

        # Buat grafik Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df[TS_COL], y=plot_df["threshold_pred"],
                                 mode='lines', name='Threshold Prediksi'))
        fig.add_trace(go.Scatter(x=plot_df[TS_COL], y=plot_df["threshold_true"],
                                 mode='lines', name='Threshold True'))

        # Atur rentang sumbu Y dengan minimum 60
        fig.update_yaxes(range=[min(80, plot_df["threshold_pred"].min(), plot_df["threshold_true"].min()),
                                max(plot_df["threshold_pred"].max(), plot_df["threshold_true"].max())])
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Waiting for data...")

with table_box:
    st.subheader("Predictions (tail)")
    if not hist_df.empty:
        cols = [TS_COL, "stroke_pred", "prediksi_gerakan", "prediksi_fatigue",
                "safe_pred", "threshold_pred"]
        cols = [c for c in cols if c in hist_df.columns]
        st.dataframe(hist_df[cols].tail(25), use_container_width=True)
        st.download_button(
            "⬇️ Download predictions (CSV)",
            hist_df.to_csv(index=False).encode("utf-8"),
            "predictions_stream.csv",
            "text/csv"
        )
    else:
        st.caption("There is no history of predictions yet.")

# Auto loop (paling bawah: proses → render → sleep → rerun)
if st.session_state.running:
    time.sleep(float(secs_per_tick))
    st.rerun()

