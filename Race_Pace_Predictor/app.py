import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Race Pace Predictor",
    page_icon="🏃",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
FC_REPOS     = 45
FC_MAX       = 185
RIEGEL_B     = 1.06
DIST_MAX_ML  = 68.5
DENI_MAX_ML  = 1864

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(Path("data/models/best_model.pkl"))
    with open(Path("data/models/features.json")) as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_data():
    df_all = pd.read_csv(
        Path("data/processed/all_activities_clean.csv"), parse_dates=["date"]
    ).sort_values("date").reset_index(drop=True)

    df_run = pd.read_csv(
        Path("data/processed/running_dataset.csv"), parse_dates=["date"]
    )

    # Compute TRIMP if missing
    def compute_trimp(duration_min, avg_hr):
        if pd.isna(avg_hr) or duration_min <= 0:
            return 0.0
        hr_norm = max(0, min((avg_hr - FC_REPOS) / max(FC_MAX - FC_REPOS, 1), 1))
        return duration_min * hr_norm * 0.64 * np.exp(1.92 * hr_norm)

    if "trimp" not in df_all.columns:
        df_all["trimp"] = df_all.apply(
            lambda r: compute_trimp(r["duration_min"], r["avg_hr"]), axis=1
        )

    trimp_daily  = df_all[["date", "trimp"]].set_index("date")["trimp"].resample("D").sum()
    atl          = trimp_daily.rolling(7,  min_periods=1).mean()
    ctl          = trimp_daily.rolling(42, min_periods=1).mean()
    tsb          = ctl - atl
    load_7d      = trimp_daily.rolling(7,  min_periods=1).sum()
    load_28d     = trimp_daily.rolling(28, min_periods=1).sum()

    training_load_med = df_run["training_load"].median()
    vo2max_last       = float(df_run["VO2max"].dropna().iloc[-1])

    return df_all, df_run, atl, ctl, tsb, load_7d, load_28d, training_load_med, vo2max_last

# ── Helper functions ──────────────────────────────────────────────────────────
def get_charge(target_date, series):
    ts   = pd.Timestamp(target_date)
    prev = series[series.index < ts]
    return float(prev.iloc[-1]) if len(prev) > 0 else 0.0

def get_rest_days(target_date, df_all):
    ts   = pd.Timestamp(target_date)
    prev = df_all[df_all["date"] < ts]
    return 0 if len(prev) == 0 else max(0, (ts - prev["date"].max()).days)

def get_training_load(target_date, df_run, training_load_med):
    ts   = pd.Timestamp(target_date)
    prev = df_run[df_run["date"] < ts]
    if len(prev) == 0:
        return float(training_load_med)
    return float(prev["training_load"].iloc[-1])

def fmt_time(seconds):
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}"

def fmt_pace(seconds_per_km):
    m = int(seconds_per_km) // 60
    s = int(seconds_per_km) % 60
    return f"{m}:{s:02d} /km"

def build_features(distance_km, elevation_m, is_trail, d,
                   atl, ctl, tsb, load_7d, load_28d,
                   df_all, df_run, training_load_med, vo2max_last):
    deq = distance_km + elevation_m / 100
    return {
        "distance_km"      : distance_km,
        "eq_distance_km"   : deq,
        "elevation_m"      : float(elevation_m),
        "elevation_per_km" : elevation_m / distance_km,
        "is_trail"         : float(is_trail),
        "month"            : float(d.month),
        "riegel_factor"    : distance_km ** RIEGEL_B,
        "difficulty_index" : deq ** RIEGEL_B,
        "log_eq_distance"  : np.log(deq),
        "relative_elevation": elevation_m / distance_km,
        "ATL"              : get_charge(d, atl),
        "CTL"              : get_charge(d, ctl),
        "TSB"              : get_charge(d, tsb),
        "load_7d"          : get_charge(d, load_7d),
        "load_28d"         : get_charge(d, load_28d),
        "rest_days"        : float(get_rest_days(d, df_all)),
        "VO2max"           : vo2max_last,
        "training_load"    : get_training_load(d, df_run, training_load_med),
    }

def predict_ml(distance_km, elevation_m, is_trail, d, model, meta,
               atl, ctl, tsb, load_7d, load_28d,
               df_all, df_run, training_load_med, vo2max_last):
    features = build_features(distance_km, elevation_m, is_trail, d,
                               atl, ctl, tsb, load_7d, load_28d,
                               df_all, df_run, training_load_med, vo2max_last)
    X = pd.DataFrame([features])[meta["features"]]
    return float(model.predict(X)[0]), features

def predict_riegel(distance_km, elevation_m, is_trail, d, model, meta,
                   atl, ctl, tsb, load_7d, load_28d,
                   df_all, df_run, training_load_med, vo2max_last):
    deni_ref   = min(elevation_m, DENI_MAX_ML)
    deq_target = distance_km + elevation_m / 100
    deq_ref    = DIST_MAX_ML + deni_ref / 100
    temps_ref, features = predict_ml(
        DIST_MAX_ML, deni_ref, is_trail, d, model, meta,
        atl, ctl, tsb, load_7d, load_28d,
        df_all, df_run, training_load_med, vo2max_last
    )
    return temps_ref * (deq_target / deq_ref) ** RIEGEL_B, features

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏃 Race Pace Predictor")
st.caption("Predict your finish time and average pace based on your Garmin training history.")

# Load everything
try:
    model, meta = load_model()
    df_all, df_run, atl, ctl, tsb, load_7d, load_28d, training_load_med, vo2max_last = load_data()
    data_ok = True
except Exception as e:
    st.error(f"⚠️ Could not load model or data: {e}")
    st.info("Make sure `data/models/best_model.pkl`, `data/models/features.json`, "
            "`data/processed/all_activities_clean.csv` and "
            "`data/processed/running_dataset.csv` exist in the working directory.")
    data_ok = False

if data_ok:
    # ── Inputs ────────────────────────────────────────────────────────────────
    st.subheader("📋 Race parameters")

    col1, col2 = st.columns(2)
    with col1:
        distance_km  = st.number_input("Distance (km)", min_value=1.0, max_value=300.0,
                                        value=10.0, step=0.5)
        race_type    = st.selectbox("Race type", ["Road", "Trail"])
    with col2:
        elevation_m  = st.number_input("Elevation gain D+ (m)", min_value=0, max_value=15000,
                                        value=100, step=50)
        race_date    = st.date_input("Target date", value=date.today(),
                                      min_value=date(2022, 1, 1))

    is_trail = 1 if race_type == "Trail" else 0
    d        = pd.Timestamp(race_date)

    # ── Predict button ────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔮 Predict", use_container_width=True, type="primary"):

        extrapolation = (distance_km > DIST_MAX_ML) or (elevation_m > DENI_MAX_ML)

        if extrapolation:
            pred_s, features = predict_riegel(
                distance_km, elevation_m, is_trail, d, model, meta,
                atl, ctl, tsb, load_7d, load_28d,
                df_all, df_run, training_load_med, vo2max_last
            )
            method = "Riegel (extrapolation)"
        else:
            pred_s, features = predict_ml(
                distance_km, elevation_m, is_trail, d, model, meta,
                atl, ctl, tsb, load_7d, load_28d,
                df_all, df_run, training_load_med, vo2max_last
            )
            method = f"ML — {meta['model']}"

        pace_s = pred_s / distance_km

        # ── Results ───────────────────────────────────────────────────────────
        st.subheader("🎯 Prediction")

        if extrapolation:
            st.warning(
                f"⚠️ **Extrapolation** — this race exceeds the training distribution "
                f"(max {DIST_MAX_ML:.0f} km / {DENI_MAX_ML:.0f} m D+). "
                f"Riegel's formula is used instead of the ML model."
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Finish time",   fmt_time(pred_s))
        c2.metric("Average pace",  fmt_pace(pace_s))
        c3.metric("Method",        method)

        # ── Fitness state ─────────────────────────────────────────────────────
        st.subheader("💪 Fitness state on race day")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("ATL (fatigue)",  f"{features['ATL']:.1f}")
        f2.metric("CTL (fitness)",  f"{features['CTL']:.1f}")
        f3.metric("TSB (freshness)", f"{features['TSB']:.1f}",
                  delta="Fresh ✅" if features['TSB'] > 0 else "Fatigued ⚠️",
                  delta_color="normal" if features['TSB'] > 0 else "inverse")
        f4.metric("VO2max",         f"{features['VO2max']:.1f}")

        # TSB interpretation
        tsb_val = features['TSB']
        if tsb_val > 10:
            st.success("You are well rested — good conditions for a peak performance.")
        elif tsb_val > -10:
            st.info("Moderate fatigue — expect a normal effort.")
        elif tsb_val > -30:
            st.warning("Mild accumulated fatigue — the prediction accounts for this.")
        else:
            st.error(f"High fatigue (TSB = {tsb_val:.1f}) — performance may be impaired.")

        # ── Splits table ──────────────────────────────────────────────────────
        st.subheader("📊 Estimated split times")
        checkpoints = []
        distances   = [d for d in [5, 10, 15, 20, 21.1, 30, 40, 42.2, 50, 60, 80, 100]
                       if d <= distance_km]
        if distance_km not in checkpoints:
            distances.append(distance_km)

        rows = []
        for cp in sorted(set(distances)):
            t = pred_s * (cp / distance_km)   # linear split (simplified)
            rows.append({"Distance (km)": f"{cp:.1f}", "Split time": fmt_time(t),
                         "Pace": fmt_pace(t / cp)})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Model: **{meta['model']}** · "
        f"Training range: up to {DIST_MAX_ML:.0f} km / {DENI_MAX_ML:.0f} m D+ · "
        f"Last VO2max: {vo2max_last:.1f}"
    )
