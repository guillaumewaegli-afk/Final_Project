# 🏃 Run Pace Predictor

A personalized race pace predictor built from individual Garmin Connect training history. Unlike commercial sports watches, this tool predicts finish times for **any distance and elevation profile**, not just standard race distances.

---

## 🎯 Objective

Predict a runner's finish time and average pace for an upcoming race based on:
- Race parameters (distance, elevation gain, race type, date)
- Personal training history from Garmin Connect
- Current physiological state (fatigue, fitness, freshness)

---

## ⚙️ Pipeline

```
Garmin Connect API → Data Cleaning → Feature Engineering → ML Modelling → Streamlit App
```

1. **Data acquisition** — incremental sync via the `garminconnect` library
2. **Feature engineering** — TRIMP-based training load (ATL, CTL, TSB) + Riegel fatigue features
3. **Modelling** — Ridge Regression, Random Forest, XGBoost (80/20 temporal split)
4. **Prediction interface** — Streamlit web app with race inputs and split time table

---

## 📁 Project Structure

```
├── data/
│   ├── activities.json          # Raw Garmin activities
│   ├── all_activities_clean.csv # Cleaned full history
│   └── running_dataset.csv      # Final modelling dataset (18 features)
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── src/
│   ├── data_sync.py             # Garmin API sync
│   ├── feature_engineering.py   # TRIMP + Riegel features
│   ├── train.py                 # Model training & evaluation
│   └── predict.py               # predict_course() function
├── app.py                       # Streamlit interface
├── model_xgboost.pkl            # Saved best model
└── README.md
```

---
