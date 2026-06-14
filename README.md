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

## 🧠 Features (18 inputs)

| Category | Features |
|---|---|
| Race geometry | `distance_km`, `eq_distance_km`, `elevation_m`, `elevation_per_km`, `is_trail` |
| Riegel fatigue | `riegel_factor`, `difficulty_index`, `log_eq_distance`, `relative_elevation` |
| Training load | `ATL`, `CTL`, `TSB`, `load_7d`, `load_28d`, `training_load` |
| Physiological | `VO2max`, `rest_days`, `month` |

---

## 📊 Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression | 6:19 | 7:31 | 0.916 |
| Random Forest | 3:55 | 5:59 | 0.947 |
| **XGBoost** | **3:20** | **4:15** | **0.973** |

XGBoost is used for all predictions. For distances > 68.5 km or elevation > 1,864 m, the model falls back to **Riegel's extrapolation formula**.

---

## 🖥️ Streamlit App

```bash
streamlit run app.py
```

The app takes four inputs — distance, elevation, race type, and target date — and returns:
- Predicted finish time and average pace
- Fitness state on race day (ATL, CTL, TSB, VO2max)
- Estimated split times at standard checkpoints

---

## 🛠️ Stack

- **Python** — pandas, scikit-learn, xgboost
- **Data source** — Garmin Connect API (`garminconnect`)
- **App** — Streamlit
- **Training methodology** — TRIMP (Banister, 1991), Riegel's formula

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

## ⚠️ Limitations

- Best performance on road runs between 5–40 km
- Riegel fallback used beyond training distribution (> 68.5 km / > 1,864 m D+)
- Slight overestimation bias on 10–15 km runs (~1.9 min)
- Single-athlete model — not generalisable without retraining

---

## 🔭 Future Work

- Enrich dataset with more long-distance trail runs
- Add external signals (temperature, altitude, sleep quality)
- Hyperparameter search + cross-validation
- Automatic Garmin sync + training load dashboard in the app

---

## 📚 References

- Banister, E. (1991). TRIMP methodology
- Riegel, P. (1977). Athletic records and human endurance
- Imbach, F. et al. (2022). Training load responses modelling in elite sports. *Nature Portfolio*
