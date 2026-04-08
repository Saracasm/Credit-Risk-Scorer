# Credit Risk Scorer

End-to-end classical ML system: loan default probability (Give Me Some Credit), SHAP explanations, Fairlearn age-group audit, Streamlit UI, Docker for Hugging Face Spaces.

## Tech stack

- **Data:** pandas, NumPy  
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE), Optuna  
- **Explainability / fairness:** SHAP, Fairlearn  
- **Tracking (optional):** MLflow (`requirements-mlflow.txt`)  
- **App:** Streamlit  
- **Deploy:** Docker (Python 3.11-slim), port **7860**

## Dataset

[Kaggle – Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)  
Place **`cs-training.csv`** in `data/` (do not commit; it is gitignored).

## Run locally

1. Create a virtual environment (Python 3.10–3.12 recommended for smoothest installs):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-mlflow.txt
   ```

   On some Windows setups, run `pip install setuptools wheel` before MLflow if `pyarrow` fails to build.

3. Put `data/cs-training.csv` in place, then run the pipeline in order:

   ```bash
   python notebooks/01_eda.py
   python src/preprocess.py
   python src/feature_engineering.py
   python src/model_selection.py
   python src/train.py
   python src/evaluate.py
   python src/fairness.py
   ```

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

5. Tests:

   ```bash
   pytest tests/ -v
   ```

## Docker

Build and run (from project root; **`models/xgb_pipeline.pkl` must exist** in the build context):

```bash
docker build -t credit-risk-scorer .
docker run -p 7860:7860 credit-risk-scorer
```

Open `http://localhost:7860`.


## Model card

| Item | Detail |
|------|--------|
| **Algorithm** | XGBoost inside a sklearn `Pipeline` with `StandardScaler` |
| **Training data** | Give Me Some Credit (cleaned); stratified 80/20 split |
| **Primary metric** | ROC-AUC (class imbalance handled with `scale_pos_weight`; SMOTE in model-selection CV) |
| **Explainability** | SHAP TreeExplainer — global bar + beeswarm plots, per-sample waterfall plots (saved as `shap_force_*.png`; SHAP waterfall is the matplotlib-native equivalent of force plots) |
| **Fairness** | Fairlearn `MetricFrame` by age band (18–30, 31–45, 46–60, 60+); see `data/fairness_report.csv` |
| **Limitations** | Synthetic demo scores if trained on small samples; not for real lending; no PII in public dataset |
| **Monitoring** | `src/monitor.py` logs predictions to `data/prediction_log.csv` and flags mean drift vs training |

## Project layout

- `src/preprocess.py` – cleaning  
- `src/feature_engineering.py` – 6 engineered features, splits, scaler reference  
- `src/model_selection.py` – LR / RF / XGB CV + optional MLflow  
- `src/train.py` – Optuna-tuned XGBoost pipeline saved to `models/xgb_pipeline.pkl`  
- `src/evaluate.py` – ROC/PR/confusion/threshold + SHAP PNGs under `data/plots/`  
- `src/fairness.py` – fairness CSV  
- `src/monitor.py` – prediction log + drift helpers  
- `src/inference.py` – single-row inference helpers (`load_pipeline`, `build_feature_row`, `predict_default`, `shap_values_row`) shared by `app.py` and the test suite  
- `app.py` – Streamlit UI  

## License / data

Training data is subject to Kaggle competition terms. This repo is a learning/portfolio project.
