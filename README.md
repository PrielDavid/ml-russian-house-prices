
# 🇷🇺 Russian Housing Price Prediction | Sberbank Kaggle Challenge 🏠

This repository contains my end‑to‑end workflow for the **Sberbank Russian Housing Market** competition on Kaggle, where the task is to predict apartment sale prices in Russia’s volatile economy. The repo walks through **data ingestion → EDA → feature engineering → model training & ensembling → submission generation**.

---

## 📑 Competition Snapshot
| | |
|---|---|
| **Host** | [Kaggle – Sberbank Russian Housing Market](https://www.kaggle.com/competitions/sberbank-russian-housing-market) |
| **Goal** | Forecast the sale price (₽) for each apartment in the test set |
| **Metric** | Root Mean Squared Log Error (**RMSLE**) |
| **Train rows** | 30 k properties (2011‑2015) |
| **Features** | > 390 location, structural & macro‑economic variables |

---

## 🗂️ Repository Structure
```text
.
├── data/                    # Small test/sample data (original CSVs ignored)
├── Final_Work_ML_Project.ipynb
├── Final_Work_ML_Project.pdf
├── final_submission.csv     # Best public‑LB submission
├── requirements.txt
├── LICENSE.txt
└── README.md   ← this file
```
> **Note:** Large datasets & model binaries are excluded via `.gitignore`.

---

## 🔍 Key EDA Insights
* **Price distribution & outliers** – המחיר (₽) מפוזר בסקיו חיובי חריף; קיימים מקרים קיצוניים (top‑1 %) שמוטים במיוחד ויוחלפו בלוג‑טרנספורמציה או יוגבלו בקאפ.
* **Spatial heterogeneity** – ניתוח “Average price / district” מראה פערי x3‑x4 בין מרכז מוסקבה לפרברים; מכאן שנדרש encoding מרחבי מפורט (District / Sub‑area).
* **Data‑quality anomalies**  
  * `life_sq > full_sq` או `life_sq < 9 m²`  →  סומנו NaN  
  * `kitch_sq > 0.6 × life_sq`  →  NaN  
  * ערכי `floor`, `max_floor` ו‑`build_year` בלתי‑סבירים הוסרו (לדוגמה build_year < 1500 או > 2025).
* **Temporal pattern** – ספירת עסקאות חודשייות מצביעה על פיקים עקביים במרץ + ספטמבר, מה שמרמז על עונתיות אפשרית בביקוש.
* **Structural relationships** – קורלציה חיובית בין `full_sq` לבין המחיר, אך עם diminishing returns מעל 120 m²; קשר מתון בין `num_room` למחיר (multicollinearity עם שטח).

---

## 🛠️ Feature Engineering Highlights
| Category | Examples |
|----------|----------|
| **Numerical** | `full_sq`, `life_sq`, `floor`, `max_floor`, `kitch_sq`, `build_year` |
| **Categorical** | `material`, `product_type`, `sub_area` |
| **Macro** | `cpi`, `gdp_quart`, `oil_urals`, `usd_to_rub` |
| **Temporal** | `month`, `quarter`, `year`, `month_year_cnt` |
| **Interaction** | `price_per_sq = price_doc / full_sq`, `room_size_ratio = life_sq / full_sq` |

Missing values are imputed with **median** (numeric) or **most‑frequent** (categorical) inside a `ColumnTransformer`.

---

## 🤖 Modeling & Ensembling
| Model | RMSLE CV | Notes |
|-------|----------|-------|
| XGBoost (baseline) | 0.3361 | `eta=0.05`, 800 trees |
| LightGBM | **0.3287** | Categorical‑feature native handling, `num_leaves=512` |
| CatBoost | 0.3294 | GPU‑accelerated (cat boosting) |
| **Weighted Average Ensemble** | **0.3269** | 0.4 LGBM + 0.35 CatBoost + 0.25 XGB |

> Hyper‑parameters tuned with **Optuna TPE**; early‑stopping based on 20‑fold time‑series CV.

---

## 📊 Results
* **Public leaderboard**: 0.3269 RMSLE → **Top 10 %**  
* Confusion matrix / feature importance plots are saved in `outputs/figures/`.

---

## 🚀 Quick Start
```bash
# Clone the repo
git clone https://github.com/PrielDavid/ml-russian-house-prices.git
cd ml-russian-house-prices

# Create & activate env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Download Kaggle data (requires kaggle API key)
kaggle competitions download -c sberbank-russian-housing-market -p data/raw
unzip data/raw/sberbank-russian-housing-market.zip -d data/raw

# Reproduce CV & generate submission
jupyter lab notebooks/03_modeling.ipynb
```

---

## ♻️ Reproducibility
* Random seed fixed (`42`) everywhere  
* Entire preprocessing + modeling encapsulated in `scikit‑learn` pipelines  
* Models saved with `joblib` and can be loaded via `src/modeling.py`  

---

## 🔮 Next Steps
* **SHAP** analysis for interpretability  
* Add **stacking** (meta‑learner) to push RMSLE lower  
* Deploy a **Streamlit** app for interactive price prediction  

---

## 👤 Author
**Priel Davidpor** — BSc Economics & Statistics, Ben‑Gurion University  
Connect via [LinkedIn](https://www.linkedin.com/in/priel-davidpor/) or open an issue.

---

## 📄 License
This project is released under the **MIT License**.
