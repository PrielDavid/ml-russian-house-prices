
# Real Estate Price Prediction 🏡

This project builds an end‑to‑end machine‑learning pipeline that predicts apartment sale prices in Israel.  
It walks through **exploratory data analysis, data cleaning, feature engineering, model training, hyper‑parameter tuning, and final evaluation** — all consolidated in the Jupyter notebook **`Final_Work_ML_Project.ipynb`**.

---

## 📈 Problem Statement
Given historical apartment transactions with structural, locational and temporal attributes, predict the sale **price** (in ILS). We frame this as a supervised regression task and optimise for the *Root Mean Squared Log Error* (RMSLE).

---

## 🗂️ Project Structure
```text
.
├── data/
│   ├── raw/               # Original CSVs
│   ├── processed/         # Cleaned & feature‑engineered datasets
├── notebooks/
│   └── Final_Work_ML_Project.ipynb
├── models/                # Saved model binaries
├── outputs/
│   └── submission.csv     # Kaggle / competition submission format
├── README.md              # ← you are here
└── requirements.txt
```

> **Note**: Only the notebook is committed; large datasets and models are excluded via `.gitignore`.

---

## 🔍 Exploratory Data Analysis (EDA)
* Descriptive statistics for numeric & categorical features  
* Correlation heat‑map to spot multicollinearity  
* Distribution plots & outlier inspection  
* Temporal trends in monthly transaction volume and average price  
* Geographical price heat‑map by district

Key insights drove targeted cleaning steps, e.g. removing impossible floor numbers (`floor > 70`), capping extreme net‑area values and correcting material‑type typos.

---

## 🛠️ Feature Engineering
| Category | Examples |
|----------|----------|
| **Numerical** | `net_size`, `num_rooms`, `age`, `floor`, `max_floor` |
| **Categorical** | `material`, `district`, `neighborhood` |
| **Temporal** | `sale_month`, `sale_quarter`, `sale_year` |
| **Spatial** | One‑hot district + external socio‑economic index |
| **Interaction** | `price_per_sqm = price / net_size` |

Missing values were imputed with *median* (numerical) or *most‑frequent* (categorical) inside a `ColumnTransformer`.

---

## 🤖 Modelling Pipeline
| Model | Pre‑processing | Highlights |
|-------|----------------|------------|
| **XGBoost** (baseline) | Target log‑transform, one‑hot, `RMSLE` metric | Robust to skew & non‑linearities |
| **XGBoost + PCA** | Added dimensionality reduction (30 PCs) | Slight speed‑up, comparable error |
| **LightGBM** (final) | Native categorical handling, histogram tree growth | **Best validation RMSLE**, faster training |

> LightGBM was selected for its **accuracy**, **scalability** and **memory efficiency** on large sparse matrices.

---

## 📊 Results
Detailed cross‑validation scores, hyper‑parameters and learning curves are provided in the notebook.  
A separate `submission.csv` is produced for leaderboard evaluation.

---

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/<your‑handle>/real‑estate‑price‑prediction.git
cd real‑estate‑price‑prediction

# 2. Create environment
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch notebook
jupyter lab notebooks/Final_Work_ML_Project.ipynb
```

`requirements.txt` (excerpt):

```
pandas>=2.0
numpy>=1.25
scikit-learn>=1.5
xgboost>=2.0
lightgbm>=4.3
matplotlib>=3.9
seaborn>=0.13
jupyterlab
```

---

## 📦 Reproducibility
* Random seeds fixed (`random_state=42`).
* Full preprocessing & training wrapped in **Scikit‑learn pipelines** (`Pipeline`, `ColumnTransformer`).
* Final model and encoders serialised with `joblib`.

To re‑generate `submission.csv` run:

```bash
python scripts/train_and_predict.py
```

---

## 📝 Future Work
* Spatial feature enrichment with latitude/longitude and proximity to amenities  
* Hyper‑parameter optimisation via Optuna  
* SHAP value analysis for model explainability  
* Deployment demo with FastAPI + Streamlit dashboard

---

## 👤 Author
**Priel Davidpor** – Economics & Statistics BSc — Ben‑Gurion University  
Feel free to reach out via [LinkedIn](https://www.linkedin.com/in/priel‑davidpor/) or open an issue.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
