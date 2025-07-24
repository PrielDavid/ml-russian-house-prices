
# 🇷🇺 Russian Housing Price Prediction (ml‑russian‑house‑prices)

A concise, practice‑oriented solution for the **Sberbank Russian Housing Market** Kaggle competition.  
The repository now keeps only the essential artefacts:

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

Large datasets, intermediate artefacts and trained models are **git‑ignored** to keep the repo lightweight.

---

## 📝 Project Overview
1. **EDA & Pre‑processing** – explore dataset, handle missing values, engineer features  
2. **Model training** – LightGBM with Optuna‑tuned hyper‑parameters (RMSLE objective)  
3. **Evaluation & leaderboard** – achieved **Top 10 %** public leaderboard score  
4. **Submission** – `final_submission.csv` ready for upload to Kaggle  

All steps are documented in **`Final_Work_ML_Project.ipynb`** (and exported as PDF for quick viewing).

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/PrielDavid/ml-russian-house-prices.git
cd ml-russian-house-prices

# Create environment
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download competition data (needs kaggle API key)
kaggle competitions download -c sberbank-russian-housing-market -p data
unzip data/sberbank-russian-housing-market.zip -d data

# Run the notebook
jupyter lab Final_Work_ML_Project.ipynb
```

---

## 🔍 Key Notebook Sections

| Section | What you’ll find |
|---------|------------------|
| **EDA** | Price trends by year, district heat‑map, macro‑economic correlation |
| **Feature Engineering** | Square‑meter ratios, temporal cycles, macro indicators |
| **Modeling** | LightGBM, XGBoost & CatBoost comparison, Optuna tuning |
| **Ensembling** | Weighted average to squeeze extra RMSLE gain |
| **Submission** | Generate `final_submission.csv` |

---

## 📦 Dependencies
See `requirements.txt` (Python 3.10). Core libs: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `jupyterlab`, `matplotlib`, `seaborn`.

---

## ♻️ Reproducibility
* Random seeds fixed (`42`).  
* Preprocessing + model wrapped in Scikit‑learn `Pipeline` to avoid leakage.  
* Notebook cells execute top‑down without manual tweaking.

---

## 👤 Author
**Priel Davidpor** – BGU Economics & Statistics  
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/priel-davidpor/).

---

## 📄 License
Distributed under the MIT License — see `LICENSE.txt` for details.
