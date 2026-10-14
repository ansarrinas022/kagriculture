# 🌱 Kaggriculture — Agricultural Simulation Analysis & Strategy Platform

> **A Python-first, data-driven analytical platform for evaluating multi-agent agricultural simulation episodes, comparing competitive farming strategies, diagnosing turn mechanics, and training predictive models.**

---

## 🌟 Overview

**Kaggriculture** is an interactive analytics suite built with **Python**, **Streamlit**, **Pandas**, **NumPy**, **Plotly**, and **Scikit-learn**. It is specifically designed to ingest match replays from the [Kaggriculture Kaggle Competition](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-2026-09-08), where autonomous agents manage 30-day farming seasons (720 turns), balance crop cultivation, dynamic market sales, land expansion, and livestock management.

---

## 🚀 Key Features

| Module | Description |
|---|---|
| **📊 Overview** | Live macro metrics (episodes, agents, win rates, memory size, peak score), score histograms, leaderboard rankings, and performance trends across match sequences. |
| **🔍 Episode Explorer** | Match search by episode ID, filtering by agent, sorting, pagination, card inspectors, and granular turn-by-turn replay timelines with CSV/JSON exports. |
| **🚜 Action Analysis** | Categorized farming activities (planting wheat/corn/carrots, watering, harvesting, livestock care, market sales, land quadrants), winning vs. losing action differentials, and resource conversion scatter plots. |
| **⚔️ Agent Comparison** | True head-to-head matched episode rivalry statistics, score gap analysis, multi-dimensional radar comparison, and box plot volatility audits. |
| **📈 Statistical Analysis** | Parametric & non-parametric feature tables (mean, std, IQR, skewness), correlation heatmaps with Pearson/Spearman selection, outlier detection via IQR & Z-Score, and dynamic subgroup cross-tabs. |
| **💡 Strategy & ML** | Empirical rules comparing Q4 vs. Q1 performance cohorts, supervised Random Forest score prediction with feature importances, and unsupervised K-Means strategy clustering with 2D PCA projection. |
| **🗄️ Data Management** | Multi-file uploader (CSV, JSON, JSONL, Parquet), Kaggle API downloader, schema mapping inspector, data validation checks, and non-destructive cleaning (deduplication, imputation, outlier trimming) with live audit logs. |
| **📑 Executive Reports** | Downloadable HTML briefing reports with embedded CSS styling, statistical disclaimers, agent comparison summaries, and CSV leaderboard exports. |

---

## 🛠️ Technology Stack

- **Core Application**: Python 3.10+
- **Frontend Dashboard**: Streamlit
- **Data Engineering**: Pandas, NumPy, PyArrow
- **Interactive Visualizations**: Plotly
- **Machine Learning**: Scikit-learn, SciPy
- **Testing**: Pytest

---

## 💻 Local Windows Installation & Quickstart

### 1. Clone or Open Workspace
Open the project directory in VS Code or PowerShell:
```powershell
cd c:\Users\User\Downloads\kagri
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1
# (or in Windows Command Prompt: .venv\Scripts\activate.bat)
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Platform
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 📥 Kaggle Dataset Ingestion

### Official Dataset Link:
[https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-2026-09-08](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-2026-09-08)

### Ingestion Methods:
1. **Interactive File Upload**: Go to **Data Management** in the sidebar, drag and drop your downloaded CSV, JSON, JSONL, or Parquet files.
2. **Kaggle CLI**:
   Place your `kaggle.json` into `C:\Users\<Username>\.kaggle\kaggle.json` and run:
   ```powershell
   kaggle datasets download -d kaggle/kaggriculture-episodes-2026-09-08 --unzip -p data/
   ```
3. **Instant Sample Mode**: The application ships with high-fidelity pre-generated sample benchmarks (`data/kaggriculture_sample_episodes.csv` and `data/kaggriculture_sample_replays.json`) clearly labeled as sample data so you can test and explore immediately.

---

## 🧪 Running Unit Tests

Run the full automated test suite:
```powershell
pytest
```
Expected output:
```
============================== 8 passed in 0.44s ==============================
```

---

## 🛡️ Statistical Limitations & Disclaimers

- Correlation does **not** imply causation. Observed associations between farming actions (e.g., land expansion) and winning scores reflect empirical historical tendencies.
- Predictions and strategy recommendations demonstrate statistical patterns and do not guarantee future tournament victories against novel agent architectures.
# kagriculture
