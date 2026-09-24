# 🎓 Student Performance Analysis & Academic Risk Prediction

---

## 📌 Project Overview

This end-to-end Business Intelligence project analyses the academic performance of 1,000 students across key dimensions — grades, attendance, study habits, parental background, and socioeconomic indicators — to predict academic risk (Pass / Fail) using a supervised Machine Learning model.

The deliverable is a fully interactive **Streamlit dashboard** covering:

| Layer | Coverage |
|---|---|
| Data Engineering | CSV ingestion, cleaning, feature engineering |
| Exploratory Analysis | KPI cards, distribution plots, correlation heatmaps |
| Machine Learning | Random Forest classifier with feature importance |
| Business Intelligence | Interactive Plotly visualisations, global filters |
| Risk Prediction | Live form → real-time Pass/Fail probability + recommendations |

---

## 🗂️ Repository Structure

```
Student Academic Performance/
├── student_info.csv                               # Raw dataset (1,000 records)
├── SudiptaMaity_Student-Performance-Analysis.py   # Main Streamlit application
├── requirements.txt                               # Python dependency list
├── SudiptaMaity_ProjectReport.docx                # Full project report document
└── README.md                                      # This file
```

---

## 📊 Dataset Description

**Source:** [Kaggle — Student Performance Dataset](https://www.kaggle.com/datasets)  
**File:** `student_info.csv`  
**Records:** 1,000 students | **Columns:** 15 attributes

| Column | Type | Description |
|---|---|---|
| `student_id` | String | Unique student identifier |
| `name` | String | Student name |
| `gender` | Categorical | Male / Female / Other |
| `age` | Numeric | Student age (13–19) |
| `grade_level` | Numeric | Grade 9–12 |
| `math_score` | Numeric | Math exam score (0–100) |
| `reading_score` | Numeric | Reading exam score (0–100) |
| `writing_score` | Numeric | Writing exam score (0–100) |
| `attendance_rate` | Numeric | Attendance % |
| `parent_education` | Categorical | High School / Bachelor's / Master's / PhD |
| `study_hours` | Numeric | Average daily study hours |
| `internet_access` | Categorical | Yes / No |
| `lunch_type` | Categorical | Standard / Free or reduced |
| `extra_activities` | Categorical | Yes / No |
| `final_result` | **Target** | **Pass / Fail** |

---

## ✨ Key Features

### 📊 Tab 1 — Executive Overview
- **3 Primary KPIs:** Total Students, Pass Rate %, Average Attendance Rate %
- Pass/Fail donut chart & grade-level pass rate bar chart
- Gender-wise pass rate and subject score distributions

### 📈 Tab 2 — Performance Drivers
- Scatter plot: Study Hours vs. Math Score (with OLS trendline)
- Violin plot: Attendance Rate vs. Final Result
- Bar charts: Parent Education impact, Internet Access, Lunch Type

### 🤖 Tab 3 — ML Model & Insights
- Random Forest model metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix heatmap
- Top-10 feature importance chart
- Full classification report table
- Numeric feature correlation heatmap

### ⚠️ Tab 4 — Live Risk Predictor
- Interactive form with all 13 feature inputs
- Real-time Pass/Fail prediction with confidence probability bar chart
- Personalised actionable recommendations based on risk factors

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1 — Clone / download the project

```bash
git clone https://github.com/your-username/student-performance-analysis.git
cd student-performance-analysis
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the application

```bash
python -m streamlit run SudiptaMaity_Student-Performance-Analysis.py
```

### Step 5 — Open in browser

The application will automatically open at:

```
http://localhost:8501
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32.0 | Web UI framework |
| `pandas` | ≥ 2.2.0 | Data manipulation |
| `numpy` | ≥ 1.26.0 | Numerical computing |
| `scikit-learn` | ≥ 1.4.0 | ML model (Random Forest) |
| `plotly` | ≥ 5.20.0 | Interactive visualisations |
| `openpyxl` | ≥ 3.1.2 | Excel export support |

---

## 🤖 Machine Learning Model

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 estimators |
| Max Depth | 10 |
| Train/Test Split | 80% / 20% |
| Stratification | Yes (class-balanced) |
| Target Variable | `final_result` (Pass / Fail) |

**Key Feature Importances** (from trained model):
1. `avg_score` — composite of math, reading, writing
2. `attendance_rate`
3. `study_hours`
4. `math_score`
5. `reading_score`

---

## 💡 Business Intelligence Insights

1. **Pass Rate** hovers around 50% — balanced but improvable through targeted interventions.
2. **Attendance** is the single strongest non-score predictor. Students below 85% are at high risk.
3. **Study Hours < 2 h/day** correlate strongly with Fail outcomes.
4. **Parental Education** shows a positive gradient: PhD-educated parents correlate with higher pass rates.
5. **Standard lunch** students outperform *Free or reduced* students by ~5 points on average score.
6. **Internet Access** shows a marginal positive effect, most visible in reading scores.

---

## 📋 Dashboard Screenshots Guide

1. Launch the app at `http://localhost:8501`
2. Use the **sidebar filters** (Gender, Grade Level) for segmented analysis
3. Navigate tabs: Overview → Performance Drivers → ML Insights → Risk Predictor
4. In the **Risk Predictor** tab, adjust sliders and dropdowns, then click **Predict Risk**

---

## 📄 License

This project is created for **IBM SkillsBuild Capstone** academic submission.  
Dataset courtesy of [Kaggle](https://www.kaggle.com/datasets).

---
