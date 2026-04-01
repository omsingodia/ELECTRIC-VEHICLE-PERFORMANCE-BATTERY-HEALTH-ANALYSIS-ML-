⚡ EV Battery Health Prediction — CS1138 ML Project
Python scikit-learn Pandas Status Course

Course Project — CS1138 | Team Size: 2–3 | End-to-End Machine Learning Pipeline on Electric Vehicle Battery Analytics

📌 Problem Statement
Predict the battery health of electric vehicles using real-world sensor and usage data. Battery degradation is a critical concern for EV adoption — early detection of unhealthy batteries can prevent failures and reduce costs.

We solve this as two tasks:

🔢 Regression — Predict the exact Battery_Health_% value
🏷️ Classification — Predict if a battery is Healthy (1) or Unhealthy (0)
📁 Repository Structure
ev-battery-health-ml/
│
├── data/
│   └── electric_vehicle_analytics_CHANGED.csv   # Dataset (Kaggle)
│
├── notebooks/
│   └── ev_battery_analysis.ipynb                # Main Colab notebook
│
├── src/
│   └── ev_ml_pipeline.py                        # Clean Python script version
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_comparison.png
│   ├── f1_score_comparison.png
│   ├── r2_comparison.png
│   ├── mae_comparison.png
│   └── correlation_heatmap.png
│
├── requirements.txt
├── .gitignore
└── README.md
📊 Dataset
Property	Value
Source	Kaggle — Electric Vehicle Analytics
Format	CSV
Target (Regression)	Battery_Health_%
Target (Classification)	Battery_Status (0 = Unhealthy, 1 = Healthy)
Threshold	Median of Battery_Health_% (auto-balanced)
Features used:

Feature	Description
Charge_Cycles	Number of charge/discharge cycles
Mileage_km	Total distance driven
Temperature_C	Operating temperature
Battery_Capacity_kWh	Rated battery capacity
Energy_Consumption_kWh_per_100km	Energy efficiency
Charging_Time_hr	Average charging duration
Charging_Power_kW	Charging power used
🤖 Models Implemented
Regression
Model	R² Score	MAE
Linear Regression	~0.62	~7.8
Random Forest	~0.91	~3.2
KNN	~0.74	~5.9
SVM (SVR)	~0.70	~6.1
Classification
Model	Accuracy	F1 Score
Logistic Regression	~81%	~0.80
KNN	~84%	~0.83
SVM	~88%	~0.87
Random Forest	~86%	~0.85
✅ Best Model: SVM — Highest accuracy and F1 score on balanced test set

🔁 Pipeline Overview
Raw CSV Data
    │
    ▼
Data Cleaning (drop Make, Model, Vehicle_Type)
    │
    ▼
EDA (Distribution plots, Scatter plots, Correlation Heatmap)
    │
    ▼
Feature Engineering (Auto-threshold → Battery_Status)
    │
    ▼
Class Balancing (Downsampling majority class)
    │
    ▼
Train/Test Split (80/20, stratified)
    │
    ▼
StandardScaler (Normalization)
    │
    ▼
Model Training (4 Regression + 4 Classification models)
    │
    ▼
Evaluation (R², MAE, Accuracy, F1, Confusion Matrix)
    │
    ▼
Visualization & Comparison
🚀 How to Run
Option 1 — Google Colab (Recommended)
Open notebooks/ev_battery_analysis.ipynb in Google Colab
Upload the CSV when prompted (files.upload())
Run all cells
Option 2 — Local
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ev-battery-health-ml.git
cd ev-battery-health-ml

# Install dependencies
pip install -r requirements.txt

# Place your CSV in data/ folder and run
python src/ev_ml_pipeline.py
📦 Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
Install all:

pip install -r requirements.txt
📈 Key Results
Random Forest dominates regression with R² ≈ 0.91
SVM wins classification with ~88% accuracy and strong F1
Confusion matrix diagonal is high (correct predictions) — class balance was achieved via median threshold + downsampling
Correlation heatmap confirms Charge_Cycles and Mileage_km are most predictive features
👥 Team
Name	Role
Member 1	Data Preprocessing + EDA
Member 2	Model Training + Evaluation
Member 3	Visualization + Report
📚 References
Dataset: Kaggle — Electric Vehicle Analytics
scikit-learn Documentation: https://scikit-learn.org
CS1138 Course Guidelines
📝 License
This project is submitted for academic purposes under CS1138.
