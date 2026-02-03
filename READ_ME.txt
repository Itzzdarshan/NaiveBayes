# 🎰 Vegas Edge: Real-Time Casino Intelligence & Behavioral Classification

## 🚀 Business Problem
Managing high-traffic casino floors requires the immediate identification of diverse player profiles—from high-value VIPs to potential security risks. **Vegas Edge** implements a probabilistic classification engine to transform floor telemetry into actionable security and hospitality protocols, optimizing floor management and mitigating financial risk.

## 🧠 Technical Architecture
The system utilizes a supervised learning pipeline optimized for probabilistic inference and real-time biometric-style telemetry analysis.

### 1. Multi-Dimensional Feature Analysis
The engine evaluates 5 critical behavioral vectors:
* **Volatility:** `Betting Pattern` (Steady vs. Erratic).
* **Financial Impact:** `Win Amount` and `Average Bet Size`.
* **Impulse Indicators:** `Alcohol Consumption` status.
* **Temporal Data:** `Time at Table` (Session duration).

### 2. Engineering Decisions: Gaussian Naive Bayes & Scaling
* [cite_start]**Algorithm Choice:** **Gaussian Naive Bayes (GNB)**[cite: 2].
    * *Justification:* GNB is exceptionally efficient for real-time applications where features are assumed to follow a normal distribution. [cite_start]It provides fast, probabilistic outputs ("AI Certainty") essential for security environments[cite: 1].
* [cite_start]**Preprocessing:** Implemented `StandardScaler` (Z-score normalization)[cite: 2].
    * [cite_start]*Critical Insight:* To ensure numerical stability, scaling was required to reconcile high-magnitude values like `Bet Amount` ($10,000+) with binary flags like `Drinking Status` (0 or 1)[cite: 2].


## 🛠️ Tech Stack & Tooling
* **Language:** Python 3.14
* [cite_start]**ML Engine:** Scikit-Learn (GaussianNB) [cite: 2]
* [cite_start]**Frontend:** Streamlit (Custom "Vegas Gold" UI with Glassmorphism and CSS Injection) [cite: 1]
* **Data Science:** Pandas, NumPy
* [cite_start]**Serialization:** Joblib (Persistence for Model and Scaling weights) [cite: 2]

## 📁 Repository Structure
```text
[cite_start]├── app.py              # Surveillance Terminal UI (Streamlit Frontend) [cite: 1]
[cite_start]├── train_model.py      # GNB Training pipeline with 80/20 train-test split [cite: 2]
[cite_start]├── model.pkl           # Serialized Naive Bayes "Brain" [cite: 2]
[cite_start]├── scaler.pkl          # Serialized Normalization Parameters [cite: 2]
[cite_start]└── casino_intel.csv    # 1,000+ row dataset of player telemetry [cite: 2]