import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the 1,000-row dataset
df = pd.read_csv("casino_intel.csv")
X = df.drop('Player_Status', axis=1)
y = df['Player_Status']

# 2. Professional Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scaling (Essential for comparing Win Rate % with Bet $ values)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Gaussian Naive Bayes Model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 5. Advanced Metrics Calculation
y_pred = model.predict(X_test_scaled)
print(f"🚀 MODEL ACCURACY: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# 6. Save Assets
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ AI Brain (model.pkl) and Scaler (scaler.pkl) saved!")