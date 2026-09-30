# train.py  -  loan-approval model (synthetic data for practice)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib, os
# Pretend the 8 features are income, EMI burden, credit score, age, ...
X, y = make_classification(n_samples=2000, n_features=8, n_informative=5,
                           random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
acc = accuracy_score(y_val, model.predict(X_val))
print(f"Validation accuracy: {acc:.3f}")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/loan_model.joblib")