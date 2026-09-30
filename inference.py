# inference.py  -  score one new loan applicant
import joblib
import numpy as np
model = joblib.load("models/loan_model.joblib")   # created by train.py
# 8 feature values for one applicant (same order as training)
applicant = np.array([[0.5, -1.2, 0.3, 1.1, -0.4, 0.8, 0.0, -0.6]])
decision = model.predict(applicant)[0]
print("Loan APPROVED" if decision == 1 else "Loan REJECTED")