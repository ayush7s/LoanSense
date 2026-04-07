import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
print("Dataset Loaded! Shape:", df.shape)

# Fix missing values — MUST assign back!
df['Gender']           = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']          = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']       = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']    = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Encode all text columns INCLUDING Loan_Status
le = LabelEncoder()
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

print("\nFeature order:")
print(list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# class_weight='balanced' fixes approval/rejection imbalance
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")

# Test a strong profile — should be Approved
test = np.array([[1, 1, 0, 0, 0, 5000, 0, 100, 360, 1, 1]])
result    = model.predict(test)[0]
confidence = model.predict_proba(test).max()
print(f"Strong profile test: {'Approved ✅' if result == 1 else 'Rejected ❌'} ({confidence*100:.1f}%)")

joblib.dump(model, 'model.pkl')
print("\nmodel.pkl saved!")
print("To load the model later, use: model = joblib.load('model.pkl')" )