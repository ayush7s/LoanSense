import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df['Loan_Status'] = df['Loan_Status'].fillna(df['Loan_Status'].mode()[0])

le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'].astype(str))

print("Loan_Status classes:", le.classes_)
print("N =", le.transform(['N'])[0])
print("Y =", le.transform(['Y'])[0])
print("\nApproval rate in dataset:")
print(df['Loan_Status'].value_counts())