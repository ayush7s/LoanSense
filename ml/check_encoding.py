import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

le = LabelEncoder()
cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    le.fit(df[col].astype(str))
    print(f"\n{col}:")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls} = {i}")