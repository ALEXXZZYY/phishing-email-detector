import pandas as pd

file_path = "data/emails.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of emails:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nDataset:")
print(df)