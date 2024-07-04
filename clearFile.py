import pandas as pd

Malware = pd.read_csv("Final_file.csv")
unique_columns = Malware.loc[:, Malware.nunique() == 1]
unique_columns = list(unique_columns.columns)
Malware_ = Malware.drop(unique_columns, axis=1)
Malware_cleaned = Malware_.dropna()
Malware_cleaned.to_csv("cleaned_file.csv", index = False)
