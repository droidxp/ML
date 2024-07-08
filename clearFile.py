import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier

Malware = pd.read_csv("Final_file.csv")

Malware = Malware.fillna(Malware.iloc[:,5699].median())

unique_columns = Malware.loc[:, Malware.nunique() == 1]
unique_columns = list(unique_columns.columns)
Malware_cleaned = Malware.drop(unique_columns, axis=1)

X = Malware_cleaned.drop(['hash','malicious'], axis=1)
y = Malware_cleaned['malicious']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

model = RandomForestClassifier(n_estimators=230, random_state=123, min_samples_split= 16, min_samples_leaf= 19, max_samples= 3274, max_features= 2000, max_depth=10)
model.fit(X_train, y_train)

feature_names = model.feature_names_in_
importances = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 

Malware_cleaned = Malware_cleaned.loc[:,list(feature_imp_df.Feature[:20])+['hash','malicious']]
print(Malware_cleaned)
#Malware_cleaned = Malware_.dropna()
Malware_cleaned.to_csv("cleaned_file.csv", index = False)