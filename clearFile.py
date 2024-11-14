import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from os.path import dirname, join

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

current_dir = dirname(__file__)
file_path = join(current_dir, "Final_file.csv")
Malware = pd.read_csv(file_path)

unique_columns = Malware.loc[:, Malware.nunique() == 1]
unique_columns = list(unique_columns.columns)
Malware_cleaned = Malware.drop(unique_columns, axis=1)

#Malware_cleaned = Malware_cleaned.fillna(Malware_cleaned[Malware_cleaned.columns[:5373]].median())
Malware_cleaned = Malware_cleaned.fillna(0)
Malware_cleaned = Malware_cleaned.dropna()

X = Malware_cleaned.drop(['hash','malicious'], axis=1)
y = Malware_cleaned['malicious']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

model = RandomForestClassifier(n_estimators=230, random_state=123, min_samples_split= 16, min_samples_leaf= 19, max_samples= 2846, max_features= 2000, max_depth=10)
model.fit(X_train, y_train)


# predict probabilities
lr_probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))


feature_names = model.feature_names_in_
importances = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 

Malware_cleaned = Malware_cleaned.loc[:,list(feature_imp_df.Feature[:20])+['hash','malicious']]

Malware_cleaned.to_csv("cleaned_file.csv", index_label = 'index', index = True)

file_path = join(current_dir, "large_ds.csv")
TSE = pd.read_csv(file_path)

Malware_cleaned_family = pd.merge(Malware_cleaned, TSE[['sha256','family']], left_on='hash', right_on='sha256', how='left')
Malware_cleaned_family = Malware_cleaned_family.drop(['sha256'], axis=1)
Malware_cleaned_family = Malware_cleaned_family.dropna()
Malware_cleaned_family.replace("None", "NoFamily", inplace=True)
Malware_cleaned_family.to_csv("cleaned_file_family.csv", index_label = 'index', index = True)