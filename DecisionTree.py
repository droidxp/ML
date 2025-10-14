import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

import csv

# Load CSV file
data = pd.read_csv('cleaned_file.csv',index_col=0)

# Select features and target variable
X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']

y_list = y.values.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)


rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


model = RandomForestClassifier(n_estimators=210, random_state=123, min_samples_split= 10, min_samples_leaf= 3, max_samples= 2277, max_features= 'log2', max_depth=5)


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

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(model.estimators_[0],
               filled = True);
fig.savefig('rf_individualtree.png')

predict = model.predict(X_test).tolist()

print(confusion_matrix(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))

predict = model.predict(X_test).tolist()

dict = {'Index': X_test.index , 'Malicious': y_test, 'Pred': predict}
       
# create a Pandas DataFrame from the dictionary
df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
df.to_csv('RandomForest.csv', index=False) 



