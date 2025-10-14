import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from scipy.stats import uniform


import csv

data = pd.read_csv('cleaned_file.csv',index_col=0)

# Select features and target variable
X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']

y_list = y.values.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)


#all features
model = LogisticRegression(C=0.00010714932985218937, solver='liblinear', max_iter=100, penalty= 'l1')

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


predict = model.predict(X_test).tolist()

dict = {'Index': X_test.index , 'Malicious': y_test, 'Pred': predict}
       
# create a Pandas DataFrame from the dictionary
df = pd.DataFrame(dict) 
# write the DataFrame to a CSV file
df.to_csv('LogisticRegression.csv', index=False) 


print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))

