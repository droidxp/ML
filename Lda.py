import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import csv

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
print(y_resampled.value_counts())

model = LDA(store_covariance=True)

model.fit(X_train, y_train)
#model.fit(X_resampled,y_resampled)


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



#predict = model.predict(X).tolist()

#dict = {'Repack': y_list, 'Hash': z_list, 'Pred': predict}  
       
# create a Pandas DataFrame from the dictionary
#df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
#df.to_csv('Lda.csv') 

predict = model.predict(X).tolist()
print(classification_report(y, predict))
print(confusion_matrix(y, predict))
dict = {'Index': X.index , 'Malicious': y, 'Pred': predict}


df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
df.to_csv('Lda.csv', index=False) 


print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))

