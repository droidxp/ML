import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import csv


# Load CSV file
data = pd.read_csv('DS_Network_Flow_Final_Protocol_avg_min_max.csv')

# Select features and target variable
X = data.drop(['repack','hash'], axis=1)
y = data['repack']
z = data['hash']
z_list = z.values.tolist()
y_list = y.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

model = LogisticRegression(solver='liblinear', random_state=0, max_iter=100000)
model.fit(X_train, y_train)

importance = model.coef_[0]
#summarize feature importance
#for i,v in enumerate(importance):
#	print('Feature: %0d, Score: %.5f' % (i,v))

predict = model.predict(X_test).tolist()
print('Total tested: %0d'%len(predict))
print('Total trained: %0d'%len(y_train))
print('Total samples: %0d'%len(y_list))

#print(len(z_list))
#dict = {'Repack': y_list, 'Hash': z_list, 'Pred': predict}  
       
# create a Pandas DataFrame from the dictionary
#df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
#df.to_csv('LogisticRegression.csv') 
print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))

