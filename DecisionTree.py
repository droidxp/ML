import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import csv

# Load CSV file
data = pd.read_csv('cleaned_file.csv')

# Select features and target variable
X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']
y_list = y.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

model = RandomForestClassifier(n_estimators=400, random_state=123, min_samples_split= 18, min_samples_leaf= 3, max_samples= 3296, max_features= 'log2', max_depth=None)
model.fit(X_train, y_train)


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(model.estimators_[0],
               filled = True);
fig.savefig('rf_individualtree.png')

#predict = model.predict(X).tolist()
#dict = {'Repack': y_list, 'Hash': z_list, 'Pred': predict}  
       # create a Pandas DataFrame from the dictionary
#df = pd.DataFrame(dict) 
    # write the DataFrame to a CSV file
#df.to_csv('randomForest.csv')

predict = model.predict(X_test).tolist()
print('Total tested: %0d'%len(predict))
print('Total trained: %0d'%len(y_train))
print('Total samples: %0d'%len(y_list))
print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))
