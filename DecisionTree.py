import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

import csv

# Load CSV file
data = pd.read_csv('cleaned_file.csv')

# Select features and target variable
X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']
z = data['hash']
z_list = z.values.tolist()
y_list = y.values.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

#model = tree.DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=390, random_state=123, max_depth=13)
model = RandomForestClassifier(n_estimators=270, random_state=123, min_samples_split= 14, min_samples_leaf= 19, max_samples= 3274, max_features= 2000, max_depth=12)

#param_dist = {'n_estimators': randint(50,500),
#              'max_depth': randint(1,20)}
#param_dist = {"n_estimators": np.arange(10, 500, 10),
#           "max_depth": [None, 3, 5, 10, 15],
#           "min_samples_split": np.arange(2, 20, 2),
#           "min_samples_leaf": np.arange(1, 20, 2),
#           "max_features": [1,2000 , "sqrt", "log2"],
#           "max_samples": [3274]}

#model = RandomForestClassifier(n_jobs=-1, random_state=123)

#rand_search = RandomizedSearchCV(model, 
#                                 param_distributions = param_dist, 
#                                 n_iter=5, 
#                                 cv=5,verbose=True)

model.fit(X_train, y_train)
#rand_search.fit(X_train, y_train)

#best_rf = rand_search.best_estimator_

#Print the best hyperparameters
#print('Best hyperparameters:',  rand_search.best_params_)


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(model.estimators_[0],
               filled = True);
fig.savefig('rf_individualtree.png')

#predict = model.predict(X).tolist()

#dict = {'Repack': y_list, 'Hash': z_list, 'Pred': predict}  
       
# create a Pandas DataFrame from the dictionary
#df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
#df.to_csv('Neighbors.csv') 

print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))