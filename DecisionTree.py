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

#model = tree.DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=390, random_state=123, max_depth=13)
#model = RandomForestClassifier(n_estimators=270, random_state=123, min_samples_split= 14, min_samples_leaf= 19, max_samples= 3274, max_features= 2000, max_depth=10)
#features without dst_port
#model = RandomForestClassifier(n_estimators=190, random_state=123, min_samples_split= 4, min_samples_leaf= 7, max_samples= 3274, max_features= 2000, max_depth=10)

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(y_resampled.value_counts())

model = RandomForestClassifier(n_estimators=400, random_state=123, min_samples_split= 18, min_samples_leaf= 3, max_samples= 1668, max_features= 'log2', max_depth=None)


#param_dist = {'n_estimators': randint(50,500),
#              'max_depth': randint(1,20)}
#param_dist = {"n_estimators": np.arange(10, 500, 10),
#           "max_depth": [None, 3, 5, 10, 15],
#           "min_samples_split": np.arange(2, 20, 2),
#           "min_samples_leaf": np.arange(1, 20, 2),
#           "max_features": [1,2000 , "sqrt", "log2"],
#           "max_samples": [3296]}

#model = RandomForestClassifier(n_jobs=-1, random_state=123)

#rand_search = RandomizedSearchCV(model, 
#                                 param_distributions = param_dist, 
#                                 n_iter=5, 
#                                 cv=5,verbose=True)

#model.fit(X_train, y_train)
model.fit(X_resampled,y_resampled)

#rand_search.fit(X_train, y_train)

#best_rf = rand_search.best_estimator_

#Print the best hyperparameters
#print('Best hyperparameters:',  rand_search.best_params_)


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(model.estimators_[0],
               filled = True);
fig.savefig('rf_individualtree.png')

#predict = model.predict(X_test).tolist()
#dict = {'Index': X_test.index , 'Malicious': y_test, 'Pred': predict}  

predict = model.predict(X).tolist()
dict = {'Index': X.index , 'Malicious': y, 'Pred': predict} 
       
# create a Pandas DataFrame from the dictionary
df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
df.to_csv('RandomForest.csv', index=False) 

predict = model.predict(X_test).tolist()
print('Total tested: %0d'%len(predict))
print('Total trained: %0d'%len(y_train))
print('Total samples: %0d'%len(y_list))
print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))




#feature_names = model.feature_names_in_

#importances = model.feature_importances_
#feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
#print(feature_imp_df.explode('Feature')[:50])


#result = permutation_importance(
#    model, X_test, y_test, n_repeats=50, random_state=42, n_jobs=2
#)
#importances = result.importances_mean
#feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
#print(feature_imp_df[:50])
