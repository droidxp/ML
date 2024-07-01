import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import csv



# Load CSV file
data = pd.read_csv('DS_Network_Flow_Final_Protocol_avg_min_max.csv')

# Select features and target variable
X = data.drop(['repack','hash','malicious'], axis=1)
y = data['malicious']
z = data['hash']
z_list = z.values.tolist()
y_list = y.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

#model = tree.DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=50, random_state=123, max_depth=8)
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
#df.to_csv('Neighbors.csv') 

print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))


