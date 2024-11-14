import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn. discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

import csv


# Load CSV file
data = pd.read_csv('cleaned_file.csv',index_col=0)

X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']

y_list = y.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

model = QDA( store_covariance=True)
model.fit(X_train, y_train)

#predict = model.predict(X).tolist()

#dict = {'Repack': y_list, 'Hash': z_list, 'Pred': predict}  
       
# create a Pandas DataFrame from the dictionary
#df = pd.DataFrame(dict) 
    
# write the DataFrame to a CSV file
#df.to_csv('Qda.csv') 

print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))


