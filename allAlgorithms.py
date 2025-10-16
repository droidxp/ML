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
from sklearn.linear_model import LogisticRegression
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn. discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Load CSV file
data = pd.read_csv('cleaned_file.csv',index_col=0)

X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']

y_list = y.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)

modelRF = RandomForestClassifier(n_estimators=210, random_state=123, min_samples_split= 10, min_samples_leaf= 3, max_samples= 2277, max_features= 'log2', max_depth=5)
modelLR = LogisticRegression(C=0.00010714932985218937, solver='liblinear', max_iter=100, penalty= 'l1')
modelLDA = LDA(tol=0.0001,solver='eigen',shrinkage=0.5540245860141142)
modelQDA = QDA( store_covariance=True,reg_param=0.9997110797761021)
modelMLP = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # Two hidden layers with 10 neurons each
    activation='relu',           # Activation function
    solver='adam',               # Optimization algorithm
    max_iter=1000,               # Maximum number of iterations
    random_state=42
)
modelSVM = SVC(
    C=5.0,              # Regularization parameter (smaller values = stronger regularization)
    kernel='rbf',       # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',      # Kernel coefficient: 'scale', 'auto', or a float
    degree=3,           # Degree of the polynomial kernel (only for 'poly' kernel)
    probability=True,   # Enable probability estimates
    random_state=42     # Seed for reproducibility
)

modelRF.fit(X_train, y_train)
modelLR.fit(X_train, y_train)
modelLDA.fit(X_train, y_train)
modelQDA.fit(X_train, y_train)
modelMLP.fit(X_train, y_train)
modelSVM.fit(X_train, y_train)

predict = modelRF.predict(X_test).tolist()
print("Random Forest:")
print(classification_report(y_test, modelRF.predict(X_test)))
print(confusion_matrix(y_test, modelRF.predict(X_test)))

predict = modelLR.predict(X_test).tolist()
print("Logistic Regression:")
print(classification_report(y_test, modelLR.predict(X_test)))
print(confusion_matrix(y_test, modelLR.predict(X_test)))

predict = modelLDA.predict(X_test).tolist()
print("Linear Discriminant Analysis:")
print(classification_report(y_test, modelLDA.predict(X_test)))
print(confusion_matrix(y_test, modelLDA.predict(X_test)))

predict = modelQDA.predict(X_test).tolist()
print("Quadratic Discriminant Analysis:")
print(classification_report(y_test, modelQDA.predict(X_test)))
print(confusion_matrix(y_test, modelQDA.predict(X_test)))

predict = modelMLP.predict(X_test).tolist()
print("Multi-layer Perceptron:")
print(classification_report(y_test, modelMLP.predict(X_test)))
print(confusion_matrix(y_test, modelMLP.predict(X_test)))

predict = modelSVM.predict(X_test).tolist()
print("Support Vector Machines")
print(classification_report(y_test, modelSVM.predict(X_test)))
print(confusion_matrix(y_test, modelSVM.predict(X_test)))