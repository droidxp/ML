import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc

import csv

data = pd.read_csv('cleaned_file.csv',index_col=0)

# Select features and target variable
X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the MLPClassifier (Neural Network)
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # Two hidden layers with 10 neurons each
    activation='relu',           # Activation function
    solver='adam',               # Optimization algorithm
    max_iter=1000,               # Maximum number of iterations
    random_state=42
)
#mlp_classifier = MLPClassifier(activation='logistic',alpha=0.030187830981676968,batch_size=122,hidden_layer_sizes= (100,50),
#                               learning_rate='constant',learning_rate_init=0.027678101427528502, solver='sgd')

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, mlp_classifier.predict(X_test)))
print(confusion_matrix(y_test, mlp_classifier.predict(X_test)))

# predict probabilities
lr_probs = mlp_classifier.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = mlp_classifier.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))