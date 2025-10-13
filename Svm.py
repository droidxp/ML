import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import csv

data = pd.read_csv('cleaned_file.csv',index_col=0)

# Select features and target variable
X = data.drop(['hash','malicious'], axis=1)
y = data['malicious']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM model with custom parameters
svm = SVC(
    C=5.0,              # Regularization parameter (smaller values = stronger regularization)
    kernel='rbf',       # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',      # Kernel coefficient: 'scale', 'auto', or a float
    degree=3,           # Degree of the polynomial kernel (only for 'poly' kernel)
    probability=True,   # Enable probability estimates
    random_state=42     # Seed for reproducibility
)
#svm = SVC(
#    C=12.34,              # Regularization parameter (smaller values = stronger regularization)
#    kernel='rbf',       # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
#    gamma=0.56,      # Kernel coefficient: 'scale', 'auto', or a float
#    degree=3,           # Degree of the polynomial kernel (only for 'poly' kernel)
#    probability=True,   # Enable probability estimates
#    random_state=42     # Seed for reproducibility
#)



# Train the model on the training data
svm.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm.predict(X_test)

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, svm.predict(X_test)))
print(confusion_matrix(y_test, svm.predict(X_test)))

# predict probabilities
lr_probs = svm.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = svm.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))