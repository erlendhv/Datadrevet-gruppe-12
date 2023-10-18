from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# Load the dataset
data = pd.read_csv("graduation_dataset.csv")
target_mapping = {
     "Dropout": 0,
     "Enrolled": 0,
     "Graduate": 1
}
data.replace({"Target": target_mapping}, inplace=True)


data = data.drop(columns=['International', 'Curricular units 1st sem (credited)', 'Educational special needs', 'Displaced', "Father\'s occupation", 'Mother\'s qualification', 'Nacionality', 'Daytime/evening attendance', 'Inflation rate', 'GDP'])
print(data.info())


# Split the data into training and test sets
training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

# Extract features and labels
column = len(data.columns) - 1
X_train = training_set.iloc[:, 0:column].values
Y_train = training_set.iloc[:, column].values
X_test = test_set.iloc[:, 0:column].values
Y_test = test_set.iloc[:, column].values

#standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100], 
              'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'poly', 'linear']} 

#param_grid = {'C': [0.01, 0.1], 
#              'gamma': [1, 0.75], 
#              'kernel': ['poly', 'linear']} 

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train, Y_train)

best_params = grid.best_params_
print(f"Best params: {best_params}")


# Encode the class labels
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

# Train the SVM classifier
classifier = SVC(**best_params)
classifier.fit(X_train, Y_train)

# Predict on the test set
Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred
    
    
print(classification_report(Y_test, Y_pred))
# Calculate accuracy
cm = confusion_matrix(list(Y_test), Y_pred)
accuracy = float(np.diagonal(cm).sum()) / len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset: ", accuracy)