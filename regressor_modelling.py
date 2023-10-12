from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM,LogisticGAM, s, f
from sklearn.model_selection import GridSearchCV
from pygam.datasets import wage
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("graduation_dataset.csv")

target_mapping = {
    "Dropout": 0,
    "Enrolled": 0,
    "Graduate": 1
}

data.replace({"Target": target_mapping}, inplace=True)


# data.columns = data.iloc[0]

# data = data[1:]

X = data.drop('Target', axis=1)
y = data['Target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

y_pred = np.round(y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("F1:", metrics.f1_score(y_test, y_pred))

