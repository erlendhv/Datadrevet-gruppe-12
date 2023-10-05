from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM,LogisticGAM, s, f
from sklearn.model_selection import GridSearchCV
from pygam.datasets import wage

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("graduation_dataset.csv")

target_mapping = {
    "Dropout": 0,
    "Enrolled": 0,
    "Graduate": 1
}

data.replace({"Target": target_mapping}, inplace=True)


data.columns = data.iloc[0]

data = data[1:]

X = data[data.columns[:-1]] # Get features
y = data[data.columns[-1]]  # Get target


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# rf = LinearRegression()

# rf = rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)

# print(y_pred)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("F1:",metrics.f1_score(y_test, y_pred))


# gam = LinearGAM()
# gam.gridsearch(X, y)
# gam.summary()

# fig = plt.figure()

# i=0
# XX = gam.generate_X_grid(term=i)
# fig.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
# fig.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')


X, y = wage(return_X_y=True)


## model
gam = LinearGAM(s(0) + s(1) + f(2))
gam.gridsearch(X, y)

i =1
XX = gam.generate_X_grid(term=i)
plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')


## plotting
plt.figure();