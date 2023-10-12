import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("graduation_dataset.csv")

print(data)

training_set, test_set = train_test_split(data, 0.2, 1)

X_train = training_set.iloc[:,0:2].values
Y_train = training_set.iloc[:,2].values
X_test = test_set.iloc[:,0:2].values
Y_test = test_set.iloc[:,2].values