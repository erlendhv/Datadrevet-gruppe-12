from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# Load the dataset
data = pd.read_csv("graduation_dataset2TokyoDrift.csv")

print(data.info())

# Split the data into training and test sets
training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)

# Extract features and labels
column= 34
X_train = training_set.iloc[:, 0:column].values
Y_train = training_set.iloc[:, column].values
X_test = test_set.iloc[:, 0:column].values
Y_test = test_set.iloc[:, column].values

# Encode the class labels
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

# Train the SVM classifier
classifier = SVC()
classifier.fit(X_train, Y_train)

# Predict on the test set
Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred

n=0
antall=0
for row in Y_pred:
    antall+=1
    if row == 1:
        n+=1
print("Antall faliures: ",n/antall*100,"%")
    
    
print(classification_report(Y_test, Y_pred))
# Calculate accuracy
cm = confusion_matrix(list(Y_test), Y_pred)
accuracy = float(np.diagonal(cm).sum()) / len(Y_test)
print("\nAccuracy Of SVM For The Given Dataset: ", accuracy)

# Visualization (2D decision boundary for binary classification)
# plt.figure(figsize=(7, 7))
# X_set, y_set = X_train, Y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
#              cmap=ListedColormap(('black', 'white')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'orange'))(i), label=j)
# plt.title('Student dataset')
# plt.legend()
# plt.show()