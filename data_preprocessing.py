import numpy as np
import pandas as pd
from ast import If
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



education_mapping = {
    1: "Secondary Education - 12th Year of Schooling or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)"
}

target_mapping = {
    "Dropout": 0,
    "Enrolled": 0, 
    "Graduate": 1
}

# Load your dataset
data = pd.read_csv("graduation_dataset.csv")  # Replace with your dataset file path

data.replace({"Target": target_mapping}, inplace=True)

# Split the data into test and train
X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(data[data.columns[data.columns != 'Target']],
 data['Target'], test_size=0.25, random_state=1)


# print(dataset.describe())


# # Removing the last column
# # dataset = dataset.iloc[:, :-1]

# # f = plt.figure(figsize=(15, 15))
# # sns.heatmap(dataset.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
# #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# # plt.show()


# BEST FEATURE SELECTION

def best_feature_selection():
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_train.columns
    print(mutual_info)

    mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 5))
    # plt.show()
    return mutual_info



def t_sne():
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    #T-SNE
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the data
    tsne_result = tsne.fit_transform(scaled_data)

    # Create a scatter plot of the t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data['Target'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
    plt.colorbar(label='Target')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()



# PCA

def pca(n_components = 2):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Initialize PCA with the number of components you want to retain
    # n_components = 2  # You can adjust this based on your needs
    pca = PCA(n_components=n_components)

    # Fit PCA to the standardized data
    pca_result = pca.fit_transform(scaled_data)

    # Create a scatter plot of the PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['Target'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
    plt.colorbar(label='Target')
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


# RANDOM FOREST

def random_forest():
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)


# RANDOM FOREST ON T-SNE data

def rf_on_tsne():
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    #T-SNE
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the data
    X_tsne = tsne.fit_transform(scaled_data)

    X_train, X_test, y_train, y_test = train_test_split(X_tsne, data["Target"], test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)

def rf_on_pca():
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Initialize PCA with the number of components you want to retain
    n_components = 2  # You can adjust this based on your needs
    pca = PCA(n_components=n_components)

    # Fit PCA to the standardized data
    pca_result = pca.fit_transform(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(pca_result, data["Target"], test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)

def rf_on_best_features():
    sel_cols = SelectKBest(mutual_info_classif, k=10)
    sel_cols.fit(X_train, y_train)
    X_train.columns[sel_cols.get_support()]
    print(X_train.columns[sel_cols.get_support()])

    X_train_new, X_test, y_train_new, y_test = train_test_split(data[X_train.columns[sel_cols.get_support()]], data["Target"], test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_new, y_train_new)

    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)




if __name__ == "__main__":
    print("Happy data preprocessing and modeling!")
    # best_feature_selection()
    # t_sne()
    # pca()
    # random_forest()
    # rf_on_tsne()
    # rf_on_pca()
    # rf_on_best_features()