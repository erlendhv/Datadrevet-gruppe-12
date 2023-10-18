import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC


class data_modeling:

    def __init__(self) -> None:
        # Load your dataset
        self.data = pd.read_csv("Datadrevet-gruppe-12/graduation_dataset.csv")  # Replace with your dataset file path
        # one hot encoding target column to numeric values
        target_mapping = {
            "Dropout": 0,
            "Enrolled": 0, 
            "Graduate": 1
        }

        self.data.replace({"Target": target_mapping}, inplace=True)

        # Split the data into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target']],
        self.data['Target'], test_size=0.25, random_state=1)


        # RANDOM FOREST
    def random_forest(self, X_train, y_train, X_test, y_test):
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        # prediction
        y_pred = rf_classifier.predict(X_test)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        # print evaluation
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", report)


    def rf_on_best_features(self, X_train, y_train, X_test, y_test, data):
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



    def rf_on_pca(self, data):
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



    # RANDOM FOREST ON T-SNE data

    def rf_on_tsne(self, data):
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



if __name__ == '__main__':
    print("Happy data preprocessing and modeling!")
    # rf_on_tsne()
    # rf_on_pca()
    # rf_on_best_features()
    # random_forest(X_train, y_train, X_test, y_test)
    data_modeling = data_modeling()