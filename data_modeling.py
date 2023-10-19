import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

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

        param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

        # Create a random forest classifier
        rf = RandomForestClassifier()

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(rf_classifier, 
                                        param_distributions = param_dist, 
                                        n_iter=5, 
                                        cv=5)

        # Fit the random search object to the data
        rand_search.fit(X_train, y_train)
        # Create a variable for the best model
        best_rf = rand_search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  rand_search.best_params_)
        print('Best model:', best_rf)



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

        rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=17)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", report)

        param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

        # Create a random forest classifier
        rf = RandomForestClassifier()

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(rf_classifier, 
                                        param_distributions = param_dist, 
                                        n_iter=5, 
                                        cv=5)

        # Fit the random search object to the data
        rand_search.fit(X_train, y_train)
        # Create a variable for the best model
        best_rf = rand_search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  rand_search.best_params_)
        print('Best model:', best_rf)




if __name__ == '__main__':
    print("Happy data preprocessing and modeling!")
    # rf_on_tsne()
    # rf_on_pca()
    # rf_on_best_features()
    # random_forest(X_train, y_train, X_test, y_test)
    data_modeling = data_modeling()
    # data_modeling.rf_on_best_features(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test, data_modeling.data)
    # data_modeling.random_forest(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)
    # data_modeling.rf_on_tsne(data_modeling.data)
    data_modeling.random_forest(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)