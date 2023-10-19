import numpy as np
import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

class data_modeling:

    def __init__(self) -> None:
        # Load your dataset
        self.data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")  # Replace with your dataset file path

        # Split the data into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target_Graduate']],
        self.data['Target_Graduate'], test_size=0.25, random_state=1)


        # RANDOM FOREST
    def random_forest(self, X_train, y_train, X_test, y_test):
        rf_classifier = RandomForestClassifier(max_depth=9, max_features=5, min_samples_leaf=1, n_estimators=400, random_state=42)

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

        param_grid = {'n_estimators': [100, 200, 300, 400, 500],
              'max_depth': [5, 9, 13, 17], 
              'min_samples_leaf': [1, 2, 4, 6, 8],
              'max_features': [3, 5, 7, 9, 11, 13]}
        
        # grid = GridSearchCV(SVC(), param_dist, refit=True, verbose=1, cv=5)
        # grid.fit(X_train, y_train)

        # best_params = grid.best_params_
        # print(f"Best params: {best_params}")

        # Create a based model
        rf = RandomForestRegressor()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)

        # Fit the random search object to the data
        grid_search.fit(X_train, y_train)
        # Create a variable for the best model
        best_rf = grid_search.best_estimator_

        # Print the best hyperparameters
        print('Best hyperparameters:',  grid_search.best_params_)
        print('Best model:', best_rf)

    #This code is based on the svm code found at https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/
    def svm(self, X_train, Y_train, X_test, Y_test):
        # # Split the data into training and test sets
        training_set, test_set = train_test_split(self.data, test_size=0.25, random_state=1)

        #standardize data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #Hyperparameter tuning, from https://www.kaggle.com/code/faressayah/support-vector-machine-pca-tutorial-for-beginner
        param_grid = {'C': [10], 
                    'gamma': [0.01], 
                    'kernel': ['rbf']} 

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


if __name__ == '__main__':
    print("Happy data preprocessing and modeling!")

    data_modeling = data_modeling()
    data_modeling.random_forest(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)
    # data_modeling.svm(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)