from ctgan import CTGAN
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sdv

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from data_modeling import plot_learning_curve


# Load your dataset from the CSV file
real_data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")

# discrete_columns = [
#     "Marital status",
#     "Application mode",
#     "Course",
#     "Daytime/evening attendance",
#     "Previous qualification",
#     "Father's qualification",
#     "Mother's occupation",
#     "Father's occupation",
#     "Debtor",
#     "Tuition fees up to date",
#     "Gender",
#     "Scholarship holder",
#     "Age at enrollment",
#     "International",
#     "Curricular units 1st sem (enrolled)",
#     "Curricular units 1st sem (evaluations)",
#     "Curricular units 1st sem (approved)",
#     "Curricular units 1st sem (grade)",
#     "Curricular units 2nd sem (credited)",
#     "Curricular units 2nd sem (enrolled)",
#     "Curricular units 2nd sem (evaluations)",
#     "Curricular units 2nd sem (approved)",
#     "Curricular units 2nd sem (grade)",
#     "units_approved_rate_1st",
#     "units_approved_rate_2nd",
#     "Target_Graduate"
# ]

# Fit the CTGAN model to your data
ctgan = CTGAN(verbose = True)
ctgan.fit(real_data)

# Generate synthetic data
num_samples = 1000  # You can change this value as needed
synthetic_data = ctgan.sample(num_samples)

# Save the synthetic data to a new CSV file
synthetic_data.to_csv('synthetic_data.csv', index=False)
    
gan_data = pd.read_csv("synthetic_data.csv")  

# # Merge real and generated data
data = pd.concat([real_data, gan_data])

X_train, X_test, y_train, y_test =  train_test_split(data[data.columns[data.columns != 'Target_Graduate']],
        data['Target_Graduate'], test_size=0.25, random_state=1)

def metrics( modelStr, Y_test, Y_pred):

        print("\n---------")
        print(f"\nMetrics for {modelStr}:\n")

        # Define metrics
        class_report = classification_report(Y_test, Y_pred)
        acc_score = accuracy_score(Y_test, Y_pred)
        conf_matrix = confusion_matrix(list(Y_test), Y_pred)
        

        # Print metrics
        print(class_report)
        print(f"Accuracy for {modelStr}: \n{acc_score}")
        print(f"Confusion matrix for {modelStr}: \n{conf_matrix}\n")


        # RANDOM FOREST
def random_forest( X_train, y_train, X_test, y_test, tune = False, learning_curve = False):
        rf_classifier = RandomForestClassifier(max_depth=17, max_features=3, min_samples_leaf=2, n_estimators=300, random_state=42)

        rf_classifier.fit(X_train, y_train)
        # prediction
        y_pred = rf_classifier.predict(X_test)

        if tune:
            param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [5, 9, 13, 17], 
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': [3, 5, 7, 9, 11, 13]}
            
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
        
        if learning_curve:
            cv = 5
            title = "Learning Curves (Random Forest)"
            plot_learning_curve(rf_classifier, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
            plt.show()


        metrics("Random Forest Classifier", y_test, y_pred)
        

random_forest(X_train, y_train, X_test, y_test)
