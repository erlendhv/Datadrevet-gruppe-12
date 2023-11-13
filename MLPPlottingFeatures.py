

import numpy as np
import pandas as pd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from features import select_best

class data_modeling:

    def __init__(self) -> None:
        # Load your dataset
        self.data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")  # Replace with your dataset file path

        # Split the data into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target_Graduate']],
        self.data['Target_Graduate'], test_size=0.25, random_state=1)


        # RANDOM FOREST
    def random_forest(self, X_train, y_train, X_test, y_test, tune = False, learning_curve = False):
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


        self.metrics("Random Forest Classifier", y_test, y_pred)

    #This code is based on the svm code found at https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/
    def svm(self, X_train, Y_train, X_test, Y_test, tune = False, learning_curve = False):
        # # Split the data into training and test sets
        training_set, test_set = train_test_split(self.data, test_size=0.25, random_state=1)

        #standardize data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        #Hyperparameter tuning, from https://www.kaggle.com/code/faressayah/support-vector-machine-pca-tutorial-for-beginner

        print("\n---------")
        print("\nHyperparameter tuning for SVM:\n")

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
            'gamma': [10, 1, 0.5, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['rbf', 'poly', 'linear', 'sigmoid']} if tune else {'C': [100], 
                    'gamma': [0.001], 
                    'kernel': ['rbf']} 


        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
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
            
        self.metrics("SVM", Y_test, Y_pred)

        if learning_curve:
            cv = 5
            title = "Learning Curves (SVM)"
            plot_learning_curve(classifier, title, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
            plt.show()
   
    def mpl(self, X_train, Y_train, X_test, Y_test,maxIterations,tune=False,bestParameter={'alpha': 0.1, 'hidden_layer_sizes': (100,),'activation': 'logistic'}):
        #standardize data
        #from the documentation https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test = scaler.transform(X_test)

        #hyperparameter tuning
        if tune:
            parameter_space = {
                'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)], #trying diffrent number of layers and difrrent neuron count
                'alpha': 10.0 ** -np.arange(1, 7),
                'activation': ['identity', 'logistic', 'tanh', 'relu']

            }
            grid = GridSearchCV(MLPClassifier(max_iter=maxIterations), parameter_space, n_jobs=-1)
            grid.fit(X_train, Y_train)

            best_params = grid.best_params_
            print(f"Best params: {best_params}")
        else:
            best_params=bestParameter
       
        # Train the SVM classifier
        mlp = MLPClassifier(max_iter=maxIterations, **best_params)
        mlp.fit(X_train, Y_train)

        # #predict the test set
        predictions = mlp.predict(X_test)
        # #print the metrics
        #self.metrics("MLP", Y_test, predictions)
        #return F1 sc
        return accuracy_score(Y_test, predictions)

   
    def metrics(self, modelStr, Y_test, Y_pred):

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

        # Display metrics
        # disp = ConfusionMatrixDisplay(conf_matrix)
        # disp.plot()
        # plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r",
            label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g",
            label="Cross-validation score")
    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    print("Happy data preprocessing and modeling!")
    y=[]
    x=list(range(1,35))
    max_iterations=5000
    number_of_runs=3
    best_parameters=[{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 1e-06, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100,)},{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,)},{'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,)},{'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'tanh', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'identity', 'alpha': 1e-06, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 1e-06, 'hidden_layer_sizes': (100,)},{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 0.0001, 'hidden_layer_sizes': (100,)},{'activation': 'logistic', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (100,)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50, 50)},{'activation': 'identity', 'alpha': 0.1, 'hidden_layer_sizes': (50, 50, 50)}]


    for i in range(1,35):
        select_best(i)
        datamodeling = data_modeling()
        runs=[]
        #getting avg accuracy
        runs=[datamodeling.mpl(datamodeling.X_train, datamodeling.y_train, datamodeling.X_test, datamodeling.y_test,max_iterations,True,best_parameters[i-1]) for i in range(number_of_runs)]
        print(f"Average F1 for MLP after {number_of_runs} run{'s'*min(number_of_runs-1,1)}: {sum(runs)/number_of_runs}")
        y+=[sum(runs)/number_of_runs]
        print(f"Iteration {i}/34")
    print(y)

    plt.plot(x,y)
    plt.xlabel("Number of features")
    plt.ylabel("F1 score")
    plt.show()
