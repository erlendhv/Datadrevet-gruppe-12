

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

class data_modeling:

    def __init__(self) -> None:
        # Load your dataset
        self.data = pd.read_csv("graduation_dataset_preprocessed_feature_selected.csv")  # Replace with your dataset file path

        # Split the data into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target_Graduate']],
        self.data['Target_Graduate'], test_size=0.25, random_state=1)


    def mpl(self, X_train, Y_train, X_test, Y_test,maxIterations,tune=False):
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
            best_params={'alpha': 0.1, 'hidden_layer_sizes': (100,),'activation': 'logistic'}
       
        # Train the SVM classifier
        mlp = MLPClassifier(max_iter=maxIterations, **best_params)
        mlp.fit(X_train, Y_train)

        # #predict the test set
        predictions = mlp.predict(X_test)
        # #print the metrics
        self.metrics("MLP", Y_test, predictions)
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
    data_modeling = data_modeling()
    #data_modeling.random_forest(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)
    #data_modeling.svm(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)

    #getting avg accuracy
    max_iterations=2000
    number_of_runs=2

    runs=[data_modeling.mpl(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test,max_iterations,True) for i in range(number_of_runs)]
    print(f"Average accuracy for MLP after {number_of_runs} run{'s'*min(number_of_runs-1,1)}: {sum(runs)/number_of_runs}")
