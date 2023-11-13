

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
