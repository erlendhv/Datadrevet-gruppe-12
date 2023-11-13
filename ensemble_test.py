import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures


class Ensemble:
    def __init__(self):
        self.data = pd.read_csv('graduation_dataset.csv')

        """
        PREPROCESSING
        """

        self.data = pd.get_dummies(self.data, columns=['Target']) 
        self.data = self.data.astype(int) #endrer true og false til 1 og 0
        self.data.drop('Target_Enrolled', inplace=True, axis=1)
        self.data.drop('Target_Dropout', inplace=True, axis=1)

        # approved units rate for 1st and 2nd semester
        self.data['units_approved_rate_1st'] = self.data['Curricular units 1st sem (approved)']/self.data['Curricular units 1st sem (enrolled)']
        self.data['units_approved_rate_2nd'] = self.data['Curricular units 2nd sem (approved)']/self.data['Curricular units 2nd sem (enrolled)']
        # replace NaN with 0, get NaN from new rate features
        self.data.fillna(0, inplace=True)

        x = self.data.drop(['Target_Graduate'], axis=1)
        y = self.data['Target_Graduate']
        # using score function mutual information to capture complex relations
        sel_k_best = SelectKBest(k=21, score_func=mutual_info_classif)
        features = sel_k_best.fit_transform(x,y)
        # print(dict(zip(x.columns, sel_k_best.scores_)))
        best_features = [x.columns[feature] for feature in sel_k_best.get_support(indices=True)]

        self.data = self.data.drop([feature for feature in self.data.columns if feature not in best_features and feature!='Target_Graduate'], axis=1)

        # train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target_Graduate']],self.data['Target_Graduate'], test_size=0.25, random_state=1)
        # need to scale for SVM
        sc = StandardScaler()
        self.X_train_scaled = sc.fit_transform(self.X_train)
        self.X_test_scaled = sc.fit_transform(self.X_test)

    # Evaluate the performance
    # Define metrics
    def metrics(self,modelStr, Y_test, Y_pred):
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

    """
    RANDOM FOREST TUNING
    """
    def rf_tuning(self):
        param_grid = {'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 9, 13, 17], 
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': [3, 5, 7, 9, 11, 13]}
        # Create a based model
        rf = RandomForestClassifier()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)
        # Fit the random search object to the data
        grid_search.fit(self.X_train, self.y_train)
        # Create a variable for the best model
        best_rf = grid_search.best_estimator_
        # Print the best hyperparameters
        print('Best hyperparameters:',  grid_search.best_params_)
        print('Best model:', best_rf)

        # Best hyperparameters: {'max_depth': 9, 'max_features': 3, 'min_samples_leaf': 1, 'n_estimators': 300}
        # Best model: RandomForestClassifier(max_depth=9, max_features=3, n_estimators=300)


    """
    SVM TUNING
    """
    def svm(self):
        print("\n---------")
        print("\nHyperparameter tuning for SVM:\n")
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
            'gamma': [10, 1, 0.5, 0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['rbf']} # will always choose radial basis function
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, n_jobs=-1)
        grid.fit(self.X_train_scaled, self.y_train)
        best_params = grid.best_params_
        print(f"Best params: {best_params}")

        # Encode the class labels
        le = LabelEncoder()
        Y_train = le.fit_transform(self.y_train)
        # Train the SVM classifier
        classifier = SVC(**best_params)
        classifier.fit(self.X_train_scaled, Y_train)
        # Predict on the test set
        Y_pred = classifier.predict(self.X_test_scaled)
        self.metrics('SVM', self.y_test, Y_pred)


    """
    XGBOOST TUNING
    """
    def xgb(self):
        # Create an XGBoost classifier
        xgb_classifier = XGBClassifier()
        # Define a grid of hyperparameters to search
        param_grid = {
            'n_estimators': [100, 200, 300],  # Number of boosting rounds
            'max_depth': [3, 4, 5],  # Maximum depth of trees
            'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
            'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for training each tree
            'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for training each tree
            'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight (hessian) needed in a child
            'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node
            'scale_pos_weight': [1, 2, 3],  # Controls the balance of positive and negative weights
            # Add other hyperparameters as needed
        }
        # Create a GridSearchCV object with cross-validation (e.g., 5-fold cross-validation)
        grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        # Fit the grid search to your training data
        grid_search.fit(self.X_train, self.y_train)
        # Print the best hyperparameters found
        print("Best Hyperparameters for XGBoost:", grid_search.best_params_)
        # Get the best XGBoost classifier model
        best_xgb_classifier = grid_search.best_estimator_
        # Evaluate the model on your test set
        accuracy = best_xgb_classifier.score(self.X_test, self.y_test)
        print("Accuracy on Test Set:", accuracy)

        # Best Hyperparameters for XGBoost: {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 300, 'scale_pos_weight': 1, 'subsample': 0.9}
        # Accuracy on Test Set: 0.8616636528028933


    """
    STACKING TUNING
    """
    def stacking(self):
        base_models = [
            RandomForestClassifier(max_depth=9, max_features=3, n_estimators=300).fit(self.X_train, self.y_train),
            SVC(C=10, gamma=0.01, kernel='rbf').fit(self.X_train_scaled, self.y_train),
            XGBClassifier(colsample_bytree=1.0, gamma=0, learning_rate=0.1, max_depth=3, min_child_weight=5, n_estimators=300, scale_pos_weight=1, subsample=0.9).fit(self.X_train, self.y_train)
        ]

        # Initialize an empty list to store predictions from base models
        base_model_predictions = []
        # Make predictions using each base model
        for base_model in base_models:
            predictions = base_model.predict(self.X_train)
            base_model_predictions.append(predictions)

        # Convert the list of predictions into a matrix (each column is a base model's predictions)
        predictions_from_base_models = np.column_stack(base_model_predictions)

        self.tuned_base_models = [
            ('rf', RandomForestClassifier(max_depth=9, max_features=3, n_estimators=300)),
            ('svm', SVC(C=10, gamma=0.01, kernel='rbf')),
            ('xgb', XGBClassifier(colsample_bytree=1.0, gamma=0, learning_rate=0.1, max_depth=3, min_child_weight=5, n_estimators=300, scale_pos_weight=1, subsample=0.9))
        ]

        meta_model = LogisticRegression()
        # Grid search for meta-model hyperparameters
        param_grid_meta = {
            'C': [0.01, 0.1, 1.0],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            # Add other meta-model hyperparameters as needed
        }

        self.grid_search_meta = GridSearchCV(estimator=meta_model, param_grid=param_grid_meta, cv=5, n_jobs=-1, verbose=1)
        self.grid_search_meta.fit(predictions_from_base_models, self.y_train)  # `predictions_from_base_models` is the combined output of base model
        # Create and fit the stacking classifier
        stacking_clf = StackingClassifier(estimators=self.tuned_base_models, final_estimator=self.grid_search_meta.best_estimator_, cv=5)
        stacking_clf.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = stacking_clf.predict(self.X_test)
        self.metrics("Stacking Classifier", self.y_test, y_pred)

        rf = RandomForestClassifier(max_depth=9, max_features=3, n_estimators=300).fit(self.X_train, self.y_train)
        rf_pred = rf.predict(self.X_test)
        self.metrics('Random Forest', rf_pred, self.y_test)

        svm = SVC(C=10, gamma=0.01, kernel='rbf').fit(self.X_train_scaled, self.y_train)
        svm_pred = svm.predict(self.X_test_scaled)
        self.metrics('SVM', svm_pred, self.y_test)

        xgb = XGBClassifier(colsample_bytree=1.0, gamma=0, learning_rate=0.1, max_depth=3, min_child_weight=5, n_estimators=300, scale_pos_weight=1, subsample=0.9).fit(self.X_train, self.y_train)
        xgb_pred = xgb.predict(self.X_test)
        self.metrics('XGBoost', xgb_pred, self.y_test)

        """
        Currently best: 0.87432188, using 13 or 15 features. XGBoost gets the same performance. Large variations each time. Stacking not consistently better
        """
        return rf_pred, svm_pred, xgb_pred

    def two_layer_stack(self):
        models = [
            ('lr', LogisticRegression()), 
            ('svc', SVC()),
            ('gnb', GaussianNB()), 
            ('dtc', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier())
            ]

        stacked = StackingClassifier(estimators=models, final_estimator=LogisticRegression(), cv=5)
        stacked.fit(self.X_train, self.y_train)
        y_pred = stacked.predict(self.X_test)
        self.metrics("Stacked", self.y_test, y_pred)

        """
        Stacking in two layers
        """

        simple_stack_models = [
            ('lr', LogisticRegression()),
            ('dtr', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier()),
            ('nn', MLPClassifier())
        ]

        models_first_layer = [
            ('stacked', StackingClassifier(estimators=self.tuned_base_models, final_estimator=self.grid_search_meta.best_estimator_, cv=5)),
            ('xgb', XGBClassifier(colsample_bytree=1.0, gamma=0, learning_rate=0.1, max_depth=3, min_child_weight=5, n_estimators=300, scale_pos_weight=1, subsample=0.9)),
            ('stacked_simple', StackingClassifier(estimators=simple_stack_models, final_estimator=self.grid_search_meta.best_estimator_, cv=5))
        ]

        # Create and fit the stacking classifier
        second_layer = StackingClassifier(estimators=models_first_layer, final_estimator=self.grid_search_meta.best_estimator_, cv=5)
        second_layer.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = second_layer.predict(self.X_test)
        self.metrics("Stacking Classifier in two layers", self.y_test, y_pred)
        """
        Stacking in two layers does not improve performance compared to just one layer.

        In conclusion: XGBoost is most often best with a performance of 0.87432188. Stacking is not able to top this, and will only take longer time.
        """

    """
    See when base models predict different
    """
    def model_disagreements(self, rf_pred, svm_pred, xgb_pred):

        predictions_df = pd.DataFrame({
            'RandomForest': rf_pred,
            'SVC': svm_pred,
            'XGBoost': xgb_pred,
            'TrueLabel': self.y_test
        })

        predictions_df['Disagreement'] = predictions_df.apply(lambda x: len(set(x[:-2])) > 1, axis=1)

        disagreements = predictions_df[predictions_df['Disagreement']]

        all_disagreements = disagreements[(disagreements['RandomForest']!=disagreements['SVC']) | (disagreements['RandomForest']!=disagreements['XGBoost']) | (disagreements['SVC']!=disagreements['XGBoost'])]

        print(all_disagreements)
        print('Total number of disagreements is ' + str(len(all_disagreements)))
        print('The length of the test set is ' + str(len(self.y_test)))
        print('Fraction of disagreements is ' + str(len(all_disagreements)/len(self.y_test)))

if __name__ == '__main__':
    ensemble = Ensemble()
    rf_pred, svm_pred, xgb_pred = ensemble.stacking()
    ensemble.model_disagreements(rf_pred, svm_pred,xgb_pred)