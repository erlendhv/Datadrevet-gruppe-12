from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier

import data_modeling
class StackingEnsemble:
    def __init__(self, X_train, y_train, X_test, y_test, rf_model, svm_model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.rf_model = rf_model
        self.svm_model = svm_model
        #self.bagging_clf = BaggingClassifier(base_estimator=self.rf_model, n_estimators=10, random_state=42)

    def train_ensemble(self):
        # Define base models
        # base_models = [
        #     ('rf', RandomForestClassifier(max_depth=17, max_features=3, min_samples_leaf=2, n_estimators=300, random_state=42)),
        #     ('svm', SVC(probability=True, random_state=42)),
        #     ('log_reg', LogisticRegression(random_state=42))
        # ]
        #self.bagging_clf.fit(self.X_train, self.y_train)
        base_models = [
            ('rf', self.rf_model),
            ('svm', self.svm_model),
            ('log_reg', LogisticRegression(max_iter=1000, random_state=42))

        ]
        # Define meta-model
        # meta_model = LogisticRegression(random_state=42)

        # # Create and fit the stacking classifier
        # self.stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
        # self.stacking_clf.fit(self.X_train, self.y_train)
        meta_model = LogisticRegression(random_state=42)

        # Create and fit the stacking classifier
        self.stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
        self.stacking_clf.fit(self.X_train, self.y_train)

    def evaluate_ensemble(self):
        # Make predictions
        y_pred = self.stacking_clf.predict(self.X_test)

        # Evaluate the performance
        self.metrics("Stacking Classifier", self.y_test, y_pred)
        #bagging_y_pred = self.bagging_clf.predict(self.X_test)
        #self.metrics("Bagging Classifier", self.y_test, bagging_y_pred)

    def metrics(self, model_str, y_test, y_pred):
        # Define metrics
        class_report = classification_report(y_test, y_pred)
        acc_score = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print metrics
        print(f"\nMetrics for {model_str}:\n")
        print(class_report)
        print(f"Accuracy for {model_str}: \n{acc_score}")
        print(f"Confusion matrix for {model_str}: \n{conf_matrix}\n")

if __name__ == '__main__':
    # Assume data_modeling is an instance of your data_modeling class
    # data_modeling = data_modeling.data_modeling()
    # stacking_ensemble = StackingEnsemble(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)
    # stacking_ensemble.train_ensemble()
    # stacking_ensemble.evaluate_ensemble()
    data_modeling = data_modeling.data_modeling()
    # Train the Random Forest and SVM models in data_modeling
    data_modeling.random_forest(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)
    data_modeling.svm(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test)
    # Create an instance of StackingEnsemble, providing the trained Random Forest and SVM models
    stacking_ensemble = StackingEnsemble(data_modeling.X_train, data_modeling.y_train, data_modeling.X_test, data_modeling.y_test, data_modeling.rf_classifier, data_modeling.classifier)
    stacking_ensemble.train_ensemble()
    stacking_ensemble.evaluate_ensemble()