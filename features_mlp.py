import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from MLP import MLP
from data_preprocessing import Preprocessing
from sklearn.metrics import accuracy_score, f1_score


def test_optimal_num_features(max_features=35, max_iterations=3000, runs=5, cv_folds=5):

    feature_results = []

    # For every possible number of features
    for num_features in range(1, max_features+1):

        print(f"\n Testing number of features: {num_features}\n")
        # Generate dataset with right number of features
        preproc = Preprocessing()
        preproc.generate_dataset_mlp(num_features)

        # Generate model instance using these features
        model = MLP()
        pred = model.mlp(max_iterations, verbose=False)[0]

        acc_score = accuracy_score(model.y_test, pred)
        f1 = f1_score(model.y_test, pred)

        feature_results.append((num_features, acc_score, f1))

    return feature_results

def plot_num_features(feature_results):

    # Extract data for plotting
    num_features = [result[0] for result in feature_results]
    acc_scores = [result[1] for result in feature_results]
    f1_scores = [result[2] for result in feature_results]

    # Create subplots for acc and f1
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(num_features, acc_scores, marker='o', linestyle='-')
    plt.title('Accuracy vs. Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(num_features, f1_scores, marker='o', linestyle='-')
    plt.title('F1 Score vs. Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_num_features(test_optimal_num_features())

