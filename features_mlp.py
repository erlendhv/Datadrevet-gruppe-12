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


def test_optimal_num_features(max_features=2, max_iterations=3000, runs=5, cv_folds=5, save_to_file=True):

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

        if save_to_file:
            save_num_features(feature_results)

    return feature_results

def save_num_features(feature_results, save_to_file="MLP_num_features_with_result"):
    df = pd.DataFrame(feature_results, columns=['Number of Features', 'Accuracy', 'F1 Score'])

    # Save the DataFrame to a CSV file if a filename is provided
    if save_to_file:
        df.to_csv(save_to_file, index=False)

def plot_num_features(feature_results):

    # Extract data for plotting
    num_features = [result[0] for result in feature_results]
    acc_scores = [result[1] for result in feature_results]
    f1_scores = [result[2] for result in feature_results]

    # Create subplots for acc and f1
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(num_features, acc_scores, marker='o', linestyle='-', color="r")
    plt.title('Accuracy')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')

    plt.xticks(num_features)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(num_features, f1_scores, marker='o', linestyle='-', color="b")
    plt.title('F1 Score')
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')
    
    plt.xticks(num_features)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Call the plot function

feature_res = test_optimal_num_features()
save_num_features(feature_res)
plot_num_features(feature_res)

