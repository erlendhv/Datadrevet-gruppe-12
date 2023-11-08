import numpy as np
import pandas as pd
from ast import If
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split

class Preprocessing:

    def __init__(self) -> None:
        # Load your dataset
        self.data = pd.read_csv("graduation_dataset.csv")  # Replace with your dataset file path

    def one_hot_encoding(self):
        self.data = pd.get_dummies(self.data, columns=['Target']) 
        self.data = self.data.astype(int) #endrer true og false til 1 og 0
        self.data.drop('Target_Enrolled', inplace=True, axis=1)
        self.data.drop('Target_Dropout', inplace=True, axis=1)
        
    def standarize(self):
        scaler = MinMaxScaler()
        self.data.iloc[:, :] = scaler.fit_transform(self.data.iloc[:, :])    
        
    def describe(self):
        print(self.data.head())
        print(self.data.describe())

    def heatmap(self):
        # Removing the last column
        self.data = self.data.iloc[:, :-1]
        f = plt.figure(figsize=(15, 15))
        sns.heatmap(self.data.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()


    # BEST FEATURE SELECTION
    def best_feature_selection(self, plot=False):
        mutual_info = mutual_info_classif(self.X_train, self.y_train)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = self.X_train.columns
        if plot:
            mutual_info.sort_values(ascending=False).plot.bar(figsize=(15, 5))
            plt.show()
        return mutual_info


    #T-SNE
    def t_sne(self):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        # Initialize t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        # Fit and transform the data
        tsne_result = tsne.fit_transform(scaled_data)
        # Create a scatter plot of the t-SNE results

        return tsne_result

    def plot_t_sne(self, tsne):
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne[:, 0], tsne[:, 1], c=self.data['Target'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
        plt.colorbar(label='Target')
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()

    # PCA
    def pca(self, plot=False, n_components=2):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        # Initialize PCA with the number of components you want to retain
        # n_components = 2  # You can adjust this based on your needs
        pca = PCA(n_components=n_components)
        # Fit PCA to the standardized data
        pca_result = pca.fit_transform(scaled_data)
        # Create a scatter plot of the PCA results
        if plot:
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.data['Target'], cmap=plt.cm.get_cmap("viridis"), alpha=0.5)
            plt.colorbar(label='Target')
            plt.title('PCA Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()
        
        return pca_result


if __name__ == "__main__":
    print("Happy data preprocessing and modeling!")
    preprocessing = Preprocessing()
    preprocessing.one_hot_encoding()
    #preprocessing.standarize()
    preprocessing.data.to_csv("graduation_dataset_preprocessed.csv")
