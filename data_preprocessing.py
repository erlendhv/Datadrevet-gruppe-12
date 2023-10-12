import numpy as np
import pandas as pd
from ast import If
import pandas as pd
import matplotlib.pyplot as plt
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
        # one hot encoding target column to numeric values
        target_mapping = {
            "Dropout": 0,
            "Enrolled": 0, 
            "Graduate": 1
        }
        self.data.replace({"Target": target_mapping}, inplace=True)
        # Split the data into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.data.columns[self.data.columns != 'Target']],self.data['Target'], test_size=0.25, random_state=1)

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


# def kernel_pca():
#     # Standardize the data
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(data)
#     # initialize the Kernel PCA object
#     Kernel_pca = KernelPCA(n_components = 2, kernel= "rbf")# extracts 2 features, specify the kernel as rbf
#     # transform and fit the feature of the training set
#     pca_result = Kernel_pca.fit_transform(scaled_data)
#     X_train, X_test, y_train, y_test = train_test_split(pca_result, data["Target"], test_size=0.2, random_state=42)
#     # transform features of the test set
#     XTest = Kernel_pca.transform(X_test)
    
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_classifier.fit(X_train, y_train)

#     y_pred = rf_classifier.predict(XTest)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     confusion = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred)

#     print("Accuracy:", accuracy)
#     print("Confusion Matrix:\n", confusion)
#     print("Classification Report:\n", report)






if __name__ == "__main__":
    print("Happy data preprocessing and modeling!")
    preprocessing = Preprocessing()
    t_sne = preprocessing.pca(plot=True, n_components=5)