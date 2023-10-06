import numpy as np
import pandas as pd
from ast import If
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Datadrevet-gruppe-12/graduation_dataset.csv')

print(dataset.head(5))

# Removing the last column
dataset = dataset.iloc[:, :-1]

f = plt.figure(figsize=(15, 15))
sns.heatmap(dataset.corr(),annot=False, cmap='RdBu',vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()