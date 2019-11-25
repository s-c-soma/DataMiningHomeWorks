import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



filename = 'C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW4_Submission/wine.csv'
data = pd.read_csv(filename)
print(data)

#Checking if there is any existing null value or not
print("Per column number of null value:", data.isnull().sum())

#Count the unique values in "Class"
print('Member per class:',data["Class"].value_counts())

#Plot for quality
data["Class"].value_counts().plot.bar(color='Green')
plt.xlabel("Class count")
plt.legend()
plt.show()

#Checking the dimensions
print('Dimensions:', data.shape)

# distributing the dataset into two components X and Y (target)
X = data.iloc[:, 1:13].values
y = data.iloc[:, 0].values # target column = class

print(X)

print(y)

#split into training and test dataset
X_train=X
y_train=y

# Preprocessing: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print('explained_variance:',explained_variance)
print('pca.n_components_', pca.n_components_)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance: Using Library')
plt.show()


#Cumulative sum
print('Cumulative sum',np.cumsum(pca.explained_variance_ratio_ *100))

data["PC1"] = X_train[:,0]
data["PC2"] = X_train[:,1]
sns.lmplot(data = data[["PC1","PC2","Class"]], x = "PC1", y = "PC2",fit_reg=False, hue = "Class",\
                 size = 6, aspect=1.5, scatter_kws = {'s':50}, )
plt.show()




