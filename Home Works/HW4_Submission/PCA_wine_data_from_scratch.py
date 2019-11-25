import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

columns=['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash','Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols',
       'Proanthocyanins', 'Color_intensity', 'Hue','OD280/OD315of_diluted_wines', 'Proline ']
filename = 'C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW4_Submission/wine.csv'
data = pd.read_csv(filename)
print(data)

#Checking if there is any existing null value or not
print("Per column number of null value:", data.isnull().sum())

#Count the unique values in "Class"
print('Member per class:',data["Class"].value_counts())

#Plot for class
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
X_train= X
y_train= y


# Preprocessing: Feature Scaling-standardizing the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#Calculate covariance matrix
mean_vec = np.mean(X_train, axis=0)
cov_matrix = ( ((X_train - mean_vec).T).dot(X_train - mean_vec) ) / (X_train.shape[0]-1)
print('Covariance matrix \n%s' %cov_matrix)
print('NumPy covariance matrix: \n%s' %np.cov(X_train.T))

#calculate eigen values and eigen vectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
print('\nEigenvalues: \n%s' %eigen_vals)
print('Eigenvectors: \n%s' %eigen_vecs)


# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort()
eigen_pairs.reverse()

print('Eigenvalues in descending order:')
for i in eigen_pairs:
    print(i[0]) #print eigen values

#variance calculation
total = sum(eigen_vals)
print('Sum of eigen values:', total)
explained_variance = [(i / total)*100 for i in sorted(eigen_vals, reverse=True)]
print('explained_variance:', explained_variance)

#Cumulative sum
print('Cumulative sum',np.cumsum(explained_variance))

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance: From Scratch')
plt.show()

#converting eigen pairs to matrix
matrix_w = np.hstack((eigen_pairs[0][1].reshape(12,1),
                      eigen_pairs[1][1].reshape(12,1)
                      #eigen_pairs[2][1].reshape(12,1) #if 3 pc required
                      ))
print('Matrix W:\n', matrix_w)

#projection to new feature space by taking dot product of matrix
X_train_new = X_train.dot(matrix_w)
print('X_train_new: \n',X_train_new)
# taking first and second pca
pca1 = X_train_new[:, 0]
pca2 = X_train_new[:, 1]
data["PC1"] = X_train_new[:,0]
data["PC2"] = X_train_new[:,1]


#pca vs variance
fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.bar(np.arange(12), explained_variance)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance % ")
plt.show()

#Class distributions
fig.add_subplot(1,2,2)
plt.scatter(pca1, pca2, c=y_train, marker='x', cmap='jet')
plt.title('Class distributions')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Distribution checking of new data
sns.lmplot(data = data[["PC1","PC2","Class"]], x = "PC1", y = "PC2",fit_reg=False, hue = "Class",\
                 size = 6, aspect=1.5, scatter_kws = {'s':50}, )
plt.show()



#Now verifying using scikit learn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_new = pca.fit_transform(X_train)
print('X_new: \n',X_new)
data["PC1"] = X_new[:,0]
data["PC2"] = X_new[:,1]
sns.lmplot(data = data[["PC1","PC2","Class"]], x = "PC1", y = "PC2",fit_reg=False, hue = "Class",\
                 size = 6, aspect=1.5, scatter_kws = {'s':50}, )
plt.show()

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance: Using Library')
plt.show()