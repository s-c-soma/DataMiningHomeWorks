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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Preprocessing: Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print('explained_variance:',explained_variance)
print('pca.n_components_', pca.n_components_)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_ *100))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance: Using Library')
plt.show()

#Cumulative sum
print('Cumulative sum',np.cumsum(pca.explained_variance_ratio_ *100))

'''
pca = PCA()
X_train = pca.fit_transform(X)
pca1 = X_train[:, 0]
pca2 = X_train[:, 1]
#pca vs variance
fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.bar(np.arange(pca.n_components_), 100*pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance % ")
plt.show()

#
fig.add_subplot(1,2,2)
plt.scatter(pca1, pca2, c=y, marker='x', cmap='jet')
plt.title('Class distributions')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
'''


# Fitting Logistic Regression To the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set result using
# predict function under LogisticRegression
y_pred = classifier.predict(X_test)

print("The accuracy of  LogisticRegression is:",(accuracy_score(y_test, y_pred) * 100) ,"%")

# making confusion matrix between
#  test set of Y and predicted value.
cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix',cm)

# Predicting the training set
# result through scatter plot
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c=ListedColormap(('red', 'green', 'blue'))(i), label=j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')  # for Xlabel
plt.ylabel('PC2')  # for Ylabel
plt.legend()  # to show legend

# show scatter plot
plt.show()

# Visualising the Test set results through scatter plot
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                  X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)

# title for scatter plot
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')  # for Xlabel
plt.ylabel('PC2')  # for Ylabel
plt.legend()

# show scatter plot
plt.show()

print(data.columns)
