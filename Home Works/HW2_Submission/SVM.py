# Load libraries
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

#Load data with only two classes
iris = datasets.load_iris()
X = iris.data[:100,:]
y = iris.target[:100]

print(X,y)

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create support vector classifier object
svc = SVC(kernel='linear', random_state=0)

# Train classifier
model = svc.fit(X_std, y)

# View support vectors
print(model.support_vectors_)

# View indices of support vectors
print(model.support_)

# View number of support vectors for each class
print(model.n_support_)

