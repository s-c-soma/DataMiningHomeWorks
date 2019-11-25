import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# reading csv file
data = pd.read_csv("MyData.csv")

# remove id column
data = data.drop('Id', 1)
print(data)

# convert to numeric
for x in range(2, 10):
    data.iloc[:, x] = pd.to_numeric(data.iloc[:, x])

# Change Y values to 1's and 0's
data['Class'] = np.where(data['Class'] == 'malignant', 1, 0)
data['Class'] = data['Class'].astype('category')
print(data)

# Prep Training and Test data.
trainData, testData = train_test_split(data, train_size=0.7, random_state=100)
print('Total data count:', len(data))
print('Training data count:',len(trainData))
print('Test data count:',len(testData))


# Class distribution of train data
print('Class distribution of training data:',trainData.Class.value_counts())
#print(trainData.groupby('Class').size())

# Separate majority and minority classes
trainData_majority = trainData[trainData.Class == 0]
trainData_minority = trainData[trainData.Class == 1]


# Downsample majority class
trainData_downsampled = resample(trainData_majority,
                                 replace=False,     # sample without replacement
                                 n_samples=print(len(trainData_minority)),     # to match minority class
                                 random_state=100)  # reproducible results

# Combine minority class with downsampled majority class
down_train = pd.concat([trainData_downsampled, trainData_minority])
#print(down_train.groupby('Class').size())
# Display new class counts
#print('downsample', down_train.Class.value_counts())


# Build Logistic Model with down sampled data
X_traindown = down_train[['Cl.thickness' , 'Cell.size' , 'Cell.shape']]
Y_traindown = down_train.Class
#print(down_train)
X_testdown = testData[['Cl.thickness' , 'Cell.size' , 'Cell.shape']]
Y_testdown = testData.Class
Downlogmodel = LogisticRegression()
Downlogmodel.fit(X_traindown,Y_traindown)
predictionsDown = Downlogmodel.predict(X_testdown)
# Accuracy calculation
print('Accuracy with down sampled data:',(accuracy_score(Y_testdown, predictionsDown) * 100) ,"%")



# Upsample majority class
trainData_upsampled = resample(trainData_minority,
                                 replace=True,     # sample without replacement
                                 n_samples=print(len(trainData_majority)),     # to match majority class
                                 random_state=100)  # reproducible results

# Combine minority class with upsampled majority class
up_train = pd.concat([trainData_upsampled, trainData_majority])
#print(up_train.groupby('Class').size())
# Display new class counts
#print('upsample:', up_train.Class.value_counts())


# Build Logistic Model with upsampled data
X_trainup = up_train[['Cl.thickness' , 'Cell.size' , 'Cell.shape']]
Y_trainup = up_train.Class
#print(up_train)
X_testup = testData[['Cl.thickness' , 'Cell.size' , 'Cell.shape']]
Y_testup = testData.Class
Uplogmodel = LogisticRegression()
Uplogmodel.fit(X_trainup,Y_trainup)
predictionsUp = Uplogmodel.predict(X_testup)


# Accuracy calculation
print('Accuracy with up sampled data:',(accuracy_score(Y_testup, predictionsUp) * 100) ,"%")