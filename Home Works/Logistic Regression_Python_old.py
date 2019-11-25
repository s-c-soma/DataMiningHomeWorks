import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# reading csv file
data = pd.read_csv("C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/MyData.csv")

# Preview the first 5 lines of the loaded data
data.head()

# remove id column
data = data.drop('Id', 1)
#data = data.drop(data.columns[[0]], axis=1, inplace=True)
print(data)
# convert to numeric
for x in range(2, 10):
    # convert column "x" of a DataFrame
    data.iloc[:, x] = pd.to_numeric(data.iloc[:, x])

# Change Y values to 1's and 0's
data['Class'] = np.where(data['Class'] == 'malignant', 1, 0)
data['Class'] = data['Class'].astype('category')
print(data)

# Prep Training and Test data.
#library(caret)
#'%ni%' <- Negate('%in%')  # define 'not in' func
#options(scipen=999)  # prevents printing scientific notations.
#set.seed(100)
#trainDataIndex <- createDataPartition(bc$Class, p=0.7, list = F)
#trainData <- bc[trainDataIndex, ]
#testData <- bc[-trainDataIndex, ]
trainData, testData = train_test_split(data, train_size=0.7, random_state=100)
print(len(data))
print(len(trainData))
print(len(testData))

#print(trainData.describe())

# Class distribution of train data
print(data.Class.value_counts())
print(trainData.groupby('Class').size())
# Down Sample
#set.seed(100)
#down_train <- downSample(x = trainData[, colnames(trainData) %ni% "Class"], y = trainData$Class)

# Separate majority and minority classes
trainData_majority = trainData[trainData.Class == 0]
trainData_minority = trainData[trainData.Class == 1]
# Downsample majority class
trainData_downsampled = resample(trainData_majority,
                                 replace=False,    # sample without replacement
                                 n_samples=164,     # to match minority class
                                 random_state=100) # reproducible results
# Combine minority class with downsampled majority class
down_train = pd.concat([trainData_downsampled, trainData_minority])
print(down_train.groupby('Class').size())
# Display new class counts
print('downsampleo', down_train.Class.value_counts())


# Build Logistic Model
#logitmod <- glm(Class ~ Cl.thickness + Cell.size + Cell.shape, family = "binomial", data=down_train)


y_train = down_train.Class
#X_train = down_train.drop('Class', axis=1)
donw_train_limited_col = down_train[['Cl.thickness' , 'Cell.size' , 'Cell.shape']]
print(down_train)
print(donw_train_limited_col)
X_train = donw_train_limited_col
y_test = testData.Class
X_test = testData[['Cl.thickness' , 'Cell.size' , 'Cell.shape']] #testData.drop('Class', axis=1)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#predictions = logmodel.predict(X_test)
predictions = logmodel.predict(X_test)

#print(accuracy_score(y_train, predictions))
print(accuracy_score(y_test, predictions))