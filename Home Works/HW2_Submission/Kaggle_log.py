import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import os
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score

# reading csv file
data = pd.read_csv("C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/\HW2_Submission/adult_census_income.csv")

print(data)

# iterating the columns
for col in data.columns:
    print(col)

# Prep Training and Test data.
train_data, test_data = train_test_split(data, train_size=0.7, random_state=100)
print(train_data.info())

num_attributes = train_data.select_dtypes(include=['int64'])
print(num_attributes.columns)

num_attributes.hist(figsize=(10,10))
#plt.show()

print(train_data.describe())

cat_attributes = train_data.select_dtypes(include=['object'])
print(cat_attributes.columns)

sns.countplot(y='workclass', hue='income', data = cat_attributes)
#plt.show()
sns.countplot(y='occupation', hue='income', data = cat_attributes)
#plt.show()

class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type = type
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X.select_dtypes(include=[self.type])

num_pipeline = Pipeline(steps=[("num_attr_selector", ColumnsSelector(type='int64')),("scaler", StandardScaler())])

class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy
    def fit(self,X, y=None):
        if self.columns is None:
            self.columns = X.columns
        if self.strategy is 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
        else:
            self.fill ={column: '0' for column in self.columns}
        return self
    def transform(self,X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        return X_copy

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dropFirst=True):
        self.categories=dict()
        self.dropFirst=dropFirst
    def fit(self, X, y=None):
        join_df = pd.concat([train_data, test_data])
        join_df = join_df.select_dtypes(include=['object'])
        for column in join_df.columns:
            self.categories[column] = join_df[column].value_counts().index.tolist()
            return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column:CategoricalDtype(self.categories[column])})
        return pd.get_dummies(X_copy, drop_first=self.dropFirst)

cat_pipeline = Pipeline(steps=[
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=['workClass','occupation', 'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
    ])

full_pipeline = FeatureUnion([("num_pipe", num_pipeline),("cat_pipeline", cat_pipeline)])

train_data.drop(['fnlwgt', 'education'], axis=1)
test_data.drop(['fnlwgt', 'education'], axis=1)

train_copy = train_data.copy()
train_copy["income"] = train_copy["income"].apply(lambda x:0 if x=='<=50K' else 1)
X_train = train_copy.drop('income', axis =1)
Y_train = train_copy['income']

X_train_processed=full_pipeline.fit_transform(X_train)
model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)