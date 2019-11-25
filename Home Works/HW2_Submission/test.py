# Load libraries
import pandas as pd
import numpy as np
from id3 import Id3Estimator
from id3 import export_graphviz
import pydot


col_names = ['Windy', 'AirQualityGood', 'Hot', 'PlayTennis']
# load dataset
pima = pd.read_csv("C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW2_Submission/Tennis.csv")
print(pima)
pima['Windy'] = np.where(pima['Windy'] == 'Yes', 1, 0)
pima['AirQualityGood'] = np.where(pima['AirQualityGood'] == 'Yes', 1, 0)
pima['Hot'] = np.where(pima['Hot'] == 'Yes', 1, 0)
pima['PlayTennis'] = np.where(pima['PlayTennis'] == 'Yes', 1, 0)

print(pima)
#split dataset in features and target variable
feature_cols = ['Windy', 'AirQualityGood', 'Hot']
X = pima[feature_cols] # Features
y = pima.PlayTennis # Target variable


estimator = Id3Estimator()
estimator = estimator.fit(X, y)
export_graphviz(estimator.tree_, 'tree.dot', feature_cols)


(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('ID3.png')
