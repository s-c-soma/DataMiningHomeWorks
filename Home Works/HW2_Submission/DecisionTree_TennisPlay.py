# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


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

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X,y)



dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())