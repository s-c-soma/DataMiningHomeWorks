# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
import csv
from csv import reader
from math import sqrt
import pandas as pd
import numpy as np
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#get numerical data from cansas adult data set:
def getNumericalColumns(filepath):
    data = pd.read_csv(filepath)
    # Change Y values to 1's and 0's
    data['income'] = np.where(data['income'] == '>50K', 1, 0)
    data['income'] = data['income'].astype('int64')
    #print(data)
    data = data.select_dtypes(include=['int64'])
    # csv write the numerical data
    filepath = "C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW3_Submission/adult_census_income_numeric.csv"
    headers = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'income'] #, 'fnlwgt'
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(headers)
        for j in range(0, 100):
            writer.writerow([data['age'][j],  data['education.num'][j], data['capital.gain'][j],
                             data['capital.loss'][j], data['hours.per.week'][j], data['income'][j]]) #data['fnlwgt'][j],


# Load a CSV file
def loadCsv(filenameKnn):
    dataset = list()
    with open(filenameKnn, "r") as file:
        csv_reader = reader(file)
        #next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    print (dataset)

    return dataset


# Convert string column to numeric
def getStrToNumericVal(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into k sets
def train_test_split(dataset, k_val):
    dataset_split = list()
    dataset_copy = list(dataset)
    k_set_size = int(len(dataset) / k_val)
    for i in range(k_val):
        fold = list()
        while len(fold) < k_set_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Algorithm
def knn_algorithm(k_Set ,k_val,distMethods):
    print('k_Set',k_Set)
    scores = list()
    for k in k_Set:
        train_set = list(k_Set)
        train_set.remove(k)
        train_set = sum(train_set, [])
        test_set = list()
        for row in k:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = k_nearest_neighbors(train_set, test_set, k_val,distMethods)
        actual = [row[-1] for row in k]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate the Euclidean distance
def getEuclideanDistance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2

    #print(distance)
    return sqrt(distance)

#Calculate the Manhattan distance
def getManhattanDistance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i])
    #print(distance)
    return distance

#Calculate the cosine distance
def getCosineDistance(row1, row2):
    dist= 0.0
    for i in range(len(row1)-1):
        dist += distance.cosine(row1[i], row2[i], 2)
    return dist

#Calculate the Minkowski distance
def getMinkowskiDistance(row1, row2):
    dist= 0.0
    for i in range(len(row1)-1):
        dist += distance.minkowski(row1[i], row2[i], 2)
    return dist


#Calculate the Mahalanobis distance
def getMahalanobisDistance(row1, row2):
    dist= 0.0
    for i in range(len(row1)-1):
        dist += distance.mahalanobis(row1[i], row2[i], 2)
    return dist

# find the k nearest neighbours
def getNeighbors(train, test_row, k_val, distMethods):
    distances = list()
    for train_row in train:
        dist = distMethods(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k_val):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def getPredictedClass(train, test_row, k_val, distMethods):
    neighbors = getNeighbors(train, test_row, k_val, distMethods)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, k_val, distMethods):
    predictions = list()
    for row in test:
        output = getPredictedClass(train, row, k_val, distMethods)
        predictions.append(output)
    return (predictions)



# Test the kNN on adult_census_income dataset from kaggle
seed(1)
filepath = "C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW3_Submission/adult_census_income.csv"
filepathKnn="C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/HW3_Submission/adult_census_income_numeric.csv"
getNumericalColumns(filepath)
dataset = loadCsv(filepathKnn)
for i in range(len(dataset[0])):
    getStrToNumericVal(dataset, i)

# evaluate algorithm
k_Val = 5
splitRatio= 0.7
k_Set = train_test_split(dataset, k_Val)
distMethods= ['getEuclideanDistance', 'getManhattanDistance','getMinkowskiDistance']
#print(distMethods[0])

#calling KNN algo with different distance function
scores = knn_algorithm(k_Set, k_Val, getEuclideanDistance)
print('Scores with Euclidean Distance: %s' % scores)
print('Mean Accuracy with Euclidean Distance: %.3f%%' % (sum(scores)/float(len(scores))))

scores = knn_algorithm(k_Set, k_Val, getManhattanDistance)
print('Scores with Manhattan Distance: %s' % scores)
print('Mean Accuracy with Manhattan Distance: %.3f%%' % (sum(scores)/float(len(scores))))

scores = knn_algorithm(k_Set, k_Val, getMinkowskiDistance)
print('Scores with Minkowski Distance: %s' % scores)
print('Mean Accuracy with Minkowski Distance: %.3f%%' % (sum(scores)/float(len(scores))))

scores = knn_algorithm(k_Set, k_Val, getCosineDistance)
print('Scores with Cosine Distance: %s' % scores)
print('Mean Accuracy with Cosine Distance: %.3f%%' % (sum(scores)/float(len(scores))))

scores = knn_algorithm(k_Set, k_Val, getMahalanobisDistance)
print('Scores with Mahalanobis Distance: %s' % scores)
print('Mean Accuracy with Mahalanobis Distance: %.3f%%' % (sum(scores)/float(len(scores))))