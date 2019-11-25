import csv
import random
import math
import pandas as pd
import numpy as np


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    dataset.pop(0)
    print(dataset)
    print(len(dataset))
    for i in range(len(dataset)):
        if(dataset[i][11]  == 'malignant'): dataset[i][11] = '1'
        elif dataset[i][11] == 'benign': dataset[i][11] = '0'

    for i in range(len(dataset)):
        if(dataset[i][11]  == 'malignant'): dataset[i][11] = '1'
        elif dataset[i][11] == 'benign': dataset[i][11] = '0'

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]

    x_train = []
    for j in range(len(dataset)):
        column = []
        for i in range(10):
            column.append(0)
        x_train.append(column)
# taking only required columns, dropping ID column
    for i in range(len(dataset)):
        for j in range(10):
            x_train[i][j] = dataset[i][j+2]
    print(x_train)
    print(dataset)
    #print(dataset)
    return x_train


def train_test_split(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def train_test_split_byclass (dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def class_summary(dataset):
    separated = train_test_split_byclass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def getClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = getClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictionValues(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct_guess = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct_guess += 1
    return (correct_guess / float(len(testSet))) * 100.0


def main():
    filename = 'C:/Users/subar/Downloads/CMPE-255 Sec 99 - Data Mining/Home Works/MyData.csv'
    splitRatio = 0.7
    dataset = loadCsv(filename)
    trainingSet, testSet = train_test_split(dataset, splitRatio)
    print("Split {0} rows into train={1} and test={2} rows".format(len(dataset), len(trainingSet), len(testSet)))

    # prepare model
    summaries = class_summary(trainingSet)
    # test model
    predictions = getPredictionValues(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy: {0}%".format(accuracy))

main()