import csv
import numpy as np
from sklearn import tree as tr
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class DesisionTree():

    def __init__(self):
        self.train_set = []
        self.test_set = []
        self.x_train = self.y_train = []
        self.x_test = self.y_test = []

        with open("train.csv", 'r') as csvfile1:
            train_file = csv.reader(csvfile1)
            next(train_file, None)
            for row in train_file:
                self.train_set.append(row)
        with open("test.csv", 'r') as csvfile2:
            test_file = csv.reader(csvfile2)
            next(test_file, None)
            for row in test_file:
                self.test_set.append(row)
        self.train_set = np.array(self.train_set)
        self.test_set = np.array(self.test_set)

        self.x_train, self.y_train = self.extract_cols(self.train_set)
        self.x_test, self.y_test = self.extract_cols(self.test_set)
        self.train_set_len = len(self.train_set)
        self.test_set_len = len(self.test_set)

    def extract_cols(self, array):
        x = []
        y = []
        for row in array:
            x.append(row[0:len(row) - 1])
            y.append(row[-1])
        return x, y

    def ex1(self, min_sample_leaf=1):

        clf = tr.DecisionTreeClassifier(min_sample_leaf)
        clf = clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)
        mat = confusion_matrix(self.y_test, y_pred)
        return mat, clf
