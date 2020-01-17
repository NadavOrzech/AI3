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

    def ex8(self):
        # balancing the training set to hold equal number
        # of true samples and negative samples
        T = 0
        for i in self.y_train:          # counting number of positive samples
            if i == "1":
                T += 1

        x_train_new = []
        y_train_new = []
        for index in range(len(self.y_train)):
            if self.y_train[index] == "1":
                x_train_new.append(self.x_train[index])
                y_train_new.append("1")
            elif T > 0:
                x_train_new.append(self.x_train[index])
                y_train_new.append("0")
                T -= 1
        # print(x_train_new)
        # print(y_train_new)
        # Building a decision tree using the balanced training set
        clf = tr.DecisionTreeClassifier()
        clf = clf.fit(x_train_new, y_train_new)

        y_pred = clf.predict(self.x_test)
        # mat = confusion_matrix(self.y_test, y_pred)
        self.printMatrix(self.y_test, y_pred)

    def printMatrix(self, y_test, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("[[{} {}]".format(tp, fp))
        print("[[{} {}]".format(fn, tn))

if __name__ == '__main__':
    tree = DesisionTree()
    tree.ex8()

