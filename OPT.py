import csv
import numpy as np
# from sklearn import tree as tr
from sklearn.metrics import confusion_matrix
from operator import itemgetter
# import matplotlib.pyplot as plt
from math import sqrt
from itertools import chain, combinations

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))


class KNN_opt():

    def __init__(self, subset):
        self.train_set = []
        self.test_set = []
        self.x_train = self.y_train = []
        self.x_test = self.y_test = []
        self.subset = subset

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
        # self.train_set = np.array(self.train_set)
        # self.test_set = np.array(self.test_set)

        self.x_train, self.y_train = self.extract_certain_cols(self.train_set,self.subset)
        self.x_test, self.y_test = self.extract_certain_cols(self.test_set, self.subset)
        self.train_set_len = len(self.train_set)
        self.test_set_len = len(self.test_set)
        self.feature_len = len(self.x_train[0])

    def extract_cols(self, array):
        x = []
        y = []
        for row in array:
            x.append(row[0:len(row) - 1])
            y.append(row[-1])
        return x, y

    def extract_certain_cols(self, array, cols):
        x = []
        y = []
        for row in array:
            tmp = []
            for i in cols:
                tmp.append(row[i])
            x.append(tmp)
            y.append(row[-1])

        return x,y

    def printMatrix(self, y_test, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("[[{} {}]".format(tp, fp))
        print("[[{} {}]".format(fn, tn))

    def normalize(self):
        x_min = [np.inf]*self.feature_len
        x_max = [-np.inf]*self.feature_len

        for i in range(self.feature_len):
            x_max[i] = max(float(row[i]) for row in self.x_train)
            x_min[i] = min(float(row[i]) for row in self.x_train)

        for i in range(self.train_set_len):
            for j in range(self.feature_len):
                self.x_train[i][j] = (float(self.x_train[i][j]) - float(x_min[j])) / (float(x_max[j])-float(x_min[j]))
        for i in range(self.test_set_len):
            for j in range(self.feature_len):
                self.x_test[i][j] = (float(self.x_test[i][j]) - float(x_min[j])) / (float(x_max[j])-float(x_min[j]))

    def euclidean_distance(self, vec1, vec2):
        sum = 0
        for i in range(self.feature_len):
            sum += pow((float(vec1[i]) - float(vec2[i])), 2)
        return sqrt(sum)

    def find_classifier(self, sample):
        sort_arr = []
        for i in range(self.train_set_len):
            sort_arr.append((self.euclidean_distance(sample, self.x_train[i]), i))
        sort_arr = sorted(sort_arr, key=itemgetter(0))
        sum = 0
        for i in range(self.feature_len):
            tmp, index = sort_arr[i]
            sum += int(self.y_train[index])
        if sum > (self.feature_len/2):
            return '1'
        else:
            return '0'

    def calculate_accuracy(self, mat, len):
        return (mat[0][0] + mat[1][1]) / len

    def KNN9(self):
        self.normalize()
        y_pred = []
        for sample in self.x_test:
            y_pred.append(self.find_classifier(sample))

        mat = confusion_matrix(self.y_test, y_pred)
        self.printMatrix(self.y_test, y_pred)
        sum0 = sum1 = 0
        for i in range(len(y_pred)):
            if y_pred[i] == '0':
                sum0 += 1
            else:
                sum1 += 1
        return self.calculate_accuracy(mat, self.test_set_len)

if __name__ == '__main__':
    x = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    iterator = powerset(x)
    arr = []
    for subset in iterator:
        res = KNN_opt(subset)
        arr.append((res.KNN9(), subset))

    arr = sorted(arr, key=itemgetter(0))
    print(arr)
