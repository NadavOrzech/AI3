import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from math import sqrt


class KNN():

    def __init__(self):
        '''
            Initiate the train and test data sets using the given .csv files
        '''
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

        self.x_train, self.y_train = self.extract_cols(self.train_set)
        self.x_test, self.y_test = self.extract_cols(self.test_set)
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

    def printMatrix(self, y_test, y_pred):
        # This function print the Confusion Matrix in the correct f1 form
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("[[{} {}]".format(tp, fp))
        print("[{} {}]]".format(fn, tn))

    def normalize(self):
        # This function normalizes the train and test data sets, using max and min value of
        # every feature, by the function of feature[i] = feature[i] - min [i]/
        #                                                    max[i] - min[i]
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
        # This function return the euclidean distance of two given features vectors
        sum = 0
        for i in range(self.feature_len):
            sum += pow((float(vec1[i]) - float(vec2[i])), 2)
        return sqrt(sum)

    def find_classifier(self, sample):
        # This function returns classification for a given sample using the given train
        # data set and 9NN algorithm
        sort_arr = []
        for i in range(self.train_set_len):
            sort_arr.append((self.euclidean_distance(sample, self.x_train[i]), i))
        sort_arr = sorted(sort_arr, key=itemgetter(0))
        sum = 0
        for i in range(9):
            tmp, index = sort_arr[i]
            sum += int(self.y_train[index])
        if sum > 4:
            return '1'
        else:
            return '0'

    def calculate_accuracy(self, mat, len):
        # Calculates the accuracy factor of the algorithm using the confusion matrix
        return (mat[0][0] + mat[1][1]) / len

    def KNN9(self):
        # Calculates and prints the confusion matrix of 9NN prediction set
        self.normalize()
        y_pred = []
        for sample in self.x_test:
            y_pred.append(self.find_classifier(sample))

        mat = confusion_matrix( self.y_test, y_pred)
        self.printMatrix(self.y_test, y_pred)

if __name__ == '__main__':
    o = KNN()
    o.KNN9()
