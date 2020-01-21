import csv
import numpy as np
from sklearn import tree as tr
from sklearn.metrics import confusion_matrix

class DesisionTree():

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

    def printMatrix(self, y_test, y_pred):
        # This function print the Confusion Matrix in the correct f1 form
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("[[{} {}]".format(tp, fp))
        print("[{} {}]]".format(fn, tn))

    def ex1(self, min_sample_leaf=1):
        # Creates the Decision Tree using the train and test date sets,
        # using the weighted entropy function.
        # Prints the confusion matrix, and returns the matrix and the decision tree itself
        clf = tr.DecisionTreeClassifier(criterion ='entropy', min_samples_leaf=min_sample_leaf,
                                        class_weight={"1": 4, "0": 1})
        clf = clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)
        mat = confusion_matrix(self.y_test, y_pred)
        self.printMatrix(self.y_test, y_pred)
        return mat, clf

    def ex11(self):
        tree_mat, tmp = self.ex1(9)
        return tree_mat, tmp

if __name__ == '__main__':
    tree = DesisionTree()
    tree.ex11()

