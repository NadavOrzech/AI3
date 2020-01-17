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

        clf = tr.DecisionTreeClassifier(criterion ='entropy',min_samples_leaf=min_sample_leaf)
        clf = clf.fit(self.x_train, self.y_train)

        y_pred = clf.predict(self.x_test)
        mat = confusion_matrix(self.y_test, y_pred)
        self.printMatrix(self.y_test, y_pred)
        return mat, clf

    def calculate_accuracy(self, mat, len):
        return (mat[0][0] + mat[1][1])/len

    def split_rule(self, lamda):
        pass

    def printMatrix(self, y_test, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print("[[{} {}]".format(tp, fp))
        print("[[{} {}]".format(fn, tn))


    def ex3(self):
        tree3_mat, tmp = self.ex1(3)
        tree9_mat, tmp = self.ex1(9)
        tree27_mat, tmp = self.ex1(27)

        tree3_accu = self.calculate_accuracy(tree3_mat, self.test_set_len)
        tree9_accu = self.calculate_accuracy(tree9_mat, self.test_set_len)
        tree27_accu = self.calculate_accuracy(tree27_mat, self.test_set_len)

        objects = [3, 9, 27]
        performance = [tree3_accu, tree9_accu, tree27_accu]
        plt.plot(objects, performance, 'ro')
        plt.ylabel('Accuracy')
        plt.xlabel('X')
        plt.show()

        return tree3_accu, tree9_accu, tree27_accu

    def ex4(self):
        tree27_mat, tree27 = self.ex1(27)
        graph=plt.figure(figsize=(14, 7))
        graph = tr.plot_tree(tree27, feature_names=['Pregnancies',	'Glucose',	'BloodPressure',
                                                     'SkinThickness', 'Insulin', 'BMI',
                                                     'DiabetesPedigreeFunction','Age'])
        plt.show(graph)

    def ex7(self):
        tree1_mat, tmp = self.ex1()
        tree27_mat, tmp = self.ex1(27)

        error_w1 = tree1_mat[0][1] + 4*tree1_mat[1][0]
        error_w27 = tree27_mat[0][1] + 4*tree27_mat[1][0]
        print("error1: "+str(error_w1)+", error27: "+str(error_w27))


if __name__ == '__main__':
    tree = DesisionTree()

    # print(tree.ex3())
    # tree.ex4()
    tree.ex7()



