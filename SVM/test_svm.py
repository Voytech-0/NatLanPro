import unittest
import sklearn.svm as SVM
import numpy as np
import sklearn.datasets as datasets


class MyTestCase(unittest.TestCase):
    def test_svm(self):
        svm = SVM.SVC(kernel='linear')
        svm.fit(np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]), np.array([1, -1, -1, -1]))
        self.assertEqual(svm.predict([[1, 1]]), 1)
        self.assertEqual(svm.predict([[1, -1]]), -1)
        self.assertEqual(svm.predict([[-1, 1]]), -1)
        self.assertEqual(svm.predict([[-1, -1]]), -1)

    def test_svm_iris(self):
        iris = datasets.load_iris()
        # divide the data into a training set and a test set
        train_x, train_y, test_x, test_y = self.split_data(iris.data, iris.target)
        svm = SVM.SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(train_x, train_y)
        # Test the model
        svm.score(test_x, test_y)
        # get accurracy, precision, recall and f1 score from testing
        print(svm.score(test_x, test_y))

    def split_data(self, X, y):
        n_samples, n_features = X.shape
        n_train = n_samples // 2
        train_x = X[:n_train]
        train_y = y[:n_train]
        test_x = X[n_train:]
        test_y = y[n_train:]
        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    unittest.main()
