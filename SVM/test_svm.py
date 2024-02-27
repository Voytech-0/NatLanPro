import unittest
import sklearn.svm as SVM
from SVMCustom import SVMCustom
import numpy as np
import sklearn.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


class MyTestCase(unittest.TestCase):
    def test_svm(self):
        svm = SVM.SVC(kernel='linear')
        svm.fit(np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]), np.array([1, -1, -1, -1]))
        self.assertEqual(svm.predict([[1, 1]]), 1)
        self.assertEqual(svm.predict([[1, -1]]), -1)
        self.assertEqual(svm.predict([[-1, 1]]), -1)
        self.assertEqual(svm.predict([[-1, -1]]), -1)

    def test_svm_iris(self, model: str = "custom"):
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = iris.target
        if model == "sklearn":
            svm = SVM.SVC(kernel='linear', C=1.0, random_state=0).fit(X, y)
        else:
            obj = SVMCustom()
            obj.initialise_classifier(kernel='linear', C=1.0, random_state=0)
            svm = obj.fit(X, y)

        # get accurracy, precision, recall and f1 score from testing
        print(svm.score(X, y))
        # plot decision boundary
        disp = DecisionBoundaryDisplay.from_estimator(
            svm, X, response_method="predict", xlabel=iris.feature_names[0],
            ylabel=iris.feature_names[1])
        disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
        plt.show()

    def test_svm_all(self):
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = iris.target
        best_score = 0
        best_model = None
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            for C in [0.1, 1.0, 10.0]:
                print(f"Kernel: {kernel}, C: {C}")
                svm = SVMCustom()
                svm.initialise_classifier(kernel, C)
                svm.fit(X, y)
                print(svm.score(X, y))
                if svm.score(X, y) > best_score:
                    best_score = svm.score(X, y)
                    best_model = svm
        # plot decision boundary
        print(f"Best Model Score: {best_score} with params {best_model.get_params()}")
        best_model.plot_decision_boundary(X, y, "Best Model Decision Boundary")



if __name__ == '__main__':
    unittest = MyTestCase()
    unittest.test_svm_all()
