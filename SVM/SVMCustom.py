import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay


class SVMCustom:
    def __init__(self):
        self.classifier = None

    def initialise_classifier(self, kernel: str, C: float, random_state=42):
        self.classifier = SVC(kernel=kernel, C=C, random_state=random_state)

    def fit(self, X, y) -> SVC:
        self.classifier.fit(X, y)
        return self.classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def plot_decision_boundary(self, X, Y, plot_title="Decision Boundary"):
        disp = DecisionBoundaryDisplay.from_estimator(
            self.classifier, X, response_method="predict")
        disp.ax_.scatter(X[:, 0], X[:, 1], c=Y, edgecolor="k")
        plt.title(plot_title)
        plt.show()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.classifier.score(X, y)

    def get_params(self):
        return self.classifier.get_params()
