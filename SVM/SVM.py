import numpy as np


class SVM: # https://www.kaggle.com/code/prathameshbhalekar/svm-with-kernel-trick-from-scratch
    def __init__(self, C: int = 1, max_iter: int = 100, learning_rate: int = 0.001):
        self.C = C  # Regularisation
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.C * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.C * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def test_metrics(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        approx = np.dot(X, self.w) - self.b
        # Calculate accuracy
        accuracy = np.sum(np.where(approx == y_, 1, 0)) / len(y_)
        # Calculate precision
        precision = np.sum(np.where(approx == y_, np.where(approx == 1, 1, 0), 0)) / np.sum(np.where(approx == 1, 1, 0))
        # Calculate recall
        recall = np.sum(np.where(approx == y_, np.where(approx == 1, 1, 0), 0)) / np.sum(np.where(y_ == 1, 1, 0))
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
