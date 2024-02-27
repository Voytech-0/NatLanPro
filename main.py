"""
Main file for the SVM_custom project
Authors: Wojciech Trejter, Viki Simion, Laura M Quirós
"""
import pandas as pd
from SVM.SVMCustom import SVMCustom


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    best_score = 0
    best_model = None
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        for C in [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
            svm = SVMCustom()
            svm.initialise_classifier(kernel, C)
            print(f"Kernel: {kernel}, C: {C}")
            score = svm.fit(X_train, y_train).score(X_test, y_test)
            print(f"Score: {score}")
            if score > best_score:
                best_score = score
                best_model = svm
    print(f"Best score: {best_score}")
    best_model.plot_decision_boundary(X_test, y_test, "Best model decision boundary")


def main():
    train = pd.read_csv("train_ready_for_WS_preprocessed.csv")
    test = pd.read_csv("test_ready_for_WS_preprocessed.csv")
    X = train["essay"]
    y = train["emotion"]
    print(X.head())
    print(X.shape())
    X_test = test["essay"]
    y_test = test["emotion"]
    hyperparameter_tuning(X, y, X_test, y_test)


if __name__ == "__main__":
    main()