"""
Main file for the SVM_custom project
Authors: Wojciech Trejter, Viki Simion, Laura M Quirós
"""
import ast

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def plot_decision_boundary(model, X, y):
    # do pca to make x 2d
    X = PCA.fit_transform(X, y)
    disp = DecisionBoundaryDisplay.from_estimator(
        model, X, response_method="predict")
    disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor="k")
    plt.title("Best Model Decision Boundary")
    plt.show()


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [2, 3, 5, 7, 10]
    }
    svm = SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=params, error_score='raise', scoring='accuracy')

    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}, best score: {grid_search.best_score_}")
    best_model = grid_search.best_estimator_
    results = classification_report(y_test, best_model.predict(X_test))
    print(results)
    plot_decision_boundary(best_model, X_test, y_test)


def one_hot_encoding(train, test):
    # get all unique strings from the train["emotions"] and test set
    unique_emotions = set(train["emotion"].unique().tolist() + test["emotion"].unique().tolist())
    # one hot encode the emotions
    dict_one_hot = {}
    num = 0
    for emotion in unique_emotions:
        dict_one_hot[emotion] = num
        num += 1

    train["emotion"] = train["emotion"].map(dict_one_hot)
    test["emotion"] = test["emotion"].map(dict_one_hot)
    # save the one hot encoding dictionary
    pd.DataFrame.from_dict(data=dict_one_hot, orient='index').to_csv('data/one_hot_encoding.csv', header=False)
    return train, test


def main():
    train = pd.read_csv('data/train_ready_for_WS_vectorized_1-gram.csv', delimiter=';', header=0)
    test = pd.read_csv('data/test_vectorized_1-gram.csv', delimiter=';', header=0)
    train['vector_1-gram'] = train['vector_1-gram'].apply(ast.literal_eval)
    test['vector_1-gram'] = test['vector_1-gram'].apply(ast.literal_eval)
    train, test = one_hot_encoding(train, test)

    train_X = train["vector_1-gram"]
    # make the list of values into columns
    train_X = pd.DataFrame(train_X.tolist())
    train_y = train["emotion"]
    test_X = test["vector_1-gram"]
    test_X = pd.DataFrame(test_X.tolist())
    test_y = test["emotion"]

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    hyperparameter_tuning(train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    main()