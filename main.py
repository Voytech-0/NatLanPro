"""
Main file for the SVM_custom project
Authors: Wojciech Trejter, Viki Simion, Laura M Quirós
"""
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from SVM.SVMCustom import SVMCustom


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    params = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    }
    svm = SVMCustom()
    grid_search = GridSearchCV(svm, params, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    results = classification_report(y_test, best_model.predict(X_test))
    print(results)
    best_model.plot(X_test, y_test, "Best model decision boundary")


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
    print(train.columns)
    print(test.columns)
    train, test = one_hot_encoding(train, test)

    train_X = train["vector_1-gram"]

    train_y = train["emotion"]

    test_X = test["vector_1-gram"]
    test_y = test["emotion"]
    print(train_X.head()[0].dtype)
    hyperparameter_tuning(train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    main()
