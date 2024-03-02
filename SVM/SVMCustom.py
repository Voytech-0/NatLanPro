import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues',
                          plot=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = np.swapaxes(cm, 0, 1)  # optionally swap axes

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    # print('Confusion matrix, without normalization')

    true_positives = np.zeros(len(classes))
    for i in range(len(classes)):
        true_positives[i] = cm[i, i]

    precision = (true_positives / cm.sum(axis=1)).reshape(len(classes), 1)
    recall = (true_positives / cm.sum(axis=0)).reshape(1, len(classes))

    # just make sure that NaN's are eliminated

    Fscore = 2. * precision * recall / (precision + recall)
    Fscore[np.isnan(Fscore)] = 0

    fig = plt.figure(figsize=(5, 5))
    lc = len(classes)  # shorthand

    r_cm = plt.subplot2grid((2 * lc + 3, 2 * lc + 3), (2 * lc + 1, 0), colspan=2 * lc, rowspan=2)
    p_cm = plt.subplot2grid((2 * lc + 3, 2 * lc + 3), (0, 2 * lc + 1), colspan=2, rowspan=2 * lc)
    f_cm = plt.subplot2grid((2 * lc + 3, 2 * lc + 3), (2 * lc + 1, 2 * lc + 1), colspan=2,
                            rowspan=2)  # , sharex=p_cm, sharey=r_cm)
    a_cm = plt.subplot2grid((2 * lc + 3, 2 * lc + 3), (0, 0), colspan=2 * lc,
                            rowspan=2 * lc)  # , sharex=r_cm, sharey=p_cm)

    ## plot cm main
    a_cm.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=cm.max())

    tick_marks = np.arange(len(classes))
    a_cm.xaxis.tick_top()
    a_cm.xaxis.set_label_position('top')
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'  # if normalize else '{d}'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        a_cm.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  verticalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    a_cm.set_xlabel('True label')
    a_cm.set_ylabel('Predicted label')

    ## plot precision (right)
    p_cm.imshow(precision, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    fmt = '.2f'
    thresh = 1. / 2.
    for i, j in itertools.product(range(precision.shape[0]), range(precision.shape[1])):
        p_cm.text(j, i, format(precision[i, j], fmt),
                  horizontalalignment="center",
                  verticalalignment="center",
                  color="white" if precision[i, j] > thresh else "black")

    #     p_cm.xaxis.tick_top()
    p_cm.set_xticks([0])
    p_cm.xaxis.set_label_position('top')
    p_cm.set_xticklabels(["Precision"], rotation=0)
    p_cm.set_yticks([])

    ## plot recall (bottom)
    r_cm.imshow(recall, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    for i, j in itertools.product(range(recall.shape[0]), range(recall.shape[1])):
        r_cm.text(j, i, format(recall[i, j], fmt),
                  horizontalalignment="center",
                  verticalalignment="center",
                  color="white" if recall[i, j] > thresh else "black")
    r_cm.set_yticks([0])
    r_cm.set_yticklabels(["Recall"], rotation=0)
    r_cm.set_xticks([])

    ## plot fscore
    f_cm.imshow([[np.mean(Fscore)]], interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    f_cm.text(0, 0, format(np.mean(Fscore), fmt),
              horizontalalignment="center",
              verticalalignment="center",
              color="white" if np.mean(Fscore) > thresh else "black")

    f_cm.xaxis.tick_bottom()
    f_cm.set_xticks([0])
    f_cm.set_xticklabels(["F1 score"], rotation=0)
    f_cm.set_yticks([])

    # the following makes it all tight, no space between the axis
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00, wspace=0.0, hspace=0.0)
    #     plt.tight_layout()

    if plot:
        plt.show()
    return plt


class SVMCustom:
    def __init__(self, kernel: str = "rbf", C: float = 1.0, random_state=42):
        self.classifier = SVC(kernel=kernel, C=C, random_state=random_state)

    def fit(self, X, y) -> SVC:
        self.classifier.fit(X, y)
        return self.classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def plot(self, X, Y, plot_title="Decision Boundary"):
        disp = DecisionBoundaryDisplay.from_estimator(
            self.classifier, X, response_method="predict")
        disp.ax_.scatter(X[:, 0], X[:, 1], c=Y, edgecolor="k")
        plt.title(plot_title)
        plt.show()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.classifier.score(X, y)

    def get_params(self) -> dict:
        params = self.classifier.get_params()
        return {"kernel": params["kernel"], "C": params["C"], "random_state": params["random_state"]}
