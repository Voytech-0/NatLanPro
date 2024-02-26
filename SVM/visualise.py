import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mc


def plot_decision_boundary(model, X, Y, plot_title="Decision Boundary"):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=mc.colormaps['coolwarm'], alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=mc.colormaps['coolwarm'], edgecolors='k')
    plt.title(plot_title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary
# plot_decision_boundary(svm_clf, x, y, plot_title="SVM Decision Boundary with Polynomial Kernel")