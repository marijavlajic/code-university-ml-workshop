import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def plot_setup():
    mpl.rcParams['figure.figsize'] = 8, 8
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.edgecolor'] = '0.3'
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelcolor'] = '0.3'
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 24
    mpl.rcParams['patch.edgecolor'] = 'none'
    mpl.rcParams['xtick.major.size'] = 0
    mpl.rcParams['ytick.major.size'] = 0
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['xtick.color'] = '0.3'
    mpl.rcParams['ytick.color'] = '0.3'
    mpl.rcParams['text.color'] = '0.3'
    sns.set_style('white')
    
def plot_supervised_model(name, model, X_test, y_test, y_pred):
    plot_setup()
    cmap_light = ListedColormap(['#d5deb3', '#a3d1db', '#f5b8b8', '#ffc8a3', '#b792ba', '#aabad1', '#b2d7ca', '#ffe192'])
    cmap_bold = ListedColormap(['#8ba52c', '#00819b', '#e43939'])
    plt.figure(figsize=(8, 8))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=100, cmap=cmap_bold)
    idx = np.where(y_pred != y_test)
    plt.scatter(X_test[idx, 0], X_test[idx, 1], c=y_pred[idx], s=300, alpha=0.2, cmap=cmap_bold)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(name)
    plt.show()

def plot_unsupervised_model(name, model, X, y):
    plot_setup()
    cmap_light = ListedColormap(['#d5deb3', '#a3d1db', '#f5b8b8', '#ffc8a3', '#b792ba', '#aabad1', '#b2d7ca', '#ffe192'])
    cmap_bold = ListedColormap(['#8ba52c', '#00819b', '#e43939'])
    h = .02
    X_sub = X[:, :2]
    model.fit(X_sub, y)
    centroids = model.cluster_centers_
    x_min, x_max = X_sub[:, 0].min() - 1, X_sub[:, 0].max() + 1
    y_min, y_max = X_sub[:, 1].min() - 1, X_sub[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_sub[:, 0], X_sub[:, 1], s=50, cmap=cmap_bold)
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                marker='x', s=300, linewidths=5,
                color='w', zorder=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(name)
    plt.xlabel('Sepal Length', fontsize=20)
    plt.ylabel('Sepal Width', fontsize=20)
    plt.show()
