import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def plot_setup():
    mpl.rcParams['figure.figsize'] = 8, 6
#    mpl.rcParams['axes.prop_cycle'] = '#144181', '#E43939', '#00819B', '#FF6500', '#712177', '#FDC400', '#8BA52C', '#2A906D',  '#8A1B70', '#00819B'
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.edgecolor'] = '0.3'
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelcolor'] = '0.3'
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 24
    mpl.rcParams['patch.edgecolor'] = 'none'
    mpl.rcParams['grid.alpha'] = 0.0
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['xtick.major.size'] = 0
    mpl.rcParams['ytick.major.size'] = 0
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['xtick.color'] = '0.3'
    mpl.rcParams['ytick.color'] = '0.3'
    mpl.rcParams['text.color'] = '0.3'
    sns.set_style('white')
    

def plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    prf = precision_recall_fscore_support(y_true, y_pred)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
#    for i in range(len(cm)):
#        plt.text(len(cm)-0.3, i-0.1, str(int(prf[0][i]*100))+"%", 
#                 fontsize=18, verticalalignment='center')
#        plt.text(len(cm)-0.3, i+0.1, str(int(prf[1][i]*100))+"%", 
#                 fontsize=18, verticalalignment='center')
    tick_marks = np.arange(len(cm))
    labels = ['did not survive', 'survived']
    if len(cm) == 2:
        labels = labels[0:]
    plt.xticks(tick_marks, labels, rotation=45, fontsize=16)
    plt.yticks(tick_marks, labels, fontsize=16)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            plt.text(j, i, val, fontsize=30, 
                    horizontalalignment='center',
                    verticalalignment='center')
