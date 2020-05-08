import matplotlib.pyplot as plt
import numpy

from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_acc(acc1,acc2,acc3,acc4,acc5,fscore1,fscore2,fscore3,fscore4,fscore5,loss1,loss2,loss3,loss4,loss5):

    labels = ['1', '2', '3','4','5']
    acc_means = [acc1,acc2,acc3,acc4,acc5]
    fscore_means = [fscore1, fscore2,fscore3,fscore4,fscore5]
    loss_means = [loss1, loss2,loss3,loss4,loss5]
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    '''
    rects1 = ax.bar(x + 0.00, acc_means, color = 'b', width = 0.25)
    rects2 = ax.bar(x + 0.25, fscore_means, color = 'g', width = 0.25)
    rects3 = ax.bar(x + 0.50, loss_means, color = 'r', width = 0.25)   
    '''
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x+ 0.00, acc_means, width, label='balanced accuracy',color ='m')
    rects2 = ax.bar(x+ 0.25, fscore_means, width, label='f-score',color = 'b')
    rects3 = ax.bar(x+ 0.50, loss_means, width, label='misclassification rate')
    
    #rects4 = ax.bar(x + width/2, fscore_means, width, label='fscore')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('%')
    ax.set_xlabel('ensemble size')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    autolabel(rects1,ax)
    autolabel(rects2,ax)
    autolabel(rects3,ax)
    
    fig.tight_layout()    
    plt.show()


