from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn import model_selection 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import  sklearn.model_selection 
from sklearn.datasets import make_circles
from matplotlib import pyplot
from numpy import where
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from plot_confusion_matrix import *
from classification import *
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from functions import *


'''
id1 = np.genfromtxt('../files/plots/faceindices/gender-lbprgbdindices.txt', dtype=int, delimiter=',', names=None)
id2 = np.genfromtxt('../files/plots/faceindices/gender-cheeknose-lbp-rgbdind.txt', dtype=int, delimiter=',', names=None)
id3 = np.genfromtxt('../files/plots/faceindices/gender-chin-lbp-rgbdind.txt', dtype=int, delimiter=',', names=None)
id4 = np.genfromtxt('../files/plots/faceindices/gender-chin+mouth+jaw-lbp-rgbdind.txt', dtype=int, delimiter=',', names=None)
id5 = np.genfromtxt('../files/plots/faceindices/gender-eyes-lbp-rgbdind.txt', dtype=int, delimiter=',', names=None)
id6 = np.genfromtxt('../files/plots/faceindices/gender-nose-lbp-rgbdind.txt', dtype=int, delimiter=',', names=None)
id7 = np.genfromtxt('../files/plots/faceindices/gender-nosemouth-lbp-rgbdind.txt', dtype=int, delimiter=',', names=None)

'''
id1 = np.genfromtxt('../files/plots/exp-indices/exp-face-rgbdindices.txt', dtype=int, delimiter=',', names=None)
id2 = np.genfromtxt('../files/plots/exp-indices/exp-cheeknose-rgbdindices.txt', dtype=int, delimiter=',', names=None)
id3 = np.genfromtxt('../files/plots/exp-indices/exp-chin-rgbdindices.txt', dtype=int, delimiter=',', names=None)
id4 = np.genfromtxt('../files/plots/exp-indices/exp-chin+mouth+jaw-rgbdindices.txt', dtype=int, delimiter=',', names=None)
id5 = np.genfromtxt('../files/plots/exp-indices/exp-eyes-rgbdindices.txt', dtype=int, delimiter=',', names=None)
id6 = np.genfromtxt('../files/plots/exp-indices/exp-nose-rgbdindices.txt', dtype=int, delimiter=',', names=None)
id7 = np.genfromtxt('../files/plots/exp-indices/exp-nosemouth-rgbdindices.txt', dtype=int, delimiter=',', names=None)

#id2 = np.genfromtxt('../files/plots/faceindices/exp-lbprgbdindices.txt', dtype=int, delimiter=',', names=None)
#id3 = np.genfromtxt('../files/plots/faceindices/race-lbprgbdindices.txt', dtype=int, delimiter=',', names=None)


'''
th1 = 7275
th2 = 2387
th3 = 473
th4 = 1440
th5 = 918
th6 = 400
th7 = 640
'''

th1 = 7275
th2 = 2387
th3 = 473
th4 = 1440
th5 = 918
th6 = 400
th7 = 640

N = 7

val1,val4 = get_item_rgbd(id1,th1)
val2,val5 = get_item_rgbd(id2,th2)
val3,val6 =  get_item_rgbd(id3,th3)
val4,val7 =  get_item_rgbd(id4,th4)
val5,val8 =  get_item_rgbd(id5,th5)
val6,val9 =  get_item_rgbd(id6,th6)
val7,val10 =  get_item_rgbd(id7,th7)

rgb = (val1,val2,val3,val4,val5,val6,val7)
depth = (val4,val5,val6,val7,val8,val9,val10)

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, rgb, width,color='m')
p2 = plt.bar(ind, depth, width,bottom=rgb,color='b')

plt.ylabel('Number of selected features')
plt.title('')
plt.xticks(ind, ('face', 'cheeknose', 'chin','chin+mouth+jaw','eyes','nose','nosemouth'))
#plt.yticks(np.arange(100, 1000, 300))
plt.legend((p1[0], p2[0]), ('rgb', 'depth'))

plt.show()