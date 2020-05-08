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
from numpy import where
from numpy import unique

#dataRl = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-face-'+parts[0]+'-RGB.txt', dtype=float, delimiter=',', names=None)
#dataDl = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-face-'+parts[0]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)



# define the class distribution
# plot dataset

task = ["gender","race","exp"]
name = ["complexity2","race","exp"]
#parts = ["face-sift"]
parts = ["face-hog","face-lbp","face-gabor",]
#parts = ["cheeknose-lbp","chin+mouth+jaw-lbp","eyes-lbp","chin-lbp","nose-lbp","nosemouth-lbp","face-lbp"]
clas = "boost"
n = 10
for i in range(len(parts)):

		fig, ax1 = plt.subplots()
	
		dataRL1 = np.genfromtxt('../files/'+task[0]+'/'+name[0]+'-'+parts[i]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDL1 = np.genfromtxt('../files/'+task[0]+'/'+name[0]+'-'+parts[i]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)

		dataRL2 = np.genfromtxt('../files/'+task[1]+'/'+name[1]+'-'+parts[i]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDL2 = np.genfromtxt('../files/'+task[1]+'/'+name[1]+'-'+parts[i]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)

		dataRL3 = np.genfromtxt('../files/'+task[2]+'/'+name[2]+'-'+parts[i]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDL3 = np.genfromtxt('../files/'+task[2]+'/'+name[2]+'-'+parts[i]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)

		Y11 = np.genfromtxt('../files/labels/labels'+name[0]+'.txt', dtype=int, delimiter=',', names=None)
		Y22 = np.genfromtxt('../files/labels/labels'+name[1]+'.txt', dtype=int, delimiter=',', names=None)
		Y33 = np.genfromtxt('../files/labels/labels'+name[2]+'.txt', dtype=int, delimiter=',', names=None)


		#unique1, counts1 = np.unique(Y11, return_counts=True)
		#unique2, counts2 = np.unique(Y22, return_counts=True)
		#unique3, counts3 = np.unique(Y33, return_counts=True)

		#plt.bar(unique1, counts1)
		#plt.bar(unique2, counts2)
		#plt.bar(unique3, counts3)
		
		#plt.show()

		X1l1 = dataRL1[:, :-1]
		X2l1 = dataDL1[:, :-1]

		X1l2 = dataRL2[:, :-1]
		X2l2 = dataDL2[:, :-1]

		X1l3 = dataRL3[:, :-1]
		X2l3 = dataDL3[:, :-1]

		Xrdl1 = conc_data(X1l1,X2l1)
		Xrdl2 = conc_data(X1l2,X2l2)
		Xrdl3 = conc_data(X1l3,X2l3)

		data11 = scale(Xrdl1)
		data22 = scale(Xrdl2)
		data33 = scale(Xrdl3)

		data1,Y1 = shuffle(data11,Y11)
		data2,Y2 = shuffle(data22,Y22)
		data3,Y3 = shuffle(data33,Y33)

		#select_from_classifier2(data1,Y1)	
		
		clf1 = AdaBoostClassifier((DecisionTreeClassifier(max_depth=1,class_weight="balanced")),n_estimators=1000)
		clf2 = AdaBoostClassifier((DecisionTreeClassifier(max_depth=1,class_weight="balanced")),n_estimators=1000)
		clf3 = AdaBoostClassifier((DecisionTreeClassifier(max_depth=1,class_weight="balanced")),n_estimators=1000)

		model1 = clf1.fit(data1,Y1)
		model2 = clf2.fit(data2,Y2)
		model3 = clf3.fit(data3,Y3)

		plt.ylabel('Features importance')
		plt.title('')
		plt.axvspan(0, len(X1l1[0]),0,0.1, facecolor='r', alpha=0.5,label="rgb features")
		plt.axvspan(len(X1l1[0]), len(Xrdl1[0]),0,0.1, facecolor='b', alpha=0.5,label="depth features")

		#plt.xticks(ind, ('LBP', 'HOG', 'GABOR','SIFT'))
		plt.plot(model1.feature_importances_,color='m',label='Gender classification')
		plt.plot(model2.feature_importances_,color='g',label='Ethnicity classification')
		plt.plot(model3.feature_importances_,color='b',label='Expressions classification')

		#plt.yticks('ssss','sss','ss')
		plt.legend()

		filename = '../files/plots/'+ parts[i]+ 'feature_importances.pdf'
		fig.savefig(filename, dpi=300)
		#plt.close(fig)		
		plt.show()
		