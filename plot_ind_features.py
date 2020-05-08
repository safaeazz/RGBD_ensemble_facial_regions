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
from sklearn.utils import shuffle


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



task = ["race"]
name = ["race"]
parts = ["face-lbp","face-hog","face-sift","face-gabor"]
#parts = ["cheeknose-lbp","chin+mouth+jaw-lbp","eyes-lbp","chin-lbp","nose-lbp","nosemouth-lbp","face-lbp"]
clas = "boost"
n = 10
for i in range(len(task)):

		fig, ax1 = plt.subplots()
	
		dataRL = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[0]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDL = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[0]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)

		dataRH = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[1]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDH = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[1]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)

		dataRG = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[2]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDG = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[2]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)

		dataRS = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[3]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataDS = np.genfromtxt('../files/'+task[i]+'/'+name[i]+'-'+parts[3]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)


		labels = np.genfromtxt('../files/labels/labels'+name[i]+'.txt', dtype=int, delimiter=',', names=None)
		
		Y = labels

		X1l = dataRL[:, :-1]
		X2l = dataDL[:, :-1]

		X1h = dataRH[:, :-1]
		X2h = dataDH[:, :-1]

		X1g = dataRG[:, :-1]
		X2g = dataDG[:, :-1]

		X1s = dataRS[:, :-1]
		X2s = dataDS[:, :-1]


		Xrdl2 = conc_data(X1l,X2l)
		Xrdh2 = conc_data(X1h,X2h)
		Xrdg2 = conc_data(X1g,X2g)
		Xrds2 = conc_data(X1s,X2s)

		Xrdl, Xrdh, Xrdg, Xrds ,Y1 = shuffle(Xrdl2, Xrdh2, Xrdg2, Xrds2,Y)
		th1 = len(X1l[0])
		th2 = len(X1h[0])
		th3 = len(X1g[0])
		th4 = len(X1s[0])

		data1,id1 = select_from_classifier2(Xrdl,Y)

		_,id2 = select_from_classifier2(Xrdh,Y)
		_,id3 = select_from_classifier2(Xrdg,Y)
		_,id4 = select_from_classifier2(Xrds,Y)

		N = 4

		val1,val5 = get_item_rgbd(id1,th1)
		val2,val6 = get_item_rgbd(id2,th2)
		val3,val7 =  get_item_rgbd(id3,th3)
		val4,val8 =  get_item_rgbd(id4,th4)

		rgb = (val1,val2,val3,val4)
		depth = (val5,val6,val7,val8)
		
		ind = np.arange(N)    # the x locations for the groups
		width = 0.35       # the width of the bars: can also be len(x) sequence
		
		p1 = plt.bar(ind, rgb, width)
		p2 = plt.bar(ind, depth, width,bottom=rgb)
		
		plt.ylabel('Number of selected features')
		plt.title('Scores by group and data type')
		plt.xticks(ind, ('LBP', 'HOG', 'SIFT','GABOR'))
		#plt.yticks(np.arange(100, 1000, 300))
		plt.legend((p1[0], p2[0]), ('rgb', 'depth'))

		filename = '../files/plots/'+ task[i]+ 'distribution.pdf'
		fig.savefig(filename, dpi=300)
		#plt.close(fig)		
		plt.show()