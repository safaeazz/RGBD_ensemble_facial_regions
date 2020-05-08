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




clas = "svm"
n = 10

def plot_AUC(task,des,clas,n,mean_fpr1,mean_tpr1,mean_auc1,std_auc1,mean_fpr2,mean_tpr2,mean_auc2,std_auc2,
                             mean_fpr3,mean_tpr3,mean_auc3,std_auc3):

    fig, ax1 = plt.subplots()

    # plot AUC

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    plt.plot(mean_fpr1, mean_tpr1, color='g',
             label=r'RGB: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc1, std_auc1),
             lw=2, alpha=.8)


    plt.plot(mean_fpr2, mean_tpr2, color='m',
             label=r'Depth: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc2, std_auc2),
             lw=2, alpha=.8)

    plt.plot(mean_fpr3, mean_tpr3, color='b',
             label=r'RGB-D: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc3, std_auc3),
             lw=2, alpha=.8)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     #label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for '+task +' classification using RGB,\n Depth and RGB-D data with '+ clas +' classifier (kfold = 10)')
    plt.legend(loc="lower right", prop={'size': 6})

    # save plot
    
    filename = '../files/plots/'+ task + '-' + des + '-' + str(n) + '-' + clas + '-' + 'ROC-curve.pdf'
    fig.savefig(filename, dpi=300)
    plt.close(fig)

                      

def compute_curve_ROC_multi(X,y,n):
  
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    cv = KFold(n_splits=n)
    classifier = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1000, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0))    classifier =  (AdaBoostClassifier(n_estimators=1000))

    tprs = []
    aucs = []

    mean_fpr = np.linspace(0, 1, 100)
    i = 0

    for train, test in cv.split(X, y):

        y_score = classifier.fit(X[train], y[train]).predict_proba(X[test])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()


        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[test][:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area

        fpr["micro"], tpr["micro"], _ = roc_curve(y[test].ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #plt.plot(fpr["micro"], tpr["micro"], lw=1, alpha=0.3,
        #     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc["micro"]))

        tprs.append(interp(mean_fpr, fpr["micro"], tpr["micro"]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc["micro"])

        i += 1   
    
    #mean tpr1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #mean auc
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0) 
    return mean_fpr,mean_tpr,mean_auc,std_auc
   


########################################################################################################
########################################################################################################

task = ["exp","race"]
des = ["hog"]
#,"lbp","hog","sift"]
clas = "svm"
n = 10
for i in range(task):
	for j in range(des):
	
		dataR = np.genfromtxt('../files/'+task[i]+'/'+task[i]+'-face-'+des[j]+'-RGB.txt', dtype=float, delimiter=',', names=None)
		dataD = np.genfromtxt('../files/'+task[i]+'/'+task[i]+'-face-'+des[j]+'-DEPTH.txt', dtype=float, delimiter=',', names=None)
		dataRD = np.genfromtxt('../files/'+task[i]+'/'+task[i]+'-face-'+des[j]+'-RGBD.txt', dtype=float, delimiter=',', names=None)
		labels = np.genfromtxt('../files/labels/labels'+task[i]+'.txt', dtype=int, delimiter=',', names=None)
		
		Y = labels
		X1 = dataR[:, :-1]
		X2 = dataD[:, :-1]
		X3 = dataRD[:, :-1]
		Xr,Xd,Xrd,Y1 = shuffle(X1,X2,X3,Y)
		
		scaler1 = MinMaxScaler(feature_range=(0, 1)) #rescale
		scaler2 = MinMaxScaler(feature_range=(0, 1)) #rescale
		scaler3 = MinMaxScaler(feature_range=(0, 1)) #rescale
		
		Xr2 = scaler1.fit_transform(Xr)
		Xd2 = scaler2.fit_transform(Xd) 
		Xrd2 = scaler3.fit_transform(Xrd) 
				
		mean_fpr1,mean_tpr1,mean_auc1,std_auc1 = compute_curve_ROC_multi(Xr2,Y1,n)
		mean_fpr2,mean_tpr2,mean_auc2,std_auc2 = compute_curve_ROC_multi(Xd2,Y1,n)
		mean_fpr3,mean_tpr3,mean_auc3,std_auc3 = compute_curve_ROC_multi(Xrd2,Y1,n)
		
		plot_AUC(task[i],des[j],clas,n,mean_fpr1,mean_tpr1,mean_auc1,std_auc1,mean_fpr2,mean_tpr2,mean_auc2,std_auc2,
		                             mean_fpr3,mean_tpr3,mean_auc3,std_auc3)
		