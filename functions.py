from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
#from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import libsvm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn import model_selection 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import  sklearn.model_selection 
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
from sklearn.metrics import log_loss
from sklearn.metrics import *
from scipy.stats import sem
from scipy.stats import t
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from collections import Counter

def select_pca(trx,tex,var):
    pca = PCA(n_components=var)
    #trx_new = pca.fit(trx).
    #tex_new = pca.fit(tex)

    trx_new = pca.fit_transform(trx)
    tex_new = pca.transform(tex)    	
    return trx_new,tex_new

def scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    #transformer = RobustScaler().fit(data)
    #data22 = transformer.transform(data)
    data2 = scaler.fit_transform(data)
    return data2

def get_item_rgbd(l,th):
	itemr = []
	itemd = []
	for i in xrange(0,len(l)):
		if l[i]<th:
			itemr.append(l[i])
		else:
			itemd.append(l[i])
			#itemdepth = itemdepth + 1
	itemrgb = len(set(itemr))
	itemdepth = len(set(itemd))
	print(len(itemr),len(itemd))
	
	return itemrgb,itemdepth

def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = np.mean(data1), np.mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = np.sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

def select_from_classifier2(data1,y):
    data = scale(data1)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,class_weight="balanced"),n_estimators=100)
    model = SelectFromModel(clf).fit(data,y)    
    indices = model.get_support(indices=True)
    data_n = model.transform(data)
    new_data = scale(data_n)
    plt.plot(model.feature_importances_)
    #plt.show()
    return new_data,indices

def select_from_classifier(trx,ytr,tex,classifier,n):

    #trx,ytr = shuffle(data,y)
    if classifier == "boost":
        #clf = DecisionTreeClassifier(max_depth = 1,n_estimators=1000)
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,class_weight="balanced"),n_estimators=n)
        #clf = AdaBoostClassifier(n_estimators=n)
    elif classifier == "GBT":
        clf = GradientBoostingClassifier(n_estimators=n)
    elif classifier == "RF":
        clf = RandomForestClassifier(max_depth=1,n_estimators=100,class_weight="balanced")
    elif  classifier == "pca":
        trx_n,tex_n = select_pca(trx,tex,n)
        trx_new = scale(trx_n)
        tex_new = scale(tex_n)
        return trx_new,tex_new
        #clf = XGBClassifier(n_estimators=n)
    model = SelectFromModel(clf).fit(trx,ytr)	
    #model = model.fit(trx,ytr)  
    indices = model.get_support(indices=True)
    trx_n = model.transform(trx)
    tex_n = model.transform(tex)
    trx_new = scale(trx_n)
    tex_new = scale(tex_n)
    return trx_new,tex_new,indices,model

def get_loss(ytest,ypred):
	l = 0
	for i in range(len(ytest)):
		if(ytest[i] != ypred[i]):
			l = l+1
	return(float(l)/len(ytest))

def compute_avg_acc(X,y,classifier,n,fusion,clas,k):
    cv = StratifiedKFold(n_splits=n,shuffle=False, random_state=42) # this was changed in depth
    avg_accs = []
    losses = []
    missed =[]
    f1s=[]
    inds=[]
    i = 0
    for train, test in cv.split(X, y):
        ytrain,ytest = y[train],y[test]
        Xtr = X[train]
        Xte = X[test]
        if (fusion == True):
            if clas == "pca":
                Xtrain,Xtest = select_from_classifier(Xtr,ytrain,Xte,clas,k)
            else:
                Xtrain,Xtest,ind,_ = select_from_classifier(Xtr,ytrain,Xte,clas,k)
                inds.append(ind)
        else:

            Xtrain,Xtest = X[train],X[test]

        classifier = classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        avg_acc = balanced_accuracy_score(ytest, y_pred) # try balanced f score accuracy
        f=f1_score(ytest,y_pred,average='weighted')
        f1s.append(f)
        l = get_loss(ytest,y_pred)        
        avg_accs.append(avg_acc)
        losses.append(l)
        i += 1
    #print(Counter(inds))
    mean_avg_acc = np.mean(avg_accs) #try the median, std
    mean_loss = np.mean(losses)
    return mean_avg_acc,mean_loss,classifier

def train_sep_models(X1,part, train_idx,test_idx, y_train, classifier,n,fusion,clasfusion,k):

    #transformer = RobustScaler().fit(X1)
    #X2 = transformer.transform(X1) # only for RGB
    #print(X2)
    X = scale(X1)#changed
    if fusion == True: 
        X_train,X_test,_,_ = select_from_classifier(X[train_idx],y_train,X[test_idx],clasfusion,k)
    else:
        X_train,X_test = X[train_idx],X[test_idx]
    #evaluation
    avg_acc,mean_loss,clf = compute_avg_acc(X_train,y_train,classifier,n,False,clasfusion,k)
    y_pred = clf.predict(X_test)

    return avg_acc,mean_loss,y_pred

def weight_function(l1,l2,th,weight):
    w = []
    for i in range(len(l1)):
        acc = l1[i]
        l = l2[i]
        if acc<=th:
            w_val= 0
        elif acc > th:
            if weight == "acc":               
                w_val = acc
            elif weight == "accloss":
                w_val = float(acc)/l
            elif weight == "log":
                w_val = np.log(float(acc))/l
            elif weight == "meanlog":
                w_val = .5*np.log(float(acc)/l)
        w.append(w_val)
    return w

def conc_data(data1,data2):
  dataf=[]
  for x in range(len(data1)):
    vec1 = data1[x]
    vec2 = data2[x]
    vec = np.concatenate((vec1,vec2), axis=0) 
    dataf.append(vec)
  dataf1 = scale(dataf)
  return dataf1


def compute_avg_acc2(X1,X2,X3,y,classifier,n,fusion,clas,k):
    cv = StratifiedKFold(n_splits=n,shuffle=False, random_state=42)
    avg_accsr = []
    avg_accsd = []
    avg_accsrd = []
    avg_accsrdpca = []
    avg_accsrdboost = []

    losses = []
    missed =[]
    f1s=[]
    inds=[]
    i = 0

    clf1 = svm.SVC(gamma= 0.001,C =100,class_weight={1: 10},random_state=42)
    clf2 = svm.SVC(gamma= 0.001,C =100,class_weight={1: 10},random_state=42)
    clf3 = svm.SVC(gamma= 0.001,C =100,class_weight={1: 10},random_state=42)
    clf4 = svm.SVC(gamma= 0.001,C =100,class_weight={1: 10},random_state=42)
    clf5 = svm.SVC(gamma= 0.001,C =100,class_weight={1: 10},random_state=42)

    for train, test in cv.split(X1, y):
        ytrain,ytest = y[train],y[test]
        XtrainR,XtestR = X1[train],X1[test]
        XtrainD,XtestD = X2[train],X2[test]
        XtrainRD,XtestRD = X3[train],X3[test]

        XtrainP,XtestP = select_from_classifier(X3[train],ytrain,X3[test],"pca",.95)
        Xtrain1,Xtest1,_,_ = select_from_classifier(X1[train],ytrain,X1[test],"boost",1000)
        Xtrain2,Xtest2,_,_ = select_from_classifier(X2[train],ytrain,X2[test],"boost",1000)
        Xtrain3 = conc_data(Xtrain1,Xtrain2)
        Xtest3 = conc_data(Xtest1,Xtest2) 
        Xtrain,Xtest,_,_ = select_from_classifier(Xtrain3,ytrain,Xtest3,"boost",1000)


        clf1 = clf1.fit(XtrainR, ytrain)
        clf2 = clf2.fit(XtrainD, ytrain)
        clf3 = clf3.fit(XtrainRD, ytrain)
        clf4 = clf4.fit(XtrainP, ytrain)
        clf5 = clf5.fit(Xtrain, ytrain)

        y_pred1 = clf1.predict(XtestR)
        y_pred2 = clf2.predict(XtestD)
        y_pred3 = clf3.predict(XtestRD)
        y_pred4 = clf4.predict(XtestP)
        y_pred5 = clf5.predict(Xtest)

        avg_acc1 = balanced_accuracy_score(ytest, y_pred1)
        avg_acc2 = balanced_accuracy_score(ytest, y_pred2)
        avg_acc3 = balanced_accuracy_score(ytest, y_pred3)
        avg_acc4 = balanced_accuracy_score(ytest, y_pred4)
        avg_acc5 = balanced_accuracy_score(ytest, y_pred5)

        #l = get_loss(ytest,y_pred)        
        avg_accsr.append(avg_acc1)
        avg_accsd.append(avg_acc2)
        avg_accsrd.append(avg_acc3)
        avg_accsrdpca.append(avg_acc4)
        avg_accsrdboost.append(avg_acc5)

        #losses.append(l)
        i += 1
    #print(Counter(inds))
    mean_avg_accr = np.mean(avg_accsr) #try the median, std
    mean_avg_accd = np.mean(avg_accsd) #try the median, std
    mean_avg_accrd = np.mean(avg_accsrd) #try the median, std
    mean_avg_accP = np.mean(avg_accsrdpca) #try the median, std
    mean_avg_acc = np.mean(avg_accsrdboost) #try the median, std

    print(mean_avg_accr,mean_avg_accd,mean_avg_accrd,mean_avg_accP,mean_avg_acc)
    mean_loss = np.mean(losses)
    return mean_avg_acc,mean_loss,classifier


def compute_curve_ROC(X,y,n,clas,fusion,clasfusion,k):
  
    cv = StratifiedKFold(n_splits=n)

    if clas == "SVM":
    	classifier = SVC(gamma= 0.001,C =100,class_weight="balanced",probability=True)
    elif clas == "RF":    	
    	classifier = RandomForestClassifier(n_estimators=1000)
    elif clas == "boost":
    	classifier = AdaBoostClassifier(n_estimators=1000)

    tprs = []
    aucs = []
    accs = []
    avg_accs = []
    y_real = []
    y_proba = []
    Fscores = []
    inds = []

    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
    	ytrain,ytest = y[train],y[test]
    	if (fusion == True):
            if clasfusion == "pca":
                Xtrain,Xtest = select_from_classifier(X[train],ytrain,X[test],clasfusion,k)
            else:
                Xtrain,Xtest,ind,_ = select_from_classifier(X[train],ytrain,X[test],clasfusion,k)
                inds.append(ind)
    	else:
    		Xtrain,Xtest = X[train],X[test]
        probas_ = classifier.fit(Xtrain, ytrain).predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(probas_[:, 1])     

        y_pred = classifier.predict(Xtest)
        avg_acc = balanced_accuracy_score(ytest, y_pred)
        #Fscores.append(Fscore)
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)
        avg_accs.append(avg_acc)

        #plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
        i += 1    
    #mean tpr1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #mean auc
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # precision-recall
    '''
    y_real = numpy.concatenate(y_real)
    y_proba = numpy.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba) 
    # standard deviation 
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    '''    
    # mean accuracy1
    mean_avg_acc = np.mean(avg_accs) 
    #print("avg acc",mean_avg_acc) 
    mean_fscore = np.mean(Fscores)  
    return mean_avg_acc,mean_fpr,mean_tpr,mean_auc,std_auc

def compute_curve_ROC_multi(X,y,n,clas,fusion,clasfusion,k):
  
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    #cv = StratifiedKFold(n_splits=n)
    cv = KFold(n_splits=n)

    if clas == "SVM":
    	classifier = OneVsRestClassifier(SVC(gamma= 0.001,C =100,probability=True,class_weight='balanced'))
    elif clas == "RF":    	
    	classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000,class_weight='balanced'))
    elif clas == "boost":
    	classifier = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=1000)) 

    tprs = []
    aucs = []
    Fscores = []

    mean_fpr = np.linspace(0, 1, 100)

    i = 0

    for train, test in cv.split(X, y):
    	ytrain,ytest = y[train],y[test]
    	if (fusion == True):
            if clasfusion == "pca":
                Xtrain,Xtest = select_from_classifier(X[train],ytrain,X[test],clasfusion,k)
            else:
                Xtrain,Xtest,ind,_ = select_from_classifier(X[train],ytrain,X[test],clasfusion,k)
                inds.append(ind)
        else:
            Xtrain,Xtest = X[train],X[test]       

        y_score = classifier.fit(Xtrain, ytrain).predict_proba(Xtest)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ytest[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(ytest.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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
   
def plot_confmat(data,labels,namefile):
    
    clf = SVC(gamma= 0.001,C =10)
    #RandomForestClassifier(max_depth=80, n_estimators=100, max_features=2),
    #AdaBoostClassifier(n_estimators=1000)
    kf = KFold(n_splits=10)

    scoring = ['precision_weighted','recall_weighted','f1_weighted']

    results = cross_validate(estimator=clf,
                                      X=data,
                                      y=labels,
                                      cv=kf,
                                      scoring=scoring)

    avg_prec = numpy.mean(results['test_precision_weighted'])
    avg_recall = numpy.mean(results['test_recall_weighted'])
    avg_fscore = numpy.mean(results['test_f1_weighted'])                                         

    y_pred = cross_val_predict(clf, data, labels, cv=kf)
    plot_confusion_matrix(labels, y_pred,namefile, classes=['M','F'], normalize=True,
                  title='Normalized confusion matrix')
    return avg_prec,avg_recall,avg_fscore

def plot_prec_rec(task,des,clas,n,prec1,rec1,prec2,rec2,prec3,rec3):

    fig, ax1 = plt.subplots()

    # plot AUC

    plt.plot(rec1, prec1, color='g',
             label=r'RGB: Overall AUC=%.4f' % (auc(rec1, prec1)),
             lw=2, alpha=.8)


    plt.plot(rec2, prec2, color='m',
             label=r'Depth: Overall AUC=%.4f' % (auc(rec2, prec2)),
             lw=2, alpha=.8)

    plt.plot(rec3, prec3, color='b',
             label=r'RGB-D:  Overall AUC=%.4f' % (auc(rec3, prec3)),
             lw=2, alpha=.8)


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve for '+task +' classification using RGB,\n Depth and RGB-D data with '+ clas +' classifier (kfold = 10)')
    plt.legend(loc="lower right", prop={'size': 6})

    # save plot
    
    filename = '../files/plots/'+ task + '-' + des + '-' + str(n) + '-' + clas + '-' + 'Prec-recall-curve.pdf'
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def plot_AUC(task,des,clas,n,mean_fpr1,mean_tpr1,mean_auc1,std_auc1,
                mean_fpr2,mean_tpr2,mean_auc2,std_auc2,
                mean_fpr3,mean_tpr3,mean_auc3,std_auc3,
                mean_fpr4,mean_tpr4,mean_auc4,std_auc4,
                mean_fpr5,mean_tpr5,mean_auc5,std_auc5):

                #mean_fpr6,mean_tpr6,mean_auc6,std_auc6):
    print(mean_auc1,mean_auc2,mean_auc3,mean_auc4,mean_auc5)
    fig, ax1 = plt.subplots()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.plot(mean_fpr1, mean_tpr1, color='g',
             label=r'RGB: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc1, std_auc1),
             lw=2, alpha=.8)

    plt.plot(mean_fpr2, mean_tpr2, color='m',
             label=r'Depth: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc2, std_auc2),
             lw=2, alpha=.8)
    plt.plot(mean_fpr3, mean_tpr3, color='k',
             label=r'RGB-D concat: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc3, std_auc3),
             lw=2, alpha=.8)
    
    plt.plot(mean_fpr4, mean_tpr4, color='b',
             label=r'RGB-D based PCA: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc4, std_auc4),
             lw=2, alpha=.8)
    plt.plot(mean_fpr5, mean_tpr5, color='r',
             label=r'RGB-D based adaboost: Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc5, std_auc5),
             lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for '+task +' classification using RGB,\n Depth and RGB-D data with '+ clas +' classifier (kfold = 10)')
    plt.legend(loc="lower right", prop={'size': 6})

    # save plot
    
    filename = '../files/plots/'+ task + '-' + des + '-' + n + '-' + clas + '-' + 'ROC-curve.pdf'
    fig.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)

                      

