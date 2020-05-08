import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from functions import *
from classification import *
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

clas = ["RF,RF,boost"]
#,"SVM","boost"]
n = 10
task = "exp" # gender, exp,race
des = ["sift"]
#,"sift","gabor","hog"] # gabor, hog, sift
name = "exp" # exp, race,complexity2

for j in range(len(clas)):
    # read data (files)
    for x in range(len(des)):
        namefile = task+"-"+des[x]+"-"+clas[j]
        classifier = svm.SVC(gamma= 0.001,C =100,class_weight="balanced",random_state=42)
        #classifier = RandomForestClassifier(max_depth=1,n_estimators=100,class_weight="balanced")
        #classifier = AdaBoostClassifier(n_estimators=1000)
        print(des[x])
        
        dataR = np.genfromtxt('../files/'+ task + '/'+name+'-face-'+des[x]+'2-RGB.txt', dtype=float, delimiter=',', names=None)
        dataD = np.genfromtxt('../files/'+task+'/'+name+'-face-'+des[x]+'2-DEPTH.txt', dtype=float, delimiter=',', names=None)
        dataRD = np.genfromtxt('../files/'+task+'/'+name+'-face-'+des[x]+'-RGBD.txt', dtype=float, delimiter=',', names=None)
        
        Y = np.genfromtxt('../files/labels/labels'+name+'.txt', dtype=int, delimiter=',', names=None)
        #labels = np.genfromtxt('../files/labels/labels'+task+'.txt', dtype=int, delimiter=',', names=None)
        #labels = np.genfromtxt('../files/labels/labels'+task+'.txt', dtype=int, delimiter=',', names=None)
        
        Xr21 = dataR[:, :-1]
        Xd21 = dataD[:, :-1]
        Xrd21 = dataRD[:, :-1] 

        Xr = scale(Xr21)
        Xd = scale(Xd21)        

        Xrdc21 = conc_data(Xr21,Xd21)
        Xrd = scale(Xrd21)
        Xr2,Xd2,Xrd2,Xrdc2,Y1 = shuffle(Xr,Xd,Xrd,Xrdc21,Y)

        #compute_avg_acc2(Xr2,Xd2,Xrd2,Y1,"classifier",n,True,"boost",100)

        # for binary case
        
        acc1,_,_ = compute_avg_acc(Xr2,Y1,classifier,n,False,'',100)
        acc2,_,_ = compute_avg_acc(Xd2,Y1,classifier,n,False,'',100)
        acc3,_,_ = compute_avg_acc(Xrd2,Y1,classifier,n,False,'boost',1000)
        acc4,_,_ = compute_avg_acc(Xrdc2,Y1,classifier,n,False,'',.95)
        acc5,_,_ = compute_avg_acc(Xrdc2,Y1,classifier,n,True,"pca",.95)

        print(acc1,acc2,acc3,acc4,acc5)
        c = "boost"
        print("rgb")
        mean_fpr1,mean_tpr1,mean_auc1,std_auc1 = compute_curve_ROC_multi(Xr2,Y1,n,c,False,"",100)
        print("depth")
        mean_fpr2,mean_tpr2,mean_auc2,std_auc2 = compute_curve_ROC_multi(Xd2,Y1,n,c,False,"",100)
        print("rgbd")
        mean_fpr3,mean_tpr3,mean_auc3,std_auc3 = compute_curve_ROC_multi(Xrdc2,Y1,n,c,False,"",100)
        print("pca")
        mean_fpr4,mean_tpr4,mean_auc4,std_auc4 = compute_curve_ROC_multi(Xrdc2,Y1,n,c,True,"pca",.95)
        print("boost")
        mean_fpr5,mean_tpr5,mean_auc5,std_auc5 = compute_curve_ROC_multi(Xrd2,Y1,n,c,False,"",100)

        #mean_avg_acc6,mean_fpr6,mean_tpr6,mean_auc6,std_auc6 = compute_curve_ROC(Xrd2,Y1,n,clas[j],True,"pca",3)
        #print("RGB",mean_avg_acc1,"depth",mean_avg_acc2,"RGBD",mean_avg_acc3,"RGBD Boost",mean_avg_acc4,"RGBD RF",mean_avg_acc5,"RGBD pca",mean_avg_acc6)
        
        plot_AUC(task,des[x],c,"10",mean_fpr1,mean_tpr1,mean_auc1,std_auc1,
                mean_fpr2,mean_tpr2,mean_auc2,std_auc2,
                mean_fpr3,mean_tpr3,mean_auc3,std_auc3,
                mean_fpr4,mean_tpr4,mean_auc4,std_auc4,
                mean_fpr5,mean_tpr5,mean_auc5,std_auc5)

#compute balanced acc, prec, confusin matrix

#prec1,rec1,fscore1 = plot_confmat(Xr2,Y1,namefile+"-RGB")
#prec2,rec2,fscore2 = plot_confmat(Xd2,Y1,namefile+"-depth")
#prec3,rec3,fscore3 = plot_confmat(Xrd2,Y1,namefile+"-RGBD")

#avg_acc1 = compute_avg_acc(Xr2,Y1)
#avg_acc2 = compute_avg_acc(Xd2,Y1)
#avg_acc3 = compute_avg_acc(Xrd2,Y1)

# balanced accuracy plot
#plot_acc(avg_acc1,avg_acc2,avg_acc3,fscore1,fscore2,fscore3)

'''
#plot ROC curve
#for multiclass case

mean_fpr1,mean_tpr1,mean_auc1,std_auc1 = compute_curve_ROC_multi(Xr2,Y1,n)
mean_fpr2,mean_tpr2,mean_auc2,std_auc2 = compute_curve_ROC_multi(Xd2,Y1,n)
mean_fpr3,mean_tpr3,mean_auc3,std_auc3 = compute_curve_ROC_multi(Xrd2,Y1,n)

plot_AUC(task,des,clas,n,mean_fpr1,mean_tpr1,mean_auc1,std_auc1,mean_fpr2,mean_tpr2,mean_auc2,std_auc2,
                             mean_fpr3,mean_tpr3,mean_auc3,std_auc3)


'''
#plot prec-recall curve
#plot_prec_rec(task,des,clas,n,prec1,rec1,prec2,rec2,prec3,rec3)

