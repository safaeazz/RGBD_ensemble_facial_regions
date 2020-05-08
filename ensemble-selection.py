#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from functions import *
from classification import *
import math
import random
from tabulate import tabulate
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# faire des tests test.py with weak learner=1000
# run rgb,depth,rgbd, sur le pc de yassine
# think about how to plot features frequencies
# compare decision fusion versus feature fusion in both jei and tom

runs = 25
final_acc1 = []
final_acc2 = []
final_acc3 = []
final_fscore1 = []
final_fscore2 = []
final_fscore3 = []
final_face_fscore = []

final_acc_part1 = []
final_acc_part2 = []
final_acc_part3 = []
final_acc_part4 = []
final_acc_part5 = []
final_acc_part6 = []

final_score_part1 = []
final_score_part2 = []
final_score_part3 = []
final_score_part4 = []
final_score_part5 = []
final_score_part6 = []

face_acc = []

#save SAE features

task = "exp" #gender
name = "exp" #complexity2,race
des = "lbp"
clas = "svm"
typed=["RGBD"]
weight = ["accloss"]
#,"acc","meanlog","log"] 
seed = 42
#parts = ["cheeknose","chin+mouth+jaw","eyes","chin","nose","nosemouth","face"]
#parts = ["nosemouth","nose","chin+mouth+jaw","cheeknose","eyes","chin"]
#parts = ["cheeknose","eyes","nose","nosemouth","chin+mouth+jaw","chin"]
parts = ["cheeknose","chin+mouth+jaw","nosemouth","nose","eyes","chin"]

for y in range(len(typed)):

	for z in range(len(weight)):
		print("-----------------------------------------------------------------------------------------")
		print("task: ",task,"data type",typed[y],"weight",weight[z])
		print("-----------------------------------------------------------------------------------------")
		for r in range(runs):
		    labels = np.genfromtxt('../files/labels/labels'+name+'.txt', dtype=int, delimiter=',', names=None)
		    Y = labels
		    preds=[]
		    fs=[]
		    w=[]
		    i = range(len(Y))
		    #suffeling
		    idx_sh = shuffle(i) # shuffeling
		    #train test split
		    train_idx,test_idx = train_test_split(idx_sh,test_size=.3,stratify=Y,random_state=seed)#changed
		    avg_accs = []
		    test_accs = []
		    fscores = []
		    losses = []

		    missed =[]
		    f1s=[]
		    inds=[]
		    i = 0
		    y_train,y_test = Y[train_idx],Y[test_idx]

		    for x in range(len(parts)):		
				classifier = svm.SVC(gamma= 0.001,C =100,class_weight="balanced",random_state=42)#changed
				
				#classifier = AdaBoostClassifier(n_estimators=1000)
				if typed[y] =="RGBD":
					path1 = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+'RGB'+'.txt'
					path2 = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+'DEPTH'+'.txt'
					path = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+typed[y]+'.txt'
					data1 = np.genfromtxt(path1, dtype=float, delimiter=',', names=None)
					data2 = np.genfromtxt(path2, dtype=float, delimiter=',', names=None)
					data = np.genfromtxt(path, dtype=float, delimiter=',', names=None)
					X1 = data1[:, :-1]
					X2 = data2[:, :-1]
					#X = conc_data(X1,X2)
					X = data[:, :-1]
					avg_acc,loss,y_pred = train_sep_models(X,parts[x],train_idx,test_idx, y_train,classifier,10,True,"boost",100)
				else: 
					path = '../files/'+task+'/'+ name +'-'+parts[x]+'-'+des+'-'+typed[y]+'.txt'
					data = np.genfromtxt(path, dtype=float, delimiter=',', names=None)
					X1 = data[:, :-1]
					X = scale(X1)
					avg_acc,loss,y_pred = train_sep_models(X,parts[x],train_idx,test_idx, y_train,classifier,10,False,"boost",100)
				#classifier = RandomForestClassifier(n_estimators=1000,max_depth=1,class_weight="balanced")
				#classifier = AdaBoostClassifier(n_estimators=1000)
				avg_accs.append(avg_acc)
				losses.append(get_loss(y_test,y_pred))
				test_accs.append(balanced_accuracy_score(y_test,y_pred))
				fscores.append(f1_score(y_test,y_pred,average='weighted'))
				preds.append(y_pred)

		    acct=[]
		    fscoret=[]
		    losst=[]
		    c = [6,5,4,3,2,1] 
		    for cpt in range(len(c)):
			    print("try",cpt)
			    n = len(parts)-c[cpt]
			    print(n)
			    w = weight_function(avg_accs,losses,0.4,weight[z])
			    new_y = []

			    for i in range(len(y_test)):
			        S0 = []
			        S1 = []
			        S2 = []
			        score1 = 0
			        score2 = 0
			        score3 = 0
			        for j in range(n):
			            if preds[j][i] == 0:
			                S0.append(w[j])
			            elif preds[j][i] == 1:
			                S1.append(w[j])
			            elif preds[j][i] == 2:
			                S2.append(w[j])		            		
			        score1 = np.sum(S0) # for depth data the median of alpha, for rgb the sum of acc
			        score2 = np.sum(S1) 
			        score3 = np.sum(S2)
			        '''
			        if score1>score2 or math.isnan(score2):
			            new_pred = 0
			        elif score1<score2 or math.isnan(score1):
			            new_pred = 1
			        elif score1 == score2:
			            print("equal")	
			            random.choice([0, 1])
			        '''
			        if (score1>score2 or math.isnan(score2)) and (score1>score3 or math.isnan(score3)) :
			            new_pred = 0
			        elif (score2>score1 or math.isnan(score1)) and (score2>score3 or math.isnan(score3)):
			            new_pred = 1
			        elif (score3>score1 or math.isnan(score1)) and (score3>score2 or math.isnan(score2)):
			            new_pred = 2
			        new_y.append(new_pred)
			    print(len(new_y),len(y_test))
			    print("acc",balanced_accuracy_score(y_test, new_y))
			    print("fscore",f1_score(y_test, new_y,average='weighted'))
			    print("loss",get_loss(y_test,new_y))
			    acct.append(balanced_accuracy_score(y_test, new_y))
			    fscoret.append(f1_score(y_test, new_y,average='weighted'))
			    losst.append(get_loss(y_test,new_y))

		    acc1  = round(acct[0],2)
		    acc2  = round(acct[1],2)
		    acc3  = round(acct[2],2)
		    acc4  = round(acct[3],2)
		    acc5  = round(acct[4],2)
		    fs1 = round(fscoret[0],2)
		    fs2 = round(fscoret[1],2)
		    fs3 = round(fscoret[2],2)
		    fs4 = round(fscoret[3],2)
		    fs5 = round(fscoret[4],2)
		    l1 = round(losst[0],2)
		    l2 = round(losst[1],2)
		    l3 = round(losst[2],2)
		    l4 = round(losst[3],2)
		    l5 = round(losst[4],2)

		    plot_acc(acc1,acc2,acc3,acc4,acc5,fs1,fs2,fs3,fs4,fs5,l1,l2,l3,l4,l5)

