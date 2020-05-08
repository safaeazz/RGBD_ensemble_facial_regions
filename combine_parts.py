
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

task = "gender" #gender
name = "complexity2" #complexity2,race
des = "lbp"
clas = "svm"
typed=["RGB","DEPTH","RGBD"]
weight = ["accloss"]
#,"acc","meanlog","log"] 
seed = 42
parts = ["cheeknose","chin+mouth+jaw","eyes","chin","nose","nosemouth","face"]

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
		    train_idx,test_idx = train_test_split(idx_sh,test_size=.4,stratify=Y,random_state=seed)#changed
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
				fscores.append(f1_score(y_test,y_pred))
				preds.append(y_pred)

		    w = weight_function(avg_accs,losses,0.4,weight[z])
		    new_y = []
		    new_y_N = []
		    new_y2 = []
		    new_y3 = []

		    for i in range(len(y_test)):
		        S0 = []
		        S1 = []
		        SN0 = 0
		        SN1 = 0

		        score1 = 0
		        score2 = 0
		        score1w2 = 0
		        score2w2 = 0
		        score1w3 = 0
		        score2w3 = 0

		        for j in range(len(parts)-1):
		            if preds[j][i] == 0:
		            	S0.append(w[j])
		            	SN0 = SN0 +1
		            elif preds[j][i] == 1:
		            	S1.append(w[j])
		            	SN1 = SN1 + 1
		        score1 = np.sum(S0) # for depth data the median of alpha, for rgb the sum of acc
		        score2 = np.sum(S1) 

		        score1w2 = np.prod(S0) 
		        score2w2 = np.prod(S1) 
		        score1w3 = np.median(S0) 
		        score2w3 = np.median(S1) 

		        if score1>score2 or math.isnan(score2):
		            new_pred = 0
		        elif score1<score2 or math.isnan(score1):
		            new_pred = 1
		        elif score1 == score2:
		            print("equal")	
		            random.choice([0, 1])
		        if SN0 > SN1 :
		            new_pred_n = 0
		        elif SN0 < SN1 :
		            new_pred_n = 1
		        if score1w2>score2w2 or math.isnan(score2w2):
		            new_pred2 = 0
		        elif score1w2<score2w2 or math.isnan(score1w2):
		            new_pred2 = 1
		        if score1w3>score2w3 or math.isnan(score2w3):
		            new_pred3 = 0
		        elif score1w3<score2w3 or math.isnan(score1w3):
		            new_pred3 = 1
		        new_y.append(new_pred)
		        new_y_N.append(new_pred_n)
		        new_y2.append(new_pred2)
		        new_y3.append(new_pred3)

		    final_acc1.append(balanced_accuracy_score(y_test, new_y))
		    final_acc2.append(balanced_accuracy_score(y_test, new_y2))
		    final_acc3.append(balanced_accuracy_score(y_test, new_y3))

		    final_acc_part1.append(test_accs[0])
		    final_acc_part2.append(test_accs[1])
		    final_acc_part3.append(test_accs[2])
		    final_acc_part4.append(test_accs[3])
		    final_acc_part5.append(test_accs[4])
		    final_acc_part6.append(test_accs[5])

		    final_score_part1.append(fscores[0])
		    final_score_part2.append(fscores[1])
		    final_score_part3.append(fscores[2])
		    final_score_part4.append(fscores[3])
		    final_score_part5.append(fscores[4])
		    final_score_part6.append(fscores[5])

		    face_acc.append(balanced_accuracy_score(y_test, preds[6]))
		    final_fscore1.append(f1_score(y_test,new_y))
		    final_fscore2.append(f1_score(y_test,new_y2))
		    final_fscore3.append(f1_score(y_test,new_y3))
		    final_face_fscore.append(f1_score(y_test,preds[6]))
		    print("-----------------------------------------------------------------------------------------")
		    print("Experiment ",r)
		    print("-----------------------------------------------------------------------------------------")
		    #f = open('tableRGBRESULTS.txt', 'w')
		    print(tabulate([[parts[0],test_accs[0],fscores[0],losses[0]],
		                    [parts[1],test_accs[1],fscores[1],losses[1]],
		                    [parts[2],test_accs[2],fscores[2],losses[2]],
		                    [parts[3],test_accs[3],fscores[3],losses[3]],
		                    [parts[4],test_accs[4],fscores[4],losses[4]],
		                    [parts[5],test_accs[5],fscores[5],losses[5]],
		                    [parts[6],test_accs[6],fscores[6],losses[6]],
		                    ["non-weighted fusion",balanced_accuracy_score(y_test, new_y_N),f1_score(y_test, new_y_N),get_loss(y_test,new_y_N)],
		                    ["weighted fusion (sum)",balanced_accuracy_score(y_test, new_y),f1_score(y_test, new_y),get_loss(y_test,new_y)],
		                    ["weighted fusion (prod)",balanced_accuracy_score(y_test, new_y2),f1_score(y_test, new_y2),get_loss(y_test,new_y2)],
		                    ["weighted fusion (median)",balanced_accuracy_score(y_test, new_y3),f1_score(y_test, new_y3),get_loss(y_test,new_y3)]],
		                    headers=['part','test accuracy','f-score',"miss-classification rate"], tablefmt='fancy_grid'))



		print("-----------------------------------------------------------------------------------------")
		print("Average =================================================================================")
		print("-----------------------------------------------------------------------------------------")
		#f = open('tableRGBRESULTS.txt', 'w')
		print(tabulate([[parts[0],np.mean(final_acc_part1),np.mean(final_score_part1)],
		                [parts[1],np.mean(final_acc_part2),np.mean(final_score_part2)],
		                [parts[2],np.mean(final_acc_part3),np.mean(final_score_part3)],
		                [parts[3],np.mean(final_acc_part4),np.mean(final_score_part4)],
		                [parts[4],np.mean(final_acc_part5),np.mean(final_score_part5)],
		                [parts[5],np.mean(final_acc_part6),np.mean(final_score_part6)],
		                [parts[6],np.mean(face_acc),np.mean(final_face_fscore)],
		                ["non-weighted fusion",balanced_accuracy_score(y_test, new_y_N),f1_score(y_test, new_y_N),get_loss(y_test,new_y_N)],
		                ["weighted fusion (sum)",balanced_accuracy_score(y_test, new_y),f1_score(y_test, new_y),get_loss(y_test,new_y)],
		                ["weighted fusion (prod)",balanced_accuracy_score(y_test, new_y2),f1_score(y_test, new_y2),get_loss(y_test,new_y2)],
		                ["weighted fusion (median)",balanced_accuracy_score(y_test, new_y3),f1_score(y_test, new_y3),get_loss(y_test,new_y3)]],
		                headers=['part','test accuracy','f-score'], tablefmt='fancy_grid'))

		print("-----------------------------------------------------------------------------------------")
		print("Mean values")
		print("-----------------------------------------------------------------------------------------")

		print("mean acc sum = ",np.mean(final_acc1),np.mean(face_acc))
		print("mean acc prod = ",np.mean(final_acc2),np.mean(face_acc))
		print("mean acc median = ",np.mean(final_acc3),np.mean(face_acc))
		print("mean fscore sum = ",np.mean(final_fscore1),np.mean(final_face_fscore))	
		print("mean fscore prod = ",np.mean(final_fscore2),np.mean(final_face_fscore))	
		print("mean fscore median = ",np.mean(final_fscore3),np.mean(final_face_fscore))	

		_, _, _, p1 = independent_ttest(final_acc1, face_acc, .05)
		_, _, _, p2 = independent_ttest(final_acc2, face_acc, .05)
		_, _, _, p3 = independent_ttest(final_acc3, face_acc, .05)

		print('sum p1=%.3f' % (p1))
		print('prod p2=%.3f' % (p2))
		print('median p3=%.3f' % (p3))

		# interpret via critical value
		'''
		if abs(t_stat) <= cv:
			print('Accept null hypothesis that the means are equal.')
		else:
			print('Reject the null hypothesis that the means are equal.')
		# interpret via p-value
		if p < .05:
			print('Accept null hypothesis that the means are equal.')
		else:
			print('Reject the null hypothesis that the means are equal.')
		'''
		#plt.plot(final_acc,color='r')
		#plt.plot(face_acc,color='b')
		#plt.show()

