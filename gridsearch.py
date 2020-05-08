import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from functions import *
from classification import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from functions import *

task = "gender" #gender
name = "complexity2" #complexity2,race
des = "lbp"
clas = "svm"
typed="RGBD"

labels = np.genfromtxt('../files/labels/labels'+name+'.txt', dtype=int, delimiter=',', names=None)
parts = ["cheeknose","chin+mouth+jaw","chin","nose","eyes","nosemouth","face"]
# read data (files)
path = '../files/'+task+'/'+ name +'-'+parts[6]+'-'+des+'-'+typed+'.txt'
data = np.genfromtxt(path, dtype=float, delimiter=',', names=None)
labels = np.genfromtxt('../files/labels/labelscomplexity2.txt', dtype=int, delimiter=',', names=None)

Y = labels
X = data[:, :-1]
X1,Y1 = shuffle(X,Y)

scaler = MinMaxScaler(feature_range=(0, 1)) #rescale
X2 = scaler.fit_transform(X1)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X2, Y1, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
'''
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

'''
scores = ['precision', 'recall','roc_auc']

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [100, 200,500,1000,1500]
             }


DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", max_depth = None)

ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid,cv=10, scoring = score)
    clf.fit(X2,Y1)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)