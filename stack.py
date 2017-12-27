#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:27:42 2017

@author: josharnold
"""

import pandas as pd

# Import data set
data_filename = 'data.csv'
data = pd.read_csv(data_filename, sep=',', decimal='.', header=None)
y = data.loc[1:, 1:400].values
X = data.loc[1:, 401:1591].values

# Split data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Let the stack begin!

from sklearn.svm import SVC
clf = SVC()
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("SVC: ", score)

#from sklearn.ensemble import RandomForestRegressor
#clf = RandomForestRegressor(n_estimators=100, n_jobs=1)
#clf = clf.fit(X_train, y_train)
#score = clf.score(X_test, y_test)
#print("Random forest: ", score)

# Create 5 objects that represent our 4 models
#rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
#et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
#ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
#gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
#svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)