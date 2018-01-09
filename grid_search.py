#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:49:23 2018

@author: josharnold
"""

from data import preprocessing
from model import nn
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import time, os, shutil

# Seed numpy to aid in reproducability 
np.random.seed(7)

# Import data set
X, y = preprocessing.import_data()

# Split test train data
X_train, X_test, y_train, y_test, y_test_mol_names = preprocessing.split_train_and_test_data(X, y)

# Feature scaling X values
X_train, X_test = preprocessing.scale_x_data(X_train, X_test)

# Prepare parameters to grid search
optimizers = ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
kernal_init = ['uniform','lecun_uniform','normal','identity','orthogonal','zero','one','glorot_normal','glorot_uniform', 'he_normal', 'he_uniform']
epochs = [50, 100, 200, 300, 400]
batches = [5, 15, 25, 40]
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, kernal_init=kernal_init, activation=activations)

# Create model
model = KerasClassifier(build_fn=nn.baseline_model, verbose=0)
grid = GridSearchCV(cv=3, estimator=model, param_grid=param_grid, n_jobs=-1, verbose=10)

# Start timer
t0 = time.time() 

# GRID SEARCH RAWWWRRR!
grid_result = grid.fit(X_train, y_train)

# End timer
t1 = time.time()
train_time = t1-t0
m, s = divmod(train_time, 60)
h, m = divmod(m, 60)
time_string = "Gridsearch total run time: %dh:%02dm:%02ds" % (h, m, s) + "\n"
print("\n\n\n")
print(time_string)
print("\n\n\n")

# summarize results & output a .txt file
directory = "grid_search_output"
if not os.path.exists(directory):
    os.makedirs(directory)
else:
    shutil.rmtree(directory)  
    time.sleep(.5)
    os.makedirs(directory)
    
file_name = directory + "/" + "grid_search_output.txt"
f = open(file_name, "w+")
f.write(time_string + "\n")
best_string = "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_) + "\n"
f.write(best_string + "\n")

print("\n\n\n")
print(best_string)
print("\n\n\n")

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    string_to_print = "%f (%f) with: %r" % (mean, stdev, param) + "\n"
    print(string_to_print)
    f.write(string_to_print)
   
f.close()