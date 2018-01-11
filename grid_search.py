#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:49:23 2018

@author: josharnold
"""

from data import preprocessing
from model import nn
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import time, shutil, os

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
kernal_init = ['uniform','lecun_uniform','normal','orthogonal','zero','one','glorot_normal','glorot_uniform', 'he_normal', 'he_uniform']

l1_w = [400, 500, 600, 700, 800, 900, 1000, 1200]
l2_w = [400, 500, 600, 700, 800, 900, 1000, 1200]
l3_w = [400,500, 600, 700, 800, 900, 1000, 1200]

d1_w = [0.1, 0.2, 0.5, 0.75]
d2_w = [0.1, 0.2, 0.5, 0.75]

epochs = [50, 100, 300, 500, 750, 1000, 1250]
batches = [10, 20, 40]

# Let's do grid search the ol' fashion way
t0 = time.time() 

checkpoint = ModelCheckpoint(filepath="best-model.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
earlystopping=EarlyStopping(monitor='mean_squared_error', patience=100, verbose=1, mode='auto')

models = []
params = []

directory = "grid_search_output"
if not os.path.exists(directory):
    os.makedirs(directory)
else:
    shutil.rmtree(directory)  
    time.sleep(.5)
    os.makedirs(directory) 

def find_best_and_save_results():
    best_acc = 0 
    best_param = None
    
    for i in range(0, len(models)):
        model = models[i]
        score = model.evaluate(X_test, y_test)
        acc = score[1]
        if (acc > best_acc):
            best_acc = acc
            best_param = params[i]
            
    # Calculate time                                       
    t1 = time.time()
    train_time = t1-t0
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)
    
    directory = "grid_search_output"       
    file_to_open = directory + "/" + str(train_time) + ".txt"
    f = open(file_to_open, "w+")  
    
    f.write("Grid search elasped time: %dh:%02dm:%02ds" % (h, m, s))
    f.write("\n\n" + "Best acc: " + str(best_acc) + "\n\n")
    f.write("(epoch, batch, opt, act, kern_init, w_1, w_2, w_3, d1, d2)\n")
    f.write("Best param: " + str(best_param))
    
    f.close()

for epoch in epochs:
    for batch in batches:
        for opt in optimizers:
            for act in activations:
                for kern_init in kernal_init:
                    for w_1 in l1_w:
                        for w_2 in l2_w:
                            for w_3 in l2_w:
                                for d1 in d1_w:
                                    for d2 in d2_w:
                                        time.sleep(1)
                                        
                                        # Create model with specific params
                                        model = nn.baseline_model(optimizer=opt, 
                                                                  kernal_init=kern_init, 
                                                                  activation=act, 
                                                                  l1_w=w_1, l2_w=w_2, l3_w=w_3, 
                                                                  l1_d=d1, l2_d=d2)
                                            
                                        # Fit model
                                        result = model.fit(X_train, y_train, batch_size=batch, 
                                                           epochs=epoch,
                                                           validation_data=(X_test, y_test), 
                                                           callbacks=[earlystopping,checkpoint, tensorboard])
                                        
                                        # Add to array
                                        models.append(model)  
                                        group = (epoch, batch, opt, act, kern_init, w_1, w_2, w_3, d1, d2)
                                        params.append(group)
                                        find_best_and_save_results()