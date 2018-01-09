#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:54:14 2018

@author: josharnold
"""
import settings
from data import postprocessing
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Sequential 
from keras.layers import Dense, Dropout
from random import randint
import time

class nn:
    def baseline_model(optimizer='RMSprop', kernal_init='glorot_normal', activation='relu'):
        model = Sequential()
        
        random_upper_bound = 50
        if (settings.num_models_to_average == 1):
            random_upper_bound = 0
        
        layer_1_weights = 900 + randint(0,random_upper_bound)
        layer_2_weights = 800 + randint(0,random_upper_bound)
        layer_3_weights = 900 + randint(0,random_upper_bound)
    
        model.add(Dense(activation=activation, input_dim=1191, units=layer_1_weights, kernel_initializer=kernal_init))
        model.add(Dropout(0.2))   
         
        model.add(Dense(layer_2_weights, kernel_initializer=kernal_init, activation=activation))
        model.add(Dropout(0.2))  
        
        model.add(Dense(activation=activation, input_dim=layer_3_weights, units=400, kernel_initializer=kernal_init))     
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy","mean_squared_error"])
        return model
    
    # Train the model
    def train_model(num_models_to_average, X_train, y_train, X_test, y_test):
        # start timing the model
        t0 = time.time() 
        
        models = []
        results = []
        
        checkpoint = ModelCheckpoint(filepath="best-model.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        earlystopping=EarlyStopping(monitor='mean_squared_error', patience=100, verbose=1, mode='auto')
        
        for i in range(0,num_models_to_average):
            time.sleep(1)
            model = nn.__baseline_model()
            result = model.fit(X_train, y_train, batch_size=40, epochs=settings.epoch_amount, validation_data=(X_test, y_test), callbacks=[earlystopping,checkpoint, tensorboard])
            models.append(model)
            results.append(result)      
            
        t1 = time.time()
        train_time = t1-t0
            
        return models, results, postprocessing.return_av_y_pred(models, X_test, y_test), train_time