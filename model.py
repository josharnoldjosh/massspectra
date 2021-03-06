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
from keras.layers import Dense, Dropout, GaussianNoise, BatchNormalization
from keras import regularizers

from random import randint
import time

class nn: 
    def baseline_model(optimizer='Nadam', kernal_init='glorot_uniform', activation='relu', input_dim_val=2858, output_dim_val=500, l1_w=2500, l2_w=2500, l3_w=2500, l1_d=0.4, l2_d=0.4):
        model = Sequential()
        
        random_upper_bound = 50
        if (settings.num_models_to_average == 1):
            random_upper_bound = 0
        
        layer_1_weights = l1_w + randint(0,random_upper_bound)
        layer_2_weights = l2_w + randint(0,random_upper_bound)
        layer_3_weights = l3_w + randint(0,random_upper_bound)
        
        model.add(Dense(activation=activation, input_dim=input_dim_val, units=layer_1_weights, kernel_initializer=kernal_init, 
                        kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(l1_d))   
         
        model.add(BatchNormalization())
        model.add(Dense(layer_2_weights, kernel_initializer=kernal_init, activation=activation ))
        model.add(Dropout(l2_d))  
        
        model.add(GaussianNoise(30))
        
        model.add(BatchNormalization())
        model.add(Dense(activation=activation, input_dim=layer_3_weights, units=output_dim_val, kernel_initializer=kernal_init))     
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=settings.model_metrics)
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
        
        c_backs = [earlystopping,checkpoint, tensorboard]
        
        for i in range(0,num_models_to_average):
            time.sleep(1)
            model = nn.baseline_model(input_dim_val=X_train.shape[1], output_dim_val=y_train.shape[1])
            result = model.fit(X_train, y_train,                                
                               batch_size=settings.batch_size, epochs=settings.epoch_amount, 
                               validation_data=(X_test, y_test), 
                               callbacks=c_backs)
            models.append(model)
            results.append(result)      
            
        t1 = time.time()
        train_time = t1-t0
            
        return models, results, postprocessing.return_av_y_pred(models, X_test, y_test), train_time
    
    def summarize(models):
        for model in models:
            model.summary()