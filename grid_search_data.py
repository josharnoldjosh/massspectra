#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:46:29 2018

@author: josharnold
"""

from data import preprocessing, postprocessing
from grid_search_manager import file, writer
from model import nn

class helper():        
    def load_data():      
        # Create output directory 
        file.create_output_directory()        
        
        # Load data
        X, y = preprocessing.import_data()        
        X_train, X_test, y_train, y_test, y_test_mol_names = preprocessing.split_train_and_test_data(X, y)        
        X_train, X_test = preprocessing.scale_x_data(X_train, X_test)                 
        return X_train, X_test, y_train, y_test, y_test_mol_names
    
    def purge_directory():
        file.destroy_output_directory()
        return
    
    def create_neural_network(data):
        model = nn.baseline_model(optimizer=data.optimizer,
                                  kernal_init=data.kernal_init,
                                  activation=data.activation,
                                  l1_w=data.layer_weight[0], l2_w=data.layer_weight[1], l3_w=data.layer_weight[2],
                                  l1_d=data.dropout_weight[0], l2_d=data.dropout_weight[1])
        return model
    
    def extract_results(nerual_network, X_test, y_test, search_data):
        # Accuracy & Loss
        score = nerual_network.evaluate(X_test, y_test)        
        loss = score[0]
        accuracy = score[1]
        
        # Average cosine similarity 
        y_pred = nerual_network.predict(X_test)
        average = postprocessing.get_average_cosine_similarity(y_pred, y_test)
        
        # Create the result
        res = result()        
        res.accuracy = accuracy # Accuracy
        res.loss = loss # Loss
        res.average_sim = average
        res.params = search_data # Parameters
        res.checkpoint = checkpoint.current() # Checkpoint
        return res
    
    def save_output(result):
        
        # Save result
        all_results = file.load_array("results.pkl")
        all_results.append(result)
        file.save_object("results.pkl", all_results)        
        
        # Update best results
        helper.update_best_results()  
        
        # Write output 
        writer.log(result)
        writer.best_result()
            
        return
    
    def update_best_results():
        results = file.load_array("results.pkl")
        
        best_accuracy = 0        
        best_loss = 999999999
        best_av_sim = 0
        best_params = [0, 0, 0]
        checkpoints = [0, 0, 0]        
        
        for res in results:
            if (res.accuracy >= best_accuracy):
                best_accuracy = res.accuracy
                best_params[0] = res.params  
                checkpoints[0] = res.checkpoint

            if (res.loss <= best_loss):
                best_loss = res.loss    
                best_params[1] = res.params
                checkpoints[1] = res.checkpoint
                
            if (res.average_sim >= best_av_sim):
                best_av_sim = res.average_sim    
                best_params[2] = res.params
                checkpoints[2] = res.checkpoint
                
        best_result = result()
        best_result.accuracy = best_accuracy
        best_result.loss = best_loss
        best_result.average_sim = best_av_sim
        best_result.best_params = best_params
        best_result.best_param_checkpoints = checkpoints
        
        file.save_object("best_result.pkl", best_result)
        
        return
                
class grid_data:
    class params:
        optimizers = []
        activations = []
        kernal_inits = []        
        layer_weights = []
        dropout_weights = []        
        batch_sizes = []   
        epochs = []
        
        optimizer = ""
        activation = ""
        kernal_init = ""
        layer_weight = []
        dropout_weight = []
        batch_size = 0
        epoch = 0
        
    def load_default_params():        
        data = grid_data.params()
        
        data.optimizers = ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
        data.activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
        data.kernal_inits = ['uniform','lecun_uniform','normal','orthogonal','zero','one','glorot_normal','glorot_uniform', 'he_normal', 'he_uniform']
        
        data.layer_weights = [[900, 800, 900]]
        data.dropout_weights = [[0.2, 0.2]]
        
        data.batch_sizes = [20]        
        data.epochs = [10]
        
        return data
    
    def get_data_at_checkpoint(checkpoint, data=params):
        count = 0
        for dropout_weight in data.dropout_weights:
            for batch_size in data.batch_sizes:
                for layer_weight in data.layer_weights:
                    for optimizer in data.optimizers:
                        for activation in data.activations:
                            for kernal_init in data.kernal_inits:
                                for epoch in data.epochs:   
                                    if (count < checkpoint):
                                        count += 1
                                    else:
                                        data.optimizer = optimizer
                                        data.activation = activation
                                        data.kernal_init = kernal_init
                                        data.layer_weight = layer_weight
                                        data.dropout_weight = dropout_weight
                                        data.batch_size = batch_size 
                                        data.epoch = epoch
                                        return data    
                                    
    def max_number_of_checkpoints(data=params):
        count = 0
        for dropout_weight in data.dropout_weights:
            for batch_size in data.batch_sizes:
                for layer_weight in data.layer_weights:
                    for optimizer in data.optimizers:
                        for activation in data.activations:
                            for kernal_init in data.kernal_inits:
                                for epoch in data.epochs:   
                                    count += 1
        return count
                                                                       
class checkpoint:
    def current():
        cpoint = file.load_integer("checkpoint.pkl")
        print("Loaded checkpoint", cpoint)
        return cpoint
    
    def save(checkpoint):
        file.save_object("checkpoint.pkl", checkpoint)
        
class result:
    accuracy = 0
    loss = 0
    params = None
    checkpoint = 0
    best_params = None
    best_param_checkpoints = []
    average_sim = 0
    
