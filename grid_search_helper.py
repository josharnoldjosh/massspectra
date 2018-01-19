#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:30:01 2018

@author: josharnold
"""

import sys, fileinput, os, shutil, time, pickle
import numpy as np
from data import preprocessing

class helper:
    def get_data():
        # Seed numpy to aid in reproducability 
        np.random.seed(7)
        
        # Import data set
        X, y = preprocessing.import_data()
        
        # Split test train data
        X_train, X_test, y_train, y_test, y_test_mol_names = preprocessing.split_train_and_test_data(X, y)
        
        # Feature scaling X values
        X_train, X_test = preprocessing.scale_x_data(X_train, X_test)
        
        return X_train, X_test, y_train, y_test, y_test_mol_names
    
    def replaceAll(file,searchExp,replaceExp):
        for line in fileinput.input(file, inplace=1):
            if searchExp in line:
                line = line.replace(searchExp,replaceExp)
            sys.stdout.write(line)
            
    def check_grid_search_output(purge=False):
        # Make sure grid search dir exists and purge it if neccessary 
        directory = "grid_search_output"
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs((directory+"/logs"))
        elif(purge == True):
            shutil.rmtree(directory)  
            time.sleep(.5)
            os.makedirs(directory) 
            os.makedirs((directory+"/logs"))
            
            
    def manage_dirs(purge=False):
        checkpoint_load = 0
        models = []
        params = []
        results = []
        current_checkpoint_time = 0
        
        helper.check_grid_search_output(purge)
            
        directory2 = "grid_search_output/checkpoint.txt"
        if not os.path.exists(directory2):
            f = open(directory2, "w+")  
            f.write("0")
            f.close()
            f = open("grid_search_output/models.pkl", "w+")  
            f.write("0")
            f = open("grid_search_output/results.pkl", "w+")  
            f.write("0")
            f.close()
            f = open("grid_search_output/params.pkl", "w+")  
            f.write("0")
            f.close()
            f = open("grid_search_output/time_elapsed.txt", "w+")  
            f.write("0")
            f.close()
        else:
            f = open(directory2, "rb") 
            checkpoint_load = int(f.read())   
            f.close()
            f = open("grid_search_output/time_elapsed.txt", "rb") 
            current_checkpoint_time = float(f.read())   
            f.close()
            with open("grid_search_output/models.pkl", 'rb') as f:
                models = pickle.load(f)    
            with open("grid_search_output/params.pkl", 'rb') as f:
                params = pickle.load(f)
            with open("grid_search_output/results.pkl", 'rb') as f:
                results = pickle.load(f)
    
        return checkpoint_load, models, results, params, current_checkpoint_time
    
    def find_best_and_save_results(models, params, model_accs, result_losses, checkpoint_num, t0, X_test, y_test, previous_time_checkpoint):

        best_param = None
        best_param_loss = None
        
        model_acc_to_save = []
        result_loss_to_save = []
        
        best_acc = 0
        best_loss = 9999
        
        length = len(models) + len(model_accs)
        
        for i in range(0, length):
            acc = 0
            loss = 0
            
            if ((len(model_accs)-1) >= i):
                acc = model_accs[i]
                loss = result_losses[i]
                print("retrieved loss: ", loss, "\n")
                print("retrieved acc: ", acc, "\n")
                
            else:            
                model = models[i - len(model_accs)]
                score = model.evaluate(X_test, y_test)
                acc = score[1]
                loss = score[0]
                
            if (acc >= best_acc):
                best_acc = acc
                best_param = params[i]
                
            if (loss <= best_loss):
                best_loss = loss
                best_param_loss = params[i]
                    
            result_loss_to_save.append(loss)
            model_acc_to_save.append(acc)
                
        # Calculate time                                       
        t1 = time.time()
        train_time = t1 - t0
                
        m, s = divmod((train_time+previous_time_checkpoint), 60)
        h, m = divmod(m, 60)
        helper.replaceAll("grid_search_output/time_elapsed.txt", str(previous_time_checkpoint), str(train_time+previous_time_checkpoint))
        
        # Grid search output        
        directory = "grid_search_output/logs"       
        file_to_open = directory + "/" + str(train_time+previous_time_checkpoint) + ".txt"
        f = open(file_to_open, "w+")          
        
        f.write("Grid search elasped time: %dh:%02dm:%02ds" % (h, m, s))
        
        f.write("\n")
        f.write("\n")
        
        f.write("Number of models: " + str(length) + "\n")
        f.write("Length of acc array: " + str(len(model_acc_to_save)) + "\n")
        f.write("Length of loss array: " + str(len(result_loss_to_save)) + "\n")
        f.write("Length of param array: " + str(len(params)) + "\n")
        f.write("\n")

        f.write("(epoch, batch, opt, act, kern_init, w_1, w_2, w_3, d1, d2)")
        
        f.write("\n\n" + "Best acc: " + str(best_acc) + "\n")       
        f.write("Best acc param: " + str(best_param))
        
        f.write("\n\n" + "Best loss: " + str(best_loss) + "\n")
        f.write("Best loss param: " + str(best_param_loss))        
             
        f.close()
        
        # Update checkpoint 
        directory2 = "grid_search_output/checkpoint.txt"    
        helper.replaceAll(directory2,str(checkpoint_num-1), str(checkpoint_num))
        
        # Update grid search model acc & params
        with open("grid_search_output/models.pkl", 'wb') as f:
            pickle.dump(model_acc_to_save, f)            
        with open("grid_search_output/params.pkl", 'wb') as f:
            pickle.dump(params, f)
        with open("grid_search_output/results.pkl", 'wb') as f:
            pickle.dump(result_loss_to_save, f)     
        
        return train_time+previous_time_checkpoint