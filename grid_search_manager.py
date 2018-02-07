#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:47:39 2018

@author: josharnold
"""

import pickle, os, shutil, time

class file:
    directory = "grid_search_output/"
    
    def destroy_output_directory():
        if os.path.exists(file.directory):
            shutil.rmtree(file.directory)  
            time.sleep(.5)
            os.makedirs(file.directory) 
            os.makedirs((file.directory+"logs")) 
        return
    
    def create_output_directory():
        if not os.path.exists(file.directory):
            os.makedirs(file.directory)
            os.makedirs((file.directory+"logs"))
        return
    
    def save_object(file_name, value):
        dir_to_load = file.directory + file_name        
        with open(dir_to_load, 'wb') as f:
            pickle.dump(value, f)
    
    def load_integer(file_name):
        dir_to_load = file.directory + file_name          
        if os.path.exists(dir_to_load):                
            with open(dir_to_load, 'rb') as f:
                return pickle.load(f) 
        else: 
            with open(dir_to_load, 'wb') as f:
                pickle.dump(0, f)    
                return 0        
            
    def load_array(file_name):
        dir_to_load = file.directory + file_name            
        if os.path.exists(dir_to_load):                
            with open(dir_to_load, 'rb') as f:
                return pickle.load(f) 
        else: 
            with open(dir_to_load, 'wb') as f:
                pickle.dump([], f)    
                return []            

class writer:
    def log(result):
        # Get time
        time_string = timer.end()        
        
        directory = file.directory+"logs/"+time_string+".txt"
        f = open(directory, "w+")  
                        
        # Elapsed time
        f.write("Grid search elasped time: " + time_string)
        f.write("\n\n")
        
        # Checkpoint
        f.write("Checkpoint: " + str(result.checkpoint))
        f.write("\n\n")
        
        # Parameters
        p = result.params 
                  
        f.write("Optimizer, activation, kernal init: " + p.optimizer + ", " + p.activation + ", " + p.kernal_init)
        f.write("\n\n")    
        
        f.write("Layer weights, dropout, batch size, epoch: " + str(p.layer_weight) + ", " + str(p.dropout_weight) + ", " + str(p.batch_size) + ", " + str(p.epoch))
        f.write("\n\n")
        
        # Accuracy
        f.write("Accuracy: " + str(result.accuracy))
        f.write("\n\n")
        
        # Loss
        f.write("Loss: " + str(result.loss))
        f.write("\n\n")    
        
        # Sim
        f.write("Average cosine similarity: " + str(result.average_sim))
        f.write("\n\n")    
        
        f.close()
        return
       
    def best_result():
        best_result = file.load_integer("best_result.pkl")
        
        if best_result == 0:
            return
        
        directory = file.directory+"best_results.txt"
        f = open(directory, "w+")  
        
        # Accuracy
        f.write("Best accuracy: " + str(best_result.accuracy))
        f.write("\n\n")
        
        # Loss
        f.write("Best loss: " + str(best_result.loss))
        f.write("\n\n") 
        
        # Sim
        f.write("Best average cosine similarity: " + str(best_result.average_sim))
        f.write("\n\n")    
        
        # Best params
        f.write("1) Accuracy at checkpoint " + str(best_result.best_param_checkpoints[0]))
        f.write("\n")
        
        f.write("2) Loss at checkpoint " + str(best_result.best_param_checkpoints[1]))
        f.write("\n")  
        
        f.write("3) Average cosine similarity at checkpoint " + str(best_result.best_param_checkpoints[2]))
        f.write("\n")  
        
        f.write("\n") 
        
        for p in best_result.best_params:
            if p != None:                         
                f.write("Optimizer, activation, kernal init: " + p.optimizer + ", " + p.activation + ", " + p.kernal_init)
                f.write("\n")    
                
                f.write("Layer weights, dropout, batch size, epoch: " + str(p.layer_weight) + ", " + str(p.dropout_weight) + ", " + str(p.batch_size) + ", " + str(p.epoch))
                f.write("\n\n")
                
        f.close()
    
        return     

class timer:
    start_time = 0
    end_time = 0
    
    def start():
        timer.start_time = time.time()
        return
    
    def end():
        timer.end_time = time.time() - timer.start_time
        
        time_elapsed = file.load_integer("time.pkl")
        time_elapsed += timer.end_time
        file.save_object("time.pkl", time_elapsed)
        
        string_value = timer.number_to_string(time_elapsed)   
        
        return string_value
    
    def number_to_string(time):            
        m, s = divmod(time, 60)
        h, m = divmod(m, 60)
        string_value = ("%dh %02dm %02ds" % (h, m, s))
        return string_value