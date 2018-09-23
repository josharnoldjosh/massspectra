#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:27:59 2018

@author: josharnold
"""

from data import postprocessing, data_bucket
import numpy as np
import os, shutil, time
    
def save(y_pred, y_test):
    print("Need to implement graph saving functionality.")
    pass
        
def sim_values(directory, file_name, y_pred, y_test, new_line='\n'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)  
        time.sleep(.5)
        os.makedirs(directory) 
    
    file_to_open = directory + "/" + file_name + ".txt"
    f = open(file_to_open, "w+")  
    
    for i in range(0, len(y_pred)):         
        molecule_name = data_bucket.mol_names_y_test[i]
        
        # Check this is calculated right:
        sim_value = postprocessing.cos_sim((y_test[i].astype(np.float)), y_pred[i])                
        sim_value_string = str('%.3f' % (sim_value*1000))
        
        string_to_write = str(i+1) + " " + sim_value_string + " " + molecule_name + new_line
        f.write(string_to_write)
        
    f.close()