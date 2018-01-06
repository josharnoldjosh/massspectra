#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:12:02 2018

@author: josharnold
"""

from data import postprocessing
import os

def export(extension, directory, y_train, y_test, y_pred, y_test_mol_names):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(0, y_test.shape[0]):  
        # open    
        molecule_name = postprocessing.get_molecule_name(y_test_mol_names[i])
        
        file_name = directory + "/" + molecule_name + "." + extension
        f = open(file_name, "w+")
        
        # name
        f.write("Name: In-silico spectrum " + str(i+1) + "\n")
        f.write("Formula:\n")
        f.write("MW:\n")
        f.write("CAS#:\n")
        f.write("Comments: in-silico spectrum\n")
        
        # num peaks
        num_peaks = 0
        for j in range(0, len(y_pred[i])):  
            y_value = y_pred[i][j]                
            if (y_value != 0):
                num_peaks += 1
                            
        f.write("Num peaks: " + str(num_peaks) + "\n")
        
        # write peaks             
        for j in range(0, len(y_pred[i])):  
            y_value = y_pred[i][j]        
            x_str = str('%.2f' % j)
            y_str = str('%.2f' % y_value)          
            if (y_value != 0):
                f.write(x_str + " " + y_str + "\n")
            
        # close
        f.close()
