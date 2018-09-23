#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:12:02 2018

@author: josharnold
"""

from data import data_bucket, postprocessing
import os

def export(extension, directory, y_train, y_test, y_pred):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(0, y_test.shape[0]):  
        # open            
        molecule_name = data_bucket.mol_names_y_test[i]
        
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
            x_str = str('%.2f' % (j+1))
            y_str = str('%.2f' % y_value)          
            if (y_value != 0):
                f.write(x_str + " " + y_str + "\n")
            
        # close
        f.close()

def export_single_MSP(extension, directory, y_train, y_test, y_pred):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_name = directory + "/" + "predicted-in-silico-spectra" + "." + extension
    f = open(file_name, "w+")    
    
    for i in range(0, y_test.shape[0]):  
        # name
        molecule_name = data_bucket.mol_names_y_test[i]
        f.write("Name: " + molecule_name + "\n")
        f.write("Formula:\n")
        f.write("MW:\n")
        f.write("CAS#:\n")
        f.write("Comments: in-silico spectrum "+ str(i+1) + "\n")
        
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
            x_str = str('%.2f' % (j+1))
            y_str = str('%.2f' % y_value)          
            if (y_value != 0):
                f.write(x_str + " " + y_str + "\n")
            
    # close
    f.close()    
