#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:27:59 2018

@author: josharnold
"""

import settings
from data import postprocessing
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, time

def get_right_trim_value(y_value_array):
    len_y_value = len(y_value_array)
    
    # search predicted values and limit the graph to the smallest value to "beautify" the graph from right
    prediction_j_trim_value = 0
    for j in range(0, len_y_value):        
        index_cut_value = len_y_value - j - 1
        
        peak_height_value = y_value_array[index_cut_value] # start at the very right of graph, e.g, number         
        if (peak_height_value > settings.graph_trim_peak_height):
            # we want to stop triming here
            prediction_j_trim_value = index_cut_value
            break;
    
    return prediction_j_trim_value

def get_left_trim_value(y_value_array):
    len_y_value = len(y_value_array)
    prediction_j_trim_value = 0
    for j in range(0, len_y_value):                
        peak_height_value = y_value_array[j] # start at the very right of graph, e.g, number         
        if (peak_height_value > settings.graph_trim_peak_height):
            # we want to stop triming here
            prediction_j_trim_value = j
            break;
    return prediction_j_trim_value

def get_trim_values(y_pred, y_actual):
    # trim from right
    prediction_j_trim_value = get_right_trim_value(y_pred)  
    actual_j_trim_value = get_right_trim_value(y_actual)
    
    final_right_trim_value = 0
    if(prediction_j_trim_value > actual_j_trim_value):
        final_right_trim_value = prediction_j_trim_value        
    else:
        final_right_trim_value = actual_j_trim_value            
    trimmed_prediction_array = y_pred[:final_right_trim_value]
    trimmed_actual_array = y_actual[:final_right_trim_value]
    
    # trim from left
    prediction_j_trim_value = get_left_trim_value(y_pred)
    actual_j_trim_value = get_left_trim_value(y_actual)
    
    final_left_trim_value = 0
    if(prediction_j_trim_value > actual_j_trim_value):
        final_left_trim_value = prediction_j_trim_value        
    else:
        final_left_trim_value = actual_j_trim_value
    
    trimmed_prediction_array = trimmed_prediction_array[final_left_trim_value:] 
    trimmed_actual_array = trimmed_actual_array[final_left_trim_value:] 
    
    return trimmed_prediction_array, trimmed_actual_array

def plot_mass_spectra_graph(y_pred_value, y_test_negative, mol_name_number):
    # adjust size of plot
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = settings.graph_width # width
    fig_size[1] = settings.graph_height # height
    plt.rcParams["figure.figsize"] = fig_size
    
    # create basic argumentst needed to be passed intto tthe plt
    N = len(y_pred_value)
    x = range(N)

    # plot both prediction and actual values
    plt.bar(x, y_pred_value, settings.chart_bar_width, color="blue")
    plt.bar(x, y_test_negative, settings.chart_bar_width, color="red")
    
    x_offset = len(y_test_negative) / 200
    
    # label the peaks of actual values
    for j in range(0, len(y_test_negative)):
        if (y_test_negative[j] <= (settings.peak_label_height*(-1))):
            # now label 
            x_value_for_label = j - x_offset
            y_value_for_label = y_test_negative[j] - settings.graph_label_text_y_offset - 25
            string_value_for_label = str(j)
            plt.text(x_value_for_label, y_value_for_label, string_value_for_label)
            
    # label the peaks of predicted values
    for j in range(0, len(y_pred_value)):
        if (y_pred_value[j] >= (settings.peak_label_height)):
            # now label 
            x_value_for_label = j - x_offset
            y_value_for_label = y_pred_value[j] + settings.graph_label_text_y_offset
            string_value_for_label = str(j)
            plt.text(x_value_for_label, y_value_for_label, string_value_for_label)
            
    # add sim value
    sim_value = postprocessing.cos_sim((y_test_negative*(-1)), y_pred_value)
    sim_str = "Cosine similarity: " + str('%.3f' % sim_value)
    plt.annotate(sim_str, xy=(0.8, 0.95), xycoords='axes fraction')
    
    # add mol names
    molecule_name_for_graph = postprocessing.get_molecule_name(mol_name_number)
    plt.annotate("Unknown", xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(molecule_name_for_graph, xy=(0.02, 0.05), xycoords='axes fraction')
    
    # set different values of graph 
    plt.title('Spectrum Similarity')
    plt.ylabel('intensity %')
    plt.xlabel('m/z')
    
    # save figure
    file_save_name = "graphs/" + molecule_name_for_graph
    plt.savefig(file_save_name)
    
    plt.show()
    
def print_and_save_mass_spectra_graphs(y_pred, y_test, y_test_mol_names):
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    else:
        shutil.rmtree('graphs')  
        time.sleep(.5)
        os.makedirs("graphs")    
    
    if (settings.show_and_save_all_plots == True):
        settings.num_comparison_plots_to_show = y_test.shape[0]
        
    for i in range(0, settings.num_comparison_plots_to_show):            
        # get y prediction & y actual (named y negative)
        y_pred_value = y_pred[i]
        y_test_negative = (y_test[i].astype(np.float)) * (-1)
        
        # Trim y prediction and y actual values if true in settings
        if (settings.should_trim_graphs == True):
            trimmed_prediction_array, trimmed_actual_array = get_trim_values(y_pred_value, (y_test_negative * (-1)))
            y_pred_value = trimmed_prediction_array
            y_test_negative = trimmed_actual_array * (-1)
            
        # Plot the graph
        mol_name_number = y_test_mol_names[i]
        plot_mass_spectra_graph(y_pred_value, y_test_negative, mol_name_number)
        
def save_list_of_sim_values_to_file(directory, file_name, y_pred, y_test, y_test_mol_names):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)  
        time.sleep(.5)
        os.makedirs(directory) 
    
    file_to_open = directory + "/" + file_name + ".txt"
    f = open(file_to_open, "w+")  
    
    for i in range(0, len(y_pred)):         
        molecule_name = postprocessing.get_molecule_name(y_test_mol_names[i])
        
        # Check this is calculated right:
        sim_value = postprocessing.cos_sim((y_test[i].astype(np.float)), y_pred[i])                
        sim_value_string = str('%.3f' % (sim_value*1000))
        
        string_to_write = str(i+1) + " " + sim_value_string + " " + molecule_name + "\n"
        f.write(string_to_write)
        
    f.close()