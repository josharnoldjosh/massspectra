#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:08:47 2018

@author: josharnold
"""

# SCRIPT PARAMTERES
epoch_amount = 10 #2000 # the amount of epochs for training

num_models_to_average = 1 # average multiple models, if 1, no random neurons are added for the model because slightly different models are generated when averaging

num_comparison_plots_to_show = 1 # number of spectrum similarity to show

show_and_save_all_plots = False



# Graph Permaters
peak_label_height = 500 # the threshold value to display the x value of a peak on the graph 

graph_width = 27 # height of the graphs displayed

graph_height = 8 # width of the graphs displayed

graph_label_text_y_offset = 20 # the amount tot offset the labels on the graph of the peaks

graph_trim_peak_height = 10

chart_bar_width = 0.4 # width of bar charts

should_trim_graphs = False