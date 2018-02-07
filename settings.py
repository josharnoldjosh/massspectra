#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:08:47 2018

@author: josharnold
"""

# Nerual network Parameters
filename='data_v2.csv' # Name of file

input_dim = 1191 # The number of X values

output_dim = 800 # The number of y values



# SCRIPT PARAMTERES
epoch_amount = 8000 # the amount of epochs for training

num_models_to_average = 1 # average multiple models, if 1, no random neurons are added for the model because slightly different models are generated when averaging

num_comparison_plots_to_show = 1 # number of spectrum similarity to show (don't make this over the max value otherwise it will crash)

show_and_save_all_plots = False # This overrides the number of comparison plots to show and plots the max number of plots



# Graph Parameters
peak_label_height = 500 # the threshold value to display the x value of a peak label, e.g "57", on the graph 

graph_width = 27 # height of the graphs displayed

graph_height = 8 # width of the graphs displayed (note there exists a glitch with peak label height if graphs are too narrow)

graph_label_text_y_offset = 20 # the amount tot offset the labels on the graph of the peaks so they don't overlap the graphed bars

graph_trim_peak_height = 10 # Used for a threshold cut of value for beautifying the graphs

chart_bar_width = 0.4 # width of bar charts

should_trim_graphs = True # wether or not to make the graphs look beautiful