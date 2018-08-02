#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:08:47 2018

@author: josharnold
"""

# Nerual network Parameters
filename='MONADB-EI-7017x500.csv' # Name of file alkane=data_v2.csv
mol_name_data_dir = "data/MONADB-EI-7017x500-names.csv" # path to mol names data

input_dim = 2858 # The number of X values alkanes=(1191)

output_dim = 500 # The number of y values alkanes=(800)

# SCRIPT PARAMTERES
epoch_amount = 800 # the amount of epochs for training

batch_size = 100 # originally 40, but batch size of 20 gives better results

num_models_to_average = 20 # average multiple models, if 1, no random neurons are added for the model because slightly different models are generated when averaging

num_comparison_plots_to_show = 5 # number of spectrum similarity to show (don't make this over the max value otherwise it will crash)

show_and_save_all_plots = False # This overrides the number of comparison plots to show and plots the max number of plots

# Data import parameters

test_train_split_value = 0.2

# Graph Parameters
peak_label_height = 500 # the threshold value to display the x value of a peak label, e.g "57", on the graph 

graph_width = 27 # height of the graphs displayed

graph_height = 8 # width of the graphs displayed (note there exists a glitch with peak label height if graphs are too narrow)

graph_label_text_y_offset = 20 # the amount tot offset the labels on the graph of the peaks so they don't overlap the graphed bars

graph_trim_peak_height = 10 # Used for a threshold cut of value for beautifying the graphs

chart_bar_width = 0.4 # width of bar charts

should_trim_graphs = False # wether or not to make the graphs look beautiful