#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:08:47 2018

@author: josharnold
"""

"""
Import data functions. Must return the X and y dataframes. Edit this to make sure your data fits the input format.
"""
def import_data():
    import pandas as pd
    filename = "data/alkanes.csv"
    df = pd.read_csv(filename, sep=',', decimal='.', header=1)
    X = df.loc[:, 'x1':'x1191'] 
    y = df.loc[:, 'y1':'y800']
    mol_names = df.loc[:, 'NAME']
    y = pd.concat([mol_names, y], axis=1) # y must have molecule names as first column 
    return X.values, y.values

"""
K - fold cross validation import
"""
def import_data_for_k_fold():
    import pandas as pd
    filename = "data/alkanes.csv"
    df = pd.read_csv(filename, sep=',', decimal='.', header=1)
    return df

"""
Loss function to optimize.
"""
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))
model_metrics = [r_square,  "mean_squared_error"] # Metrics for training the model

"""
Plotting functions.
"""
def plot_result(result):
    import matplotlib.pyplot as plt 

    plt.plot(result["r_square"], label="train")
    plt.plot(result["val_r_square"], label="validation") # Add val infront of metrics for the validation
    plt.title("R^2")
    plt.ylabel('R^2')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    plt.plot(result["mean_squared_error"], label="train")
    plt.plot(result["val_mean_squared_error"], label="validation")
    plt.title("Mean Squared Error")
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()
    
    print("\n\nPossible dictionary keys include:", result.keys(), "\n\n")
    
    return

# SCRIPT PARAMTERES
epoch_amount = 5 #50 # the amount of epochs for training
batch_size = 20 # originally 40, but batch size of 20 gives better results
num_models_to_average = 1 # average multiple models, if 1, no random neurons are added for the model because slightly different models are generated when averaging
test_train_split_value = 0.2