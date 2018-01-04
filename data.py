#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:56:28 2018

@author: josharnold
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
import numpy as np

class preprocessing:
    def import_data():
        # Import data set
        data_filename = 'data.csv'
        data = pd.read_csv(data_filename, sep=',', decimal='.', header=None)
        y = data.loc[1:, 1:400].values
        X = data.loc[1:, 401:1591].values
        return X, y
    
    def scale_x_data(X_train, X_test):
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test
        
    # Split data into test and train sets
    def split_train_and_test_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

class postprocessing:
    def summarize_results(results):
        for result in results:
            # summarize history for accuracy
            plt.plot(result.history['acc'])
            plt.plot(result.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            
            # summarize history for loss
            plt.plot(result.history['loss'])
            plt.plot(result.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            
    def print_av_score(models, X_test, y_test):
        total_acc = 0
        total_loss = 0
        counter = 0
        for model in models:
            score = model.evaluate(X_test, y_test)
            total_acc += score[1]
            total_loss += score[0]
            counter += 1
        av_acc_score = total_acc / counter
        av_acc_loss = total_loss / counter
        print("")
        print("")
        print('Test score:', av_acc_loss) 
        print('Average test accuracy:', av_acc_score)
        print("")
        print("")
        
    def print_all_scores(models, X_test, y_test):
        counter = 0
        for model in models:
            counter += 1
            score = model.evaluate(X_test, y_test)
            acc = score[1]
            loss = score[0]
            print("")
            print("")
            print('Model', counter, ' score:', loss) 
            print('Model', counter, ' acc:', acc) 
            print("")
            print("")
    
    def return_av_y_pred(models, X_test, y_test):
        y_pred_original = models[0].predict(X_test) 
        y_pred_total = y_pred_original
        counter = 1
        for model in models:
            if (model != models[0]):
                y_pred = model.predict(X_test) 
                y_pred_total = y_pred_total + y_pred
                counter += 1
        return y_pred_total / counter
    
    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
