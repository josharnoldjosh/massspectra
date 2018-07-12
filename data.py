#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:56:28 2018

@author: josharnold
"""

import settings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
import numpy as np

class preprocessing:
    def import_data():
        # Load info from settings
        filename=settings.filename 
        num_x=settings.input_dim
        num_y=settings.output_dim
        
        dir_name = 'data/' + filename
        data = pd.read_csv(dir_name, sep=',', decimal='.', header=None)
        y = data.loc[1:, 0:(num_y)]
        X = data.loc[1:, (num_y+1):(num_y+num_x)]
        return X.values, y.values
    
    def scale_x_data(X_train, X_test):
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test
    
    def drop_mol_names_from_y_train_and_test(y_train, y_test):      
        y_test_mol_names = []
        
        y_train_new = []
        for i in range(0, len(y_train)):
            y_train_new.append(np.delete(y_train[i], 0).tolist())
            
        y_test_new = []
        for i in range(0, len(y_test)):
            y_test_mol_names.append(y_test[i][0])
            y_test_new.append(np.delete(y_test[i], 0).tolist())    
        
        return np.asarray(y_train_new), np.asarray(y_test_new), y_test_mol_names
        
    # Split data into test and train sets
    def split_train_and_test_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.test_train_split_value, random_state=0)        
        y_train, y_test, y_test_mol_names = preprocessing.drop_mol_names_from_y_train_and_test(y_train, y_test)        
        return X_train, X_test, y_train, y_test, y_test_mol_names

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
            
    def print_av_score(models, X_test, y_test, train_time):
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
        
        m, s = divmod(train_time, 60)
        h, m = divmod(m, 60)
        print("")
        print("")
        print('Test score:', av_acc_loss) 
        print('Average test accuracy:', av_acc_score)
        print("NN total train time: %dh:%02dm:%02ds" % (h, m, s))
        
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
        res = dot_product / (norm_a * norm_b)
        return res 
    
    def get_molecule_name(molecule_name_number):
        mol_names = pd.read_csv(settings.mol_name_data_dir, sep=',', decimal='.', header=None).values
        molecule_name_for_graph = "Replace me"
        for molecule in mol_names:
            if (molecule[0] == molecule_name_number):
                molecule_name_for_graph = molecule[1]
        return molecule_name_for_graph
    
    def get_average_cosine_similarity(y_pred, y_test):
        import warnings
        with warnings.catch_warnings(): 
                warnings.simplefilter("ignore", category=RuntimeWarning)
                total_sim_value = 0
                for i in range(0, len(y_pred)):              
                    sim_value = postprocessing.cos_sim((y_test[i].astype(np.float)), y_pred[i].astype(np.float)) 
                    total_sim_value += np.nan_to_num(sim_value)
                average = total_sim_value / len(y_pred)
                return average

    def print_average_cosine_similarity(y_pred, y_test):        
        print("Average cosine similarity:", postprocessing.get_average_cosine_similarity(y_pred, y_test))
        return