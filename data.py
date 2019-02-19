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
import numpy as np

class KFold:    
    def __init__(self, num_k_fold=5):
        self.data = settings.import_data_for_k_fold()
        from sklearn.model_selection import KFold
        self.kf = KFold(n_splits = num_k_fold, shuffle = True, random_state = 2).split(self.data)
        self.fold = 0
        self.num_k_fold = num_k_fold

    def process_data(self, df):
        X = df.loc[:, 'x1':'x1191'] 
        y = df.loc[:, 'y1':'y800']
        mol_names = df.loc[:, 'NAME']
        y = pd.concat([mol_names, y], axis=1) # y must have molecule names as first column 
        return X.values, y.values

    def __iter__(self):
        return self

    def __next__(self):
        self.fold += 1
        if self.fold > self.num_k_fold:
            raise StopIteration
        split = next(self.kf, None)
        train = self.data.iloc[split[0]]
        test = self.data.iloc[split[1]]
        return self.process_data(train), self.process_data(test)

class data_bucket:
    mol_names_y_test = []

class preprocessing:
    def import_data():
        X, y = settings.import_data()
        return X, y
    
    def scale_x_data(X_train, X_test):
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test
    
    def drop_mol_names_from_y_train_and_test(y_train, y_test):              
        y_train_new = []
        for i in range(0, len(y_train)):
            y_train_new.append(np.delete(y_train[i], 0).tolist())
            
        y_test_new = []
        for i in range(0, len(y_test)):
            data_bucket.mol_names_y_test.append(y_test[i][0])
            y_test_new.append(np.delete(y_test[i], 0).tolist())
        
        return np.asarray(y_train_new), np.asarray(y_test_new)
        
    # Split data into test and train sets
    def split_train_and_test_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.test_train_split_value, random_state=0)        
        y_train, y_test = preprocessing.drop_mol_names_from_y_train_and_test(y_train, y_test)        
        return X_train, X_test, y_train, y_test

class postprocessing:
    def summarize_results(results):
        for result in results:
            settings.plot_result(result.history)
            """# summarize history for accuracy
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
            plt.show()"""
            
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
        i = 0
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
        return data_bucket.mol_names_y_test[int(molecule_name_number)]
    
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
        return postprocessing.get_average_cosine_similarity(y_pred, y_test)