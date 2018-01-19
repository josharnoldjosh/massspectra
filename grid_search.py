#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:49:23 2018

@author: josharnold
"""
from grid_search_helper import helper
from grid_search_params import grids
from model import nn
import time

# Get data
X_train, X_test, y_train, y_test, y_test_mol_names = helper.get_data()

# Prepare parameters to grid search
optimizers, activations, kernal_init = grids.default_opt_act_kern()
l1_w, l2_w, l3_w = grids.default_layer_weights()
d1_w, d2_w = grids.default_dropout()
epochs, batches = grids.default_batch_epochs()

# Override 
epochs = [50]
should_purge_checkpoints_and_restart_grid_search = True

# Params for keeping track of grid searching
models = []
results = []

params = []
model_accs = []
result_losses = []

checkpoint_num = 0
checkpoint_load = 0
current_checkpoint_time = 0

# Setup / load directories for the grid search
checkpoint_load, model_accs, result_losses, params, current_checkpoint_time = helper.manage_dirs(purge=should_purge_checkpoints_and_restart_grid_search)

# Keep track of time
t0 = time.time()
        
# Start brutal grid search
for d1 in d1_w:
    for d2 in d2_w:
        for batch in batches:
            for w_1 in l1_w:
                for w_2 in l2_w:
                    for w_3 in l3_w:
                        for opt in optimizers:
                            for act in activations:
                                for kern_init in kernal_init:
                                    for epoch in epochs:                                                                                                               
                                            checkpoint_num += 1
                                            
                                            if (checkpoint_load >= checkpoint_num):
                                                print("skiping ", checkpoint_num)
                                            else:                                                    
                                                time.sleep(0.5)  
                                                                                                                                                                                                                                        
                                                # Create model with specific params
                                                model = nn.baseline_model(optimizer=opt, 
                                                                          kernal_init=kern_init, 
                                                                          activation=act, 
                                                                          l1_w=w_1, l2_w=w_2, l3_w=w_3, 
                                                                          l1_d=d1, l2_d=d2)                                           
                                                
                                                # Fit model
                                                result = model.fit(X_train, y_train, 
                                                                   batch_size=batch, epochs=epoch, 
                                                                   validation_data=(X_test, y_test))                                                                                    
                                                
                                                # Add to array
                                                models.append(model)  
                                                results.append(result)
                                                group = (epoch, batch, opt, act, kern_init, w_1, w_2, w_3, d1, d2)
                                                params.append(group)
                                                
                                                current_checkpoint_time = helper.find_best_and_save_results(models, 
                                                                                                            params, 
                                                                                                            model_accs, result_losses, 
                                                                                                            checkpoint_num, t0, 
                                                                                                            X_test, y_test, 
                                                                                                            current_checkpoint_time)
                                            
print("Script finished")                                            