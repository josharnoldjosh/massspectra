#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:44:27 2018

@author: josharnold
"""

from grid_search_data import helper, grid_data, checkpoint
from grid_search_manager import timer

# Optional destroy directory for fresh grid search
helper.purge_directory()

# Load data
X_train, X_test, y_train, y_test, y_test_mol_names = helper.load_data()

# Load grid search params
search_data = grid_data.load_default_params()

# Override epochs 
search_data.epochs = [5]

# Load checkpoint 
current_checkpoint = checkpoint.current()

while True:
    # Get data at checkpoint
    search_data = grid_data.get_data_at_checkpoint(current_checkpoint, search_data)  

    # Start timer 
    timer.start()      
    
    # Run neural network
    model = helper.create_neural_network(search_data)
    model.fit(X_train, y_train, batch_size=search_data.batch_size, epochs=search_data.epoch, validation_data=(X_test, y_test)) 
   
    # Save output 
    result = helper.extract_results(model, X_test, y_test, search_data)
    helper.save_output(result)
    
    # Update checkpoint
    current_checkpoint += 1
    checkpoint.save(current_checkpoint)
    
    # End grid search once grids are exhausted
    if current_checkpoint == grid_data.max_number_of_checkpoints(search_data):
        print("Script finished after", current_checkpoint, "checkpoints.") 
        break    