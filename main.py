import settings
from data import preprocessing
from model import nn
import graph
from data import postprocessing
import msp
import numpy as np

# Seed numpy to aid in reproducability 
np.random.seed(7)

# Import data set
X, y = preprocessing.import_data()

# Split test train data
X_train, X_test, y_train, y_test, y_test_mol_names = preprocessing.split_train_and_test_data(X, y)

# Feature scaling X values
X_train, X_test = preprocessing.scale_x_data(X_train, X_test)
        
# train model & get y prediction
models, model_results, y_pred = nn.train_model(settings.num_models_to_average, X_train, y_train, X_test, y_test)

# print acc and loss graphs (optional)
postprocessing.summarize_results(model_results)

# Print and save mass spectra graphs 
graph.print_and_save_mass_spectra_graphs(y_pred, y_test, y_test_mol_names)
    
# Export predictions in MSP
msp.export("msp", "msp", y_train, y_test, y_pred, y_test_mol_names)
msp.export("txt", "msp_txt", y_train, y_test, y_pred, y_test_mol_names) 

# print average score and lost
postprocessing.print_all_scores(models, X_test, y_test)
postprocessing.print_av_score(models, X_test, y_test)
        
print("Script finished.")