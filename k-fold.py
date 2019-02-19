import settings
from data import preprocessing
from data import KFold
from model import nn
import graph
from data import postprocessing
import msp
import numpy as np

# Seed numpy to aid in reproducability 
np.random.seed(7)

# Import data set
data = KFold(num_k_fold=5)

cosine_sim = []

for (X_train, y_train), (X_test, y_test) in data:	
	print("\nStarting new K-fold")

	# Feature scaling X values
	X_train, X_test = preprocessing.scale_x_data(X_train, X_test)	

	# Drop molecule names from train & test split
	y_train, y_test = preprocessing.drop_mol_names_from_y_train_and_test(y_train, y_test)       

	# train model & get y prediction
	models, model_results, y_pred, train_time = nn.train_model(settings.num_models_to_average, X_train, y_train, X_test, y_test)

	# print acc and loss graphs (optional)
	# postprocessing.summarize_results(model_results)

	# Print and save mass spectra graphs 
	# graph.save(y_pred, y_test)
    
	# Export predictions in MSP
	# msp.export("msp", "msp", y_train, y_test, y_pred)
	# msp.export("txt", "msp_txt", y_train, y_test, y_pred) 
	# msp.export_single_MSP("msp", "msp-combined",y_train, y_test, y_pred)

	# Export a list of similarity values
	# graph.sim_values("sim_value_output", "sim_values", y_pred, y_test, new_line='\t')

	# print average score, loss, and cosine similarity
	postprocessing.print_all_scores(models, X_test, y_test)
	postprocessing.print_av_score(models, X_test, y_test, train_time)	
	cosine_sim += [postprocessing.print_average_cosine_similarity(y_pred, y_test)] # add to array to average over all k-folds

print("K-fold averaged cosine similarity: ", sum(cosine_sim)/len(cosine_sim))
print("Script finished.")