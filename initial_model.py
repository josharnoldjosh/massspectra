### currently best:
### Test accuracy: 0.681318685249
### using normalizer
### Test accuracy: 0.703296705262
### using batch size =20 and  Normalizer()   and  MaxAbsScaler()
### Test accuracy: 0.747252751183
### using model.add(Dense(800, kernel_initializer='uniform', activation='relu'))
### Test accuracy: 0.78021977105

# Functions
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

# SCRIPT PARAMTERES
epoch_amount = 10 #2000 # the amount of epochs for training
num_comparison_plots_to_show = 14 # number of spectrum similarity to show
show_and_save_all_plots = False

# Graph Permaters
peak_label_height = 100 # the threshold value to display the x value of a peak on the graph 
graph_width = 12 # height of the graphs displayed
graph_height = 7 # width of the graphs displayed
graph_label_text_y_offset = 35 # the amount tot offset the labels on the graph of the peaks
graph_trim_peak_height = 50
chart_bar_width = 0.4 # width of bar charts


import pandas as pd

# required for accuracy reporting reproducibility
import numpy as np
np.random.seed(12345)

from keras.models import Sequential 
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

# Import data set
data_filename = 'data.csv'
data = pd.read_csv(data_filename, sep=',', decimal='.', header=None)
y = data.loc[1:, 1:400].values
X = data.loc[1:, 401:1591].values

# Split data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# scaler = MinMaxScaler()   # Test accuracy: 0.670329687
scaler = Normalizer()       # Test accuracy: 0.714285716251
scaler = MaxAbsScaler()     # Test accuracy: 0.725274729205
#scaler = StandardScaler()  # Test accuracy: 0.571428572739
#scaler = RobustScaler()    # Test accuracy: 0.56043956175
#scaler = Normalizer() and 
#scaler = MaxAbsScaler()    #Test accuracy: 0.736263740194  

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Part 2: create ANN and fit data
def baseline_model():
    # Intialize the artificial neural network
    model = Sequential()

    # Input layer and hidden layer 
    model.add(Dense(activation="relu", input_dim=1191, units=700, kernel_initializer="glorot_normal"))
    # Dropout to avoid overfitting
    model.add(Dropout(0.2))
    
    # add another smaller layer
    model.add(Dense(800, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.2))
  
    # Output layer
    model.add(Dense(activation="relu", input_dim=700, units=400, kernel_initializer="uniform"))
     
    # Compile the ANN
    model.compile(optimizer="RMSprop", loss="mean_squared_error", metrics=["accuracy","mean_squared_error"])
    
    return model

# Keras callback save best models
# monitor for ['loss', 'acc', 'mean_squared_error', 'val_loss', 'val_acc', 'val_mean_squared_error']
checkpoint = ModelCheckpoint(filepath="best-model.hdf5",
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True)
                               
# Follow trends using tensorboard
# use source activate tensorflow
# start with tensorboard --logdir=logs
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

# enable eraly stopping
earlystopping=EarlyStopping(monitor='mean_squared_error', patience=100, verbose=1, mode='auto')

# Fit the ANN to the training set
model = baseline_model()

# calculate results, add callbacks ib needed
result = model.fit(X_train, y_train, batch_size=40, epochs=epoch_amount, validation_data=(X_test, y_test), callbacks=[earlystopping,checkpoint, tensorboard])


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


# Print final loss and accuracy 
score = model.evaluate(X_test, y_test)
print("")
print('Test score:', score[0]) 
print('Test accuracy:', score[1])
print("")


# Part 4: create prediction graphic
import matplotlib.pyplot as plt

# predict our y values
y_pred = model.predict(X_test) 

# collect names of compounds
mol_names = pd.read_csv("mol_names.csv", sep=',', decimal='.', header=None).values

if (show_and_save_all_plots == True):
    num_comparison_plots_to_show = y_test.shape[0]

for i in range(0, num_comparison_plots_to_show):    
    # print title
    print("Graph ", i, ":")

    # transform the "actual" y values to negative
    S = -1
    y_test_negative = S*y_test[i].astype(np.float)
        
    # adjust size of plot
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = graph_width # width
    fig_size[1] = graph_height # height
    plt.rcParams["figure.figsize"] = fig_size
    
    
    y_1_comp = y_test[i].astype(np.float)
    y_2_comp = y_pred[i]
    sim_value = cos_sim(y_1_comp, y_2_comp)
    print("Sim value!: ", sim_value)
    
    len_y_pred = len(y_pred[i])
    
    # search predicted values and limit the graph to the smallest value to "beautify" the graph from right
    prediction_j_trim_value = 0
    for j in range(0, len_y_pred):        
        index_cut_value = len_y_pred - j - 1
        
        peak_height_value = y_pred[i][index_cut_value] # start at the very right of graph, e.g, number         
        if (peak_height_value > graph_trim_peak_height):
            # we want to stop triming here
            prediction_j_trim_value = index_cut_value
            break;
            
    # search predicted values and limit the graph to the smallest value to "beautify" the graph from right
    actual_j_trim_value = 0
    for j in range(0, len_y_pred):        
        index_cut_value = len_y_pred - j - 1
        
        peak_height_value = y_test_negative[index_cut_value] * -1 # start at the very right of graph, e.g, number         
        if (peak_height_value > graph_trim_peak_height):
            # we want to stop triming here
            actual_j_trim_value = index_cut_value
            break;
            
    final_right_trim_value = 0
    if(prediction_j_trim_value > actual_j_trim_value):
        final_right_trim_value = prediction_j_trim_value        
    else:
        final_right_trim_value = actual_j_trim_value
            
    trimmed_prediction_array = y_pred[i][:final_right_trim_value]
    trimmed_actual_array = y_test_negative[:final_right_trim_value]
    
    # search predicted values and limit the graph to the smallest value to "beautify" the graph from left
    for j in range(0, len_y_pred):                
        peak_height_value = y_pred[i][j] # start at the very right of graph, e.g, number         
        if (peak_height_value > graph_trim_peak_height):
            # we want to stop triming here
            prediction_j_trim_value = j
            break;
            
    # search actual values and limit the graph to the smallest value to "beautify" the graph from left
    for j in range(0, len_y_pred):                
        peak_height_value = y_test_negative[j] * -1 # start at the very right of graph, e.g, number         
        if (peak_height_value > graph_trim_peak_height):
            # we want to stop triming here
            actual_j_trim_value = j
            break;
            
    final_left_trim_value = 0
    if(prediction_j_trim_value > actual_j_trim_value):
        final_left_trim_value = prediction_j_trim_value        
    else:
        final_left_trim_value = actual_j_trim_value
    
    trimmed_prediction_array = trimmed_prediction_array[final_left_trim_value:] 
    trimmed_actual_array = trimmed_actual_array[final_left_trim_value:] 
    
    
    
    
    # create basic argumentst needed to be passed intto tthe plt
    N = len(trimmed_prediction_array)
    x = range(N)

    # plot both prediction and actual values
    plt.bar(x, trimmed_prediction_array, chart_bar_width, color="blue")
    plt.bar(x, trimmed_actual_array, chart_bar_width, color="red")
    
    x_offset = len(trimmed_actual_array) / 100
    
    # label the peaks of actual values
    for j in range(0, len(trimmed_actual_array)):
        if (trimmed_actual_array[j] <= (peak_label_height*(-1))):
            # now label 
            x_value_for_label = j - x_offset
            y_value_for_label = trimmed_actual_array[j] - graph_label_text_y_offset - 25
            string_value_for_label = str(j)
            plt.text(x_value_for_label, y_value_for_label, string_value_for_label)
            
    # label the peaks of predictted values
    for j in range(0, len(trimmed_prediction_array)):
        if (trimmed_prediction_array[j] >= (peak_label_height)):
            # now label 
            x_value_for_label = j - x_offset
            y_value_for_label = trimmed_prediction_array[j] + graph_label_text_y_offset
            string_value_for_label = str(j)
            plt.text(x_value_for_label, y_value_for_label, string_value_for_label)
            
            
    # add sim value
    sim_str = "Cosine similarity: " + str('%.3f' % sim_value)
    plt.annotate(sim_str, xy=(0.8, 0.95), xycoords='axes fraction')
    
    # add mol names
    plt.annotate("Unknown", xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(mol_names[i][0], xy=(0.02, 0.05), xycoords='axes fraction')
    
    
    # set different values of graph 
    plt.title('Spectrum Similarity')
    plt.ylabel('intensity %')
    plt.xlabel('m/z')
    
    # save figure
    file_save_name = "graphs/" + mol_names[i][0]
    plt.savefig(file_save_name)
    
    plt.show()


# Part 5: export predictions







