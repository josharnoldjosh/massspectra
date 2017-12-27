### currently best:
### Test accuracy: 0.681318685249
### using normalizer
### Test accuracy: 0.703296705262
### using batch size =20 and  Normalizer()   and  MaxAbsScaler()
### Test accuracy: 0.747252751183
### using model.add(Dense(800, kernel_initializer='uniform', activation='relu'))
### Test accuracy: 0.78021977105

# SCRIPT PARAMTERES
epoch_amount = 2000 # the amount of epochs for training
peak_label_height = 500 # the threshold value to display the x value of a peak on the graph 
graph_width = 12 # height of the graphs displayed
graph_height = 7 # width of the graphs displayed
num_comparison_plots_to_show = 10 # number of spectrum similarity to show
graph_label_text_y_offset = 25 # the amount tot offset the labels on the graph of the peaks

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

for i in range(0, num_comparison_plots_to_show):    
    # print title
    print("Graph ", i, ":")

    # transform the "actual" y values to negative
    S = -1
    y_test_negative = S*y_test[i].astype(np.float)
    
    # create basic argumentst needed to be passed intto tthe plt
    N = len(y_pred[i])
    x = range(N)
    width = 0.5 # width of bar charts
    
    # adjust size of plot
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = graph_width # width
    fig_size[1] = graph_height # height
    plt.rcParams["figure.figsize"] = fig_size

    # plot both prediction and actual values
    plt.bar(x, y_pred[i], width, color="blue")
    plt.bar(x, y_test_negative, width, color="red")
    
    # label the peaks of actual values
    for j in range(0, len(y_test_negative)):
        if (y_test_negative[j] <= (peak_label_height*(-1))):
            # now label 
            x_value_for_label = j - 4.5
            y_value_for_label = y_test_negative[j] - graph_label_text_y_offset
            string_value_for_label = str(j)
            plt.text(x_value_for_label, y_value_for_label, string_value_for_label)
            
    # label the peaks of predictted values
    for j in range(0, len(y_pred[i])):
        if (y_pred[i][j] >= (peak_label_height)):
            # now label 
            x_value_for_label = j - 4.5
            y_value_for_label = y_pred[i][j] + graph_label_text_y_offset
            string_value_for_label = str(j)
            plt.text(x_value_for_label, y_value_for_label, string_value_for_label)
    
    # set different values of graph 
    plt.title('Spectrum Similarity')
    plt.ylabel('intensity %')
    plt.xlabel('m/z')
    plt.show()


# Part 5: export predictions



