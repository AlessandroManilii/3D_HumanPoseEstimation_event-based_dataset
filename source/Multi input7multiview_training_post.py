import random
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow.keras.backend as k 
import tensorflow.keras as keras
import time
from tensorflow.keras.models import load_model

start_time = time.time()

# Loss function 
def mse2D(y_true, y_pred):
  mean_over_ch = k.mean(k.square(y_pred - y_true), axis=-1)
  mean_over_w = k.mean(mean_over_ch, axis=-1)
  mean_over_h = k.mean(mean_over_w, axis=-1)
  return mean_over_h

model = load_model(r'/home/vrai/RiccardoRosati/checkpoint/model_19.h5', custom_objects={'mse2D': mse2D}, compile=False)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    # Define parameters
    def __init__(self, list_IDs, batch_size=32, dim=(260,344), joints=13): #minibatch_mult=64, 
        'Initialization'
        
        self.list_IDs = list_IDs
        #self.minibatch_mult = minibatch_mult
        self.batch_size = batch_size
        self.dim = dim
        self.joints = joints
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
      
        # Denotes the number of batches per epoch
        return (len(self.list_IDs) // self.batch_size) #self.minibatch_mult)   

    def __getitem__(self, index):

        # Generate indexes for a batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] #index*self.minibatch_mult:(index+1)*self.minibatch_mult]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
      
        # Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))       
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        # Initialization 
        x_2 = np.empty((self.batch_size, *self.dim, 1))
        x_3 = np.empty((self.batch_size, *self.dim, 1))
        y_2 = np.empty((self.batch_size, *self.dim, self.joints))
        y_3 = np.empty((self.batch_size, *self.dim, self.joints))
                
        count = 0

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            x_file_2 = np.load('/home/vrai/RiccardoRosati/dataset_2/x' + str(ID) + '.npy')
            x_file_3 = np.load('/home/vrai/RiccardoRosati/dataset_3/x' + str(ID) + '.npy')
            y_file_2 = np.load('/home/vrai/RiccardoRosati/dataset_2/y' + str(ID) + '.npy')
            y_file_3 = np.load('/home/vrai/RiccardoRosati/dataset_3/y' + str(ID) + '.npy')
            # Store sample
            x_2[count] = x_file_2[:,:,np.newaxis]
            x_3[count] = x_file_3[:,:,np.newaxis]       
            # Store label
            y_2[count] = y_file_2
            y_3[count] = y_file_3

            x = [x_2, x_3]
            y = [y_2, y_3]
            count += 1
            # if i == len(list_IDs_temp)-1:
            #   break
        return x, y

# Datasets
num_of_file = 62665
val_file = 16578  #=62665-46087 Validation data is 20% of training data

list_IDs = list(range(0, num_of_file))

train_set = list_IDs[0:-val_file]
validation_set = list_IDs[-val_file:]

# Generators
training_generator = DataGenerator(list_IDs = train_set)
validation_generator = DataGenerator(list_IDs = validation_set)



model.compile(
    # Optimizer
    optimizer='RMSprop',
    # Loss function to minimize
    loss = mse2D)

def scheduler(epoch):
  if epoch < 10:
    return 0.0001
  else:
    if epoch < 15:
      return 0.0001 
    else:
       return 0.00001

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):        
      self.model.save("/home/vrai/RiccardoRosati/checkpoint_post/model_{}.h5".format(epoch))

callback = [tf.keras.callbacks.LearningRateScheduler(scheduler), CustomSaver()]

# Train model on dataset
history = model.fit(training_generator,
                    validation_data=validation_generator,
                    epochs=4,
                    callbacks=callback,
                    verbose=1)

history.history

print("--- %s seconds ---" % (time.time() - start_time))

