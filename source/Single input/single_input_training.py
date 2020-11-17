import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import keras.backend as k
import random

# Define model
initializer = initializers.VarianceScaling(scale=1,mode='fan_avg',distribution='uniform',seed=None)
bias_init = initializers.zeros()

inputs = tf.keras.Input(shape=(260,344,1), dtype='float32', sparse=False)

conv1 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                      padding='same',data_format='channels_last',dilation_rate=(1,1),
                      use_bias=False, kernel_initializer=initializer, bias_initializer=bias_init,
                      kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None, 
                      activation = 'linear',kernel_constraint = None, bias_constraint=None)(inputs)

activation1 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv1)

pool1 = layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), 
                       data_format='channels_last')(activation1)

conv2a = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,
                       bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,
                       bias_constraint=None)(pool1)

activation2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2a)

conv2b = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation2)

activation3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2b)


conv2d = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation3)

activation4 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2d)

pool2 = layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2),
                       data_format='channels_last')(activation4)

conv3a = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',data_format='channels_last',
                       dilation_rate=(2,2),activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(pool2)

activation5 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3a)
                      
conv3b = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation5)

activation6 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3b)

conv3c = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',data_format='channels_last',
                       dilation_rate=(2,2),activation='linear',use_bias=False, kernel_initializer=initializer, bias_initializer=bias_init,
                       kernel_regularizer=None, bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation6)

activation7 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3c)

conv3d = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation7)

activation8 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3d)

conv3_up = layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(2,2),
                        padding='same', data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False,kernel_initializer=initializer,
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation8)
          
activation9 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3_up)

conv4a = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last', dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation9)

activation10 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4a)

conv4b = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last', dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation10)

activation11 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4b)

conv4c = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation11)

activation12 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4c)

conv4d = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation12)

activation13 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4d)

conv4_up = layers.Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(2,2),
                        padding='same', data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False, kernel_initializer=initializer, 
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation13)

activation14 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4_up)

conv5a = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation14)

activation15 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv5a)

conv5d = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation15)

activation16 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv5d)

pred_cube = layers.Conv2D(filters=13,kernel_size=(3,3),strides=(1,1),
                        padding='same',data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False, kernel_initializer=initializer, 
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation16)

outputs = layers.Activation(trainable=True, dtype='float32', activation='relu')(pred_cube)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    # Define parameters
    def __init__(self, list_IDs, minibatch_mult=4, batch_size=32, dim=(260,344), joints=13):
        'Initialization'
        self.dim = dim
        self.minibatch_mult = minibatch_mult
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.joints = joints
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.minibatch_mult))

    def __getitem__(self, index):
   
        # Generate indexes for a batch of data
        indexes = self.indexes[index*self.minibatch_mult:(index+1)*self.minibatch_mult]

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
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim, self.joints))        
        count = 0

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          x_file = np.load('/' + str(ID) + '.npy')
          y_file = np.load('/' + str(ID) + '.npy')
          for frame in range(self.minibatch_mult):
            # Store sample
            x[count] = x_file[frame]            
            # Store label
            y[count] = y_file[frame]

            count += 1
        return x, y

# Datasets
num_of_files = # the actual value is printed at the end of file_generation_singleview.py script as tot frames = ...
num_of_val_files = val_num # the actual value is printed at the end of file_generation_singleview.py script as tot val frames = ...
list_IDs = random.sample(range(0, num_of_files), num_of_files)

train_set = list_IDs[0:-num_of_val_files]
validation_set = list_IDs[-num_of_val_files:]

# Generators
training_generator = DataGenerator(list_IDs = train_set)
validation_generator = DataGenerator(list_IDs = validation_set)

# Loss function 
def mse2D(y_true, y_pred):
  mean_over_ch = k.mean(k.square(y_pred - y_true), axis=-1)
  mean_over_w = k.mean(mean_over_ch, axis=-1)
  mean_over_h = k.mean(mean_over_w, axis=-1)
  return mean_over_h

model.compile(
    # Optimizer
    optimizer='RMSprop',
    # Loss function to minimize
    loss = mse2D
)

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    if epoch < 15:
      return 0.0001 
    else:
       return 0.00001

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 20,
                    callbacks=callback,
                    verbose=1)
                    
history.history
