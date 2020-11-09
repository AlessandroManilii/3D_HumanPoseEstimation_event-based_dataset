import random
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow.keras.backend as k 
import tensorflow.keras as keras
import time

start_time = time.time()


initializer = initializers.VarianceScaling(scale=1,mode='fan_avg',distribution='uniform',seed=None)
bias_init = initializers.zeros()

input_2 = keras.Input(shape=(260,344,1), dtype='float32', sparse=False)
input_3 = keras.Input(shape=(260,344,1), dtype='float32', sparse=False)



conv1_2 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                      padding='same',data_format='channels_last',dilation_rate=(1,1),
                      use_bias=False, kernel_initializer=initializer, bias_initializer=bias_init,
                      kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None, 
                      activation = 'linear',kernel_constraint = None, bias_constraint=None, name = 'conv1_2')(input_2)
                      
conv1_3 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                      padding='same',data_format='channels_last',dilation_rate=(1,1),
                      use_bias=False, kernel_initializer=initializer, bias_initializer=bias_init,
                      kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None, 
                      activation = 'linear',kernel_constraint = None, bias_constraint=None, name='conv1_3')(input_3)





activation1_2 = layers.Activation(trainable=True, dtype='float32', activation='relu', name='activation1_1_2')(conv1_2)
activation1_3 = layers.Activation(trainable=True, dtype='float32', activation='relu', name='activation1_1_3')(conv1_3)


pool1_2 = layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), 
                       data_format='channels_last')(activation1_2)

pool1_3 = layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2), 
                       data_format='channels_last')(activation1_3)


conv2a_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,
                       bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,
                       bias_constraint=None, name='conv2a_2')(pool1_2)
                       
conv2a_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,
                       bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,
                       bias_constraint=None,name='conv2a_3')(pool1_3)


activation2_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2a_2)
activation2_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2a_3)

conv2b_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv2b_2')(activation2_2)
conv2b_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv2b_3')(activation2_3)


activation3_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2b_2)
activation3_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2b_3)


conv2d_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv2d_2')((activation3_2))

conv2d_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv2d_3')(activation3_3)


activation4_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2d_2)
activation4_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv2d_3)



pool2_2 = layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2),
                       data_format='channels_last')(activation4_2)

pool2_3 = layers.MaxPooling2D(pool_size=(2,2), padding='valid', strides=(2,2),
                       data_format='channels_last')(activation4_3)

merge = layers.concatenate([pool2_2, pool2_3])

conv3a = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',data_format='channels_last',
                       dilation_rate=(2,2),activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(merge)



activation5 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3a)

                      
conv3b = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv3b')(activation5)



activation6 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3b)


conv3c = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',data_format='channels_last',
                       dilation_rate=(2,2),activation='linear',use_bias=False, kernel_initializer=initializer, bias_initializer=bias_init,
                       kernel_regularizer=None, bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None)(activation6)



activation7 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3c)


conv3d = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv3d')(activation7)


activation8 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3d)


conv3_up = layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(2,2),
                        padding='same', data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False,kernel_initializer=initializer,
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv3_up_2')(activation8)

                    

          
activation9_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3_up)
activation9_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv3_up)

conv4a_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last', dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4a_2')(activation9_2)

conv4a_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last', dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4a_3')(activation9_3)


activation10_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4a_2)
activation10_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4a_3)


conv4b_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last', dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4b_2')(activation10_2)
conv4b_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last', dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4b_3')(activation10_3)


activation11_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4b_2)
activation11_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4b_3)

conv4c_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4c_2')(activation11_2)

conv4c_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4c_3')(activation11_3)


activation12_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4c_2)
activation12_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4c_3)


conv4d_2 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4d_2')(activation12_2)
conv4d_3 = layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(2,2),
                       activation='linear',use_bias=False, kernel_initializer=initializer, 
                       bias_initializer=bias_init,kernel_regularizer=None, bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4d_3')(activation12_3)                       


activation13_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4d_2)
activation13_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4d_3)

conv4_up_2 = layers.Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(2,2),
                        padding='same', data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False, kernel_initializer=initializer, 
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4_up_2')(activation13_2)

conv4_up_3 = layers.Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(2,2),
                        padding='same', data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False, kernel_initializer=initializer, 
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv4_up_3')(activation13_3)                       



activation14_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4_up_2)
activation14_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv4_up_3)



conv5a_2 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv5a_2')(activation14_2)
conv5a_3 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                      padding='same',data_format='channels_last',dilation_rate=(1,1),
                      activation='linear',use_bias=False, kernel_initializer=initializer,
                      bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                      activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv5a_3')(activation14_3)                      



activation15_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv5a_2)
activation15_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv5a_3)


conv5d_2 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv5d_2')(activation15_2)
conv5d_3 = layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),
                       padding='same',data_format='channels_last',dilation_rate=(1,1),
                       activation='linear',use_bias=False, kernel_initializer=initializer,
                       bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                       activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='conv5d_3')(activation15_3)                      


activation16_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv5d_2)
activation16_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(conv5d_3)



pred_cube_2 = layers.Conv2D(filters=13,kernel_size=(3,3),strides=(1,1),
                        padding='same',data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False, kernel_initializer=initializer, 
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='pred_cube_2')(activation16_2)

pred_cube_3 = layers.Conv2D(filters=13,kernel_size=(3,3),strides=(1,1),
                        padding='same',data_format='channels_last',dilation_rate=(1,1),
                        activation='linear',use_bias=False, kernel_initializer=initializer, 
                        bias_initializer=bias_init,kernel_regularizer=None,bias_regularizer=None,
                        activity_regularizer=None,kernel_constraint=None,bias_constraint=None, name='pred_cube_3')(activation16_3)

activation17_2 = layers.Activation(trainable=True, dtype='float32', activation='relu')(pred_cube_2)
activation17_3 = layers.Activation(trainable=True, dtype='float32', activation='relu')(pred_cube_3)





model = keras.Model(
    inputs=[input_2, input_3],
    outputs=[activation17_2, activation17_3],
)

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
    loss = mse2D)

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    if epoch < 15:
      return 0.0001 
    else:
       return 0.00001

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):        
      self.model.save("/model_{}.h5".format(epoch))

callback = [tf.keras.callbacks.LearningRateScheduler(scheduler), CustomSaver()]

# Train model on dataset
history = model.fit(training_generator,
                    validation_data=validation_generator,
                    epochs=20,
                    callbacks=callback,
                    verbose=1)

history.history

print("--- %s seconds ---" % (time.time() - start_time))

model.save('') 
