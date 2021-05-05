from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os

def Conv1D(NUMBER_CLASSES=2, SAMPLE_RATE=16000, DELTA_TIME=1.0):
    input_shape = (int(SAMPLE_RATE*DELTA_TIME),1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128, 
                                 pad_end=True,
                                 n_fft=512, 
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SAMPLE_RATE, 
                                 return_decibel = True,
                                 input_data_format='channels_last', 
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'))(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'))(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'))(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'))(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'))(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001))(x)
    o = layers.Dense(NUMBER_CLASSES, activation='softmax')(x) # Use sigmoid when theres multiple possible classification outputs. Softmax better for binary classifier :NOTE I HAVE CHANGED THIS TO SOFTMAX AFTER TRAINING. IF BUS OUT CHANGE TO SIGMOID
    model = Model(inputs=i.input, outputs=o)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def Conv2D(NUMBER_CLASSES=2, SAMPLE_RATE=16000, DELTA_TIME=1.0):
    input_shape = (int(SAMPLE_RATE*DELTA_TIME), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SAMPLE_RATE,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2)(i.output)
    x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)    
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001))(x)
    o = layers.Dense(NUMBER_CLASSES, activation='softmax') (x)
    model = Model(inputs=i.input, outputs=o)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def LSTM(NUMBER_CLASSES=2, SAMPLE_RATE=16000, DELTA_TIME=1.0):
    input_shape=(int(SAMPLE_RATE * DELTA_TIME), 1)
    i = get_melspectrogram_layer(input_shape=input_shape, 
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SAMPLE_RATE,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last',)
    x = LayerNormalization(axis=2)(i.output)
    
    x = TimeDistributed(layers.Reshape((-1,)))(x) # reshape used to remove channels dimension and prepare it for use in the LSTM
    s = TimeDistributed(layers.Dense(64, activation='tanh'))(x) # Learn the most relevent features before endering the LSTM - This has been shown to improve LSTM performance.
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(s) # Bidirectional looks forward and backwards in time, which results in better gradient descent updates
    x = layers.concatenate([s,x], axis=2) # combine the feautres learnt before the LSTM and the resulting ones - common in audio networks
    
    x = layers.Dense(64, activation='relu')(x) # dense and maxpooling to prevent over fitting. 1d as there is no channel information
    x = layers.MaxPooling1D()(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x) # flatten it
    
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(32, activation='relu', activity_regularizer=l2(0.001))(x)
    o = layers.Dense(NUMBER_CLASSES, activation='softmax')(x) # fit a classifier
    model = Model(inputs=i.input, outputs=o)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model