from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import kapre
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf
import os


def Conv1D(NUMBER_CLASSES=2, SAMPLE_RATE=22050, DELTA_TIME=1.0):
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
    o = layers.Dense(NUMBER_CLASSES, activation='softmax')(x)
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
    o = layers.Dense(2, activation='softmax') (x)
    model = Model(inputs=i.input, outputs=o)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model