import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D, Conv2D
from tqdm import tqdm
from glob import glob
import argparse
import warnings


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sample_rate, delta_time, number_classes, batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.delta_time = delta_time
        self.number_classes = number_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()
        
    def __len__ (self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for  k in indexes]
        
        X = np.empty((self.batch_size,int(self.sample_rate*self.delta_time),1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.number_classes),  dtype =np.float32)
        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.number_classes)
            
        return X, Y


    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    src_root = args.src_root
    sample_rate = args.sample_rate
    delta_time = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params  = {'NUMBER_CLASSES':2,  # This caused an issue when i just put the number 2
               'SAMPLE_RATE': sample_rate,
               'DELTA_TIME': delta_time}
    models = {'Conv1d': Conv1D(**params),
              'Conv2d': Conv2D(**params)  }
    print (os.listdir(args.src_root))
    wav_paths = glob('{}/**' .format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep,'/') for x in wav_paths if '.wav' in x] # This prevents the .DS store error as it only includes .wav files
    classes=sorted(os.listdir(args.src_root))
    le=LabelEncoder()
    le.fit(classes)

    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths, labels,test_size=0.1, random_state=0)

    tg = DataGenerator(wav_train, label_train, sample_rate, delta_time, params['NUMBER_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sample_rate, delta_time, params['NUMBER_CLASSES'], batch_size=batch_size)
    model=models[model_type]
    checkpoint = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', verbose=1)
    
    model.fit(tg,validation_data=vg,epochs=10, verbose=1, callbacks=[checkpoint])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio classifier training')
    parser.add_argument('--model_type', type=str, default='Conv2d')
    parser.add_argument('--src_root', type=str, default='clean')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--delta_time', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    args, _ = parser.parse_known_args()

    train(args)