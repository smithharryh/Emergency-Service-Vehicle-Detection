"""
CleanAndSplit.py

- Downsample the audio and remove any deadspace in the audio with threshold detection

"""

import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import wavio


"""
DOWNSAMPLE FUNCTION
- Reduce the sample rate to 16000 on files with higher frequency to remove the extreme frequencies which may throw off the model
- Also this lowers computational requirements and unifies across files. Some of my audio data is 44100 sample rate so reduce and even them.

"""

def downsample(path,sample_rate):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate=obj.rate
    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as e:
        print(path)
        print(e)
        pass

    wav = resample(wav, rate, sample_rate) 
    wav = wav.astype(np.int16)
    return sample_rate, wav

"""
ENVELOPE FUNCTION
- Envelope in waves is a smooth curve outlining the extremes of an oscilating signal
The signal envelope tracks how the signal changes over time. 
- PD series is a 1D array with axis labels. np.abs returns the absolute value for each val in arr. (so -1.1 becomes 1.1)
- Think of roliing window sort of like a bubble sort, as a method to find the mean it finds the mean of y.
- Variables:
    y: the signal
    rate:
    threshold: used to reduce deadspace in the audio. If it goes super quiet with no sound at the end don't include it.
"""

def envelope(y, rate, threshold): 
    mask = []
    y = pd.Series(y).apply(np.abs) 
    y_mean = y.rolling(window=int(rate/20),min_periods=1, center=True).max()
    for mean in y_mean:
        if mean >threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


"""
SAVE_SAMPLE FUNCTION


"""
def save_sample(sample, rate, target_dir, fn, index):
    fn = fn.split('.wav')[0]
    destination_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(index)))
    if os.path.exists(destination_path):
        return
    wavfile.write(destination_path, rate, sample)

"""
CHECK_DIR FUNCTION
- Checks a directory exists at the specified path. If, not create a new directory

"""

def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


"""
CALCULATE_THRESHOLD FUNCTION


"""
def calculate_threshold(args):
    data_dir = args.data_dir
    wav_paths = glob('{}/**'.format(data_dir), recursive=True)
    wav_path = [x for x in wav_paths if args.wav_filename in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.wav_filename))
        return
    rate, wav = downsample(wav_path[0], args.sample_rate)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()

"""
SPLIT_WAVS FUNCTION


"""
def split_wavs(args):
    data_dir = args.data_dir
    dst_root = args.dst_root
    delta_time = args.delta_time

    wav_paths = glob('{}/**'.format(data_dir), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(data_dir)
    check_dir(dst_root)
    classes = os.listdir(data_dir)

    for _class in classes:
        target_dir = os.path.join(dst_root, _class)
        check_dir(target_dir)
        if(_class != '.DS_Store'): # Ignores the DS_store file when looping through the list of directories
            src_dir = os.path.join(data_dir, _class)
            for fn in tqdm(os.listdir(src_dir)):
                if(fn != '.DS_Store'): # Ignores the DS_store file in each class directory
                    src_fn = os.path.join(src_dir, fn)
                    print(src_fn)
                    rate, wav = downsample(src_fn, args.sample_rate) #returns the resampledr  rate and the resampled wav as a npint16 array. 
                    mask, y_mean = envelope(wav, rate, threshold=args.threshold) #return the mask to reduce noise  and deadspace
                    wav = wav[mask] # apply the mask 
                    delta_sample = int(delta_time*rate) # sample rate times one second

                    if wav.shape[0] < delta_sample: # essentially, if the wav is less than one second then pad it with zeros.
                        sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                        sample[:wav.shape[0]] = wav
                        save_sample(sample,rate,target_dir,fn,0)
                    else: #  Otherwise go through the wav and split it into  one second intervals and save it. 
                        trunc = wav.shape[0] % delta_sample
                        for count, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                            start = int(i)
                            stop = int(i +delta_sample)
                            sample = wav[start:stop]
                            save_sample(sample, rate, target_dir, fn, count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean audio data')
    parser.add_argument('--data_dir', type=str, default='Data'), # Directory input data is being held in
    parser.add_argument('--dst_root', type=str, default='clean'), # Directory to output cleaned data split by delta time
    parser.add_argument('--delta_time', type=float, default=1.0) # Delta time refers to the time between samples. Here it is set as one second.
    parser.add_argument('--sample_rate', type=int, default=16000) # Desired sample rate used to down sample the audio.
    parser.add_argument('--wav_filename', type=str, default='1078') # Name of the wav file for testing the threshold against
    parser.add_argument('--threshold', type=str,default=100) # Threshold value for use in the envelope function.
    args, _ = parser.parse_known_args()
    split_wavs(args)
