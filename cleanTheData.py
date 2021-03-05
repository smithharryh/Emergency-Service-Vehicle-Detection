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

def downsample_mono(path,sample_rate):
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

def save_sample(sample, rate, target_dir, fn, index):
    fn = fn.split('.wav')[0]
    destination_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(index)))
    if os.path.exists(destination_path):
        return
    wavfile.write(destination_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    delta_time = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)

    for _class in classes:
        target_dir = os.path.join(dst_root, _class)
        check_dir(target_dir)
        if(_class != '.DS_Store'):
            src_dir = os.path.join(src_root, _class)
            for fn in tqdm(os.listdir(src_dir)):
                if(fn != '.DS_Store'):
                    src_fn = os.path.join(src_dir, fn)
                    print(src_fn)
                    rate, wav = downsample_mono(src_fn, args.sample_rate)
                    mask, y_mean = envelope(wav, rate, threshold=args.threshold)
                    wav = wav[mask]
                    delta_sample = int(delta_time*rate)

                    if wav.shape[0] < delta_sample:
                        sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                        sample[:wav.shape[0]] = wav
                        save_sample(sample,rate,target_dir,fn,0)
                    else:
                        trunc = wav.shape[0] % delta_sample
                        for count, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                            start = int(i)
                            stop = int(i +delta_sample)
                            sample = wav[start:stop]
                            save_sample(sample, rate, target_dir, fn, count)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    rate, wav = downsample_mono(wav_path[0], args.sample_rate)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean audio data')
    parser.add_argument('--src_root', type=str, default='Data'),
    parser.add_argument('--dst_root', type=str, default='clean'),
    parser.add_argument('--delta_time', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--fn', type=str, default='1078') #TODO:check this - this is the name of the wavfile you want to use for threshold detection. for me it'll be 1078_1.wav for example
    parser.add_argument('--threshold', type=str,default=20) #  TODO: check this
    args, _ = parser.parse_known_args()

    split_wavs(args)
