import streamlit as st
from tensorflow.keras.models import load_model
from CleanAndSplit import downsample, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm

def make_predictions(args, file_name):
    model = load_model(args.model_fn, custom_objects={
        'STFT':STFT,
        'Magnitude': Magnitude,
        'ApplyFilterbank': ApplyFilterbank,
        'MagnitudeToDecibel': MagnitudeToDecibel
    })
    classes = ["emergency", "nonEmergency"]
    le = LabelEncoder()
    y_true = le.fit_transform(classes)
    results = []

    rate, wav = downsample(file_name, args.sample_rate)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    clean_wav = wav[mask]
    step = int(args.sample_rate*args.delta_time)
    batch = []

    for i in range (0, clean_wav.shape[0],step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(-1,1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0]]= sample.flatten().reshape(-1,1)
            sample=tmp
        batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    return classes[y_pred]


def predict(file_name, option):

    # model_string = 'models/' + option + ".h5"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fn', type=str, default='models/{}.h5'.format(option))
    parser.add_argument('--delta_time', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--threshold', type=str, default=20)
    parser.add_argument('--src_dir', type=str, default='Data')

    args, _ = parser.parse_known_args()

    prediciton =  make_predictions(args, file_name)
    
    return prediciton


st.title('Siren Classifier')

file_uploader = st.file_uploader("Upload a wav file", type=([".wav"]))

option = st.selectbox('Choose DL model:', ('Conv1d', 'Conv2d', 'LSTM'))



if(st.button("Predict")): 
    print_prediciton = predict(file_uploader, option)
    st.title(print_prediciton)