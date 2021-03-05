import streamlit as st
# import librosa
# import numpy as np
# import pickle

# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2
# import kapre
# from kapre.composed import get_melspectrogram_layer
# import tensorflow as tf
# import os


from tensorflow.keras.models import load_model
from cleanTheData import downsample_mono, envelope
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
    labels = ["emergency", "nonEmergency"]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    rate, wav = downsample_mono(file_name, args.sample_rate)
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
    # print(classes[y_pred])
    return classes[y_pred]


def predict(file_name):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fn', type=str, default='models/Conv1d.h5')
    parser.add_argument('--delta_time', type=float, default=1.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--threshold', type=str, default=20)
    parser.add_argument('--src_dir', type=str, default='Data')

    args, _ = parser.parse_known_args()

    prediciton =  make_predictions(args, file_name)
    
    return prediciton
    # print(file_name)

# TODO: Integrate the below code, this code works with the model from MLP. The above works from the Conv1D

# def extract_feature(file_name):
#     try:
#         audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
#         mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
#         mfccsscaled = np.mean(mfccs.T,axis=0)
        
#     except Exception as e:
#         print("Error encountered while parsing file: ")
#         return None

#     return np.array([mfccsscaled])

# def print_prediction(file_name):

#     model = Conv1D()
#     model.load_weights('Conv1d.h5')
#     le = LabelEncoder()
#     le.fit(["emergency", "nonEmergency"])
#     prediction_feature = extract_feature(file_name) 

#     predicted_vector = model.predict(prediction_feature)
#     predicted_class = le.inverse_transform(predicted_vector) 
#     if(predicted_class[0] == "emergency"):
#         return "This is a siren"
#     else:
#         return "This is not a siren"


st.title('Siren Classifier')

file_uploader = st.file_uploader("Upload a wav file", type=([".wav"]))



if(st.button("Predict")): 
    # print_prediciton = print_prediction(file_uploader)
    print_prediciton = predict(file_uploader)
    # print(print_prediciton)
    st.title(print_prediciton)