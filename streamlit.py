import streamlit as st
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


from tensorflow import keras
model = keras.models.load_model('my_model')

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ")
        return None

    return np.array([mfccsscaled])

def print_prediction(file_name):
    le = LabelEncoder()
    le.fit(["emergency", "nonEmergency"])
    prediction_feature = extract_feature(file_name) 

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    if(predicted_class[0] == "emergency"):
        return "This is a siren"
    else:
        return "This is not a siren"


st.title('Siren Classifier')

file_uploader = st.file_uploader("Upload a wav file", type=([".wav"]))


if(st.button("Predict")): 
    print_prediciton = print_prediction(file_uploader)
    st.title(print_prediciton)