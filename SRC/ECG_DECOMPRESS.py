import warnings
warnings.simplefilter(action='ignore' , category=FutureWarning)

import streamlit as st
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import math
import datetime
import random


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import resample, shuffle # Used for resampling data to get an equal representation of al the classes, see step_1()
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss, coverage_error, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve

# keras imports
import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Softmax, Add, Flatten, Activation, Convolution1D, MaxPool1D
from keras.models import Model
from keras import backend as K
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical

# Used in the base version of CNN, not based on the arxiv model
from tensorflow.keras.layers import BatchNormalization

import warnings
warnings.filterwarnings('ignore')


st.title('ECG Viewer')

# Define a function to load and plot the signal
def load_and_plot_signal(signal_number, start_sample, end_sample):
    filename = str(signal_number)
    record = wfdb.rdrecord(filename, sampfrom=start_sample, sampto=end_sample)
    annotation = wfdb.rdann(filename, 'atr', sampfrom=start_sample, sampto=end_sample, shift_samps=True)

    # Plot the signal
    plt.figure(figsize=(15, 8))
    plt.plot(record.p_signal)
    plt.title('ECG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Plot annotations
    plt.figure(figsize=(15, 8))
    figA = wfdb.plot_wfdb(record=record, annotation=annotation, time_units='seconds')
    st.pyplot(figA)
    

# Create a slider for selecting signal number
available_signals = [
    100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 
    112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 
    123, 124, 200, 201, 202, 203, 204, 205, 207, 
    208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 
    221, 222, 223, 228, 230, 231, 232, 233, 234
]
# Create input sliders for start_sample and end_sample
start_sample = st.slider('Start Sample', min_value=0, max_value=650000, step=1)
end_sample = st.slider('End Sample', min_value=0, max_value=650000, step=1)

# Create a dropdown menu for selecting signal number
signal_number = st.selectbox('Select Signal Number', available_signals)

import csv
def load_data():
    with open("Book1.csv", "r") as file:
        reader = csv.reader(file)
        data = {int(row[0]): int(row[1]) for row in reader}
    return data
data = load_data()
selected_signal_number = signal_number
if selected_signal_number in data:
    st.markdown(f"<h2>Peak detected: {data[selected_signal_number]}</h2>", unsafe_allow_html=True)

# Add a button to trigger loading and plotting of the signal
if st.button('Load and Plot Signal'):
    load_and_plot_signal(signal_number,start_sample, end_sample)

st.set_option('deprecation.showPyplotGlobalUse', False)



