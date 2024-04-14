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


st.title('MIT-BIH Arrhythmia Database Viewer')

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
    123, 124, 200, 201, 202, 203, 204, 205, 206, 207, 
    208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 
    221, 222, 223, 228, 230, 231, 232, 233, 234
]
# Create input sliders for start_sample and end_sample
start_sample = st.slider('Start Sample', min_value=0, max_value=650000, step=1)
end_sample = st.slider('End Sample', min_value=0, max_value=650000, step=1)

# Create a dropdown menu for selecting signal number
signal_number = st.selectbox('Select Signal Number', available_signals)

# Add a button to trigger loading and plotting of the signal
if st.button('Load and Plot Signal'):
    load_and_plot_signal(signal_number,start_sample, end_sample)

st.set_option('deprecation.showPyplotGlobalUse', False)


df_mitbih_train = pd.read_csv('mitbih_train.csv', header=None) # train dataframe
df_mitbih_test = pd.read_csv(f'mitbih_test.csv', header=None) # test dataframe
df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)

df_ptbdb_normal = pd.read_csv(f'ptbdb_normal.csv', header=None)
df_ptbdb_abnormal = pd.read_csv(f'ptbdb_abnormal.csv', header=None)
df_ptbdb = pd.concat([df_ptbdb_normal, df_ptbdb_abnormal], axis=0)

# ptbdb
M_ptbdb = df_ptbdb.values
X_ptbdb = M_ptbdb[:,:-1]
y_ptbdb = M_ptbdb[:,-1]

# mitbih
M_mitbih = df_mitbih.values
X_mitbih = M_mitbih[:,:-1]
y_mitbih = M_mitbih[:,-1]

# mitbih plot
classes={0:"Normal",
         1:"Artial Premature",
         2:"Premature ventricular contraction",
         3:"Fusion of ventricular and normal",
         4:"Fusion of paced and normal"}
figB = plt.figure(figsize=(15,4))
for i in range(0,5):
    plt.subplot(2,3,i + 1)
    all_samples_indexes = np.where(y_mitbih == i)[0]
    rand_samples_indexes = np.random.randint(0, len(all_samples_indexes), 3)
    rand_samples = X_mitbih[rand_samples_indexes]
    plt.plot(rand_samples.transpose())
    plt.title("Samples of class " + classes[i], loc='left', fontdict={'fontsize':8}, x=0.01, y=0.85)
    plt.savefig("Samples of class " + classes[i], dpi=300)   
st.pyplot(figB)


# ptbdb plot
classes={0:"Normal", 1:"Abnormal (MI)"}
figC= plt.figure(figsize=(10,2))
for i in range(0,2):
    plt.subplot(1,2,i + 1)
    all_samples_indexes = np.where(y_ptbdb == i)[0]
    rand_samples_indexes = np.random.randint(0, len(all_samples_indexes), 3)
    rand_samples = X_ptbdb[rand_samples_indexes]
    plt.plot(rand_samples.transpose())
    plt.title("Samples of class " + classes[i], loc="left", fontdict={'fontsize':8})
    plt.savefig("Samples of class " + classes[i], dpi=300)
st.pyplot(figC)

repartition = df_mitbih[187].astype(int).value_counts()

figD=plt.figure(figsize=(5,5))
circle=plt.Circle( (0,0), 0.8, color='white')
plt.pie(repartition, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(circle)
plt.savefig("Plot of Unbalanced classes piechart", dpi=300)
st.pyplot(figD)

#---------------------First layer of pre-processing---------------------------------------------
df_1 : pd.core.frame.DataFrame = df_mitbih_train[df_mitbih_train[187] == 1]
df_2 : pd.core.frame.DataFrame = df_mitbih_train[df_mitbih_train[187] == 2]
df_3 : pd.core.frame.DataFrame = df_mitbih_train[df_mitbih_train[187] == 3]
df_4 : pd.core.frame.DataFrame = df_mitbih_train[df_mitbih_train[187] == 4]
df_0 : pd.core.frame.DataFrame = (df_mitbih_train[df_mitbih_train[187] == 0]).sample(n=20000,random_state=42)

df_1_upsample = resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample = resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample = resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample = resample(df_4,replace=True,n_samples=20000,random_state=126)

train_df_mitbih = pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
equilibre = train_df_mitbih[187].value_counts()

FigEE=plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(equilibre, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig("Plot of Balanced classes piechart", dpi=300)
plt.tight_layout()
st.pyplot(FigEE)

target_train = train_df_mitbih[187] # get the labels of the train data
target_test = df_mitbih_test[187] # get the labels of the test data
y_train_mitbih : np.ndarray = to_categorical(target_train)
y_test_mitbih : np.ndarray = to_categorical(target_test)

#------------------Define model for ptbdb dataset-----------------------------
def network_ptbdb(X_train, y_train, X_test, y_test):
    im_shape=(X_train.shape[1],1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')

    conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)

    conv1_1=BatchNormalization()(conv1_1)

    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)

    conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1=BatchNormalization()(conv2_1)

    pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)

    conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)

    conv3_1=BatchNormalization()(conv3_1)

    pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)

    flatten=Flatten()(pool3)
    dense_end1 = Dense(64, activation='relu')(flatten)
    dense_end2 = Dense(32, activation='relu')(dense_end1)
    main_output = Dense(2, activation='softmax', name='main_output')(dense_end2)
    # The number is 2 beacuse there are only 2 possible outputs.

    model = Model(inputs= inputs_cnn, outputs=main_output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.75)
    adam = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics = ['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
             ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)]

    history=model.fit(X_train, y_train, epochs=40, callbacks=callbacks, batch_size=32, validation_data=(X_test,y_test))

    # model.load_weights('best_model.h5')

    return(model,history)

#------------------Detecting MI with ptbdb dataset only-----------------------------
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(X_ptbdb, y_ptbdb, test_size=0.20) # 80% training set and 20% test set

X_train_ptbdb = X_train_ptbdb.reshape(len(X_train_ptbdb), X_train_ptbdb.shape[1],1)
X_test_ptbdb = X_test_ptbdb.reshape(len(X_test_ptbdb), X_test_ptbdb.shape[1],1)

y_train_ptbdb : np.ndarray = to_categorical(y_train_ptbdb)
y_test_ptbdb : np.ndarray = to_categorical(y_test_ptbdb)

model_ptbdb, history = network_ptbdb(X_train_ptbdb, y_train_ptbdb, X_test_ptbdb, y_test_ptbdb)

#-----------------------------------Evaluation models-----------------------------
def evaluate_model(history, X_test, y_test, model):

        scores = model.evaluate((X_test),y_test, verbose=0)
        
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model - Accuracy')
        plt.legend(['Training', 'Validation'], loc='lower right')
        FigF=plt.show()
        st.pyplot(FigF)

        fig2, ax_loss = plt.subplots()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model- Loss')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        FigG=plt.show()
        st.pyplot(FigG)
        target_names=['0','1','2','3','4']

        y_true = []

        for element in y_test:
            y_true.append(np.argmax(element))

        prediction_proba = model.predict(X_test)
        prediction = np.argmax(prediction_proba,axis=1)

evaluate_model(history, X_test_ptbdb, y_test_ptbdb, model_ptbdb)

y_pred_ptbdb = model_ptbdb.predict(X_test_ptbdb, batch_size=1000)
unique, counts = np.unique(y_test_ptbdb, return_counts=True)
results = model_ptbdb.evaluate(X_test_ptbdb, y_test_ptbdb, batch_size=128)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

y_pred_ptbdb = model_ptbdb.predict(X_test_ptbdb, batch_size = 1000);
cnf_matrix = confusion_matrix(y_test_ptbdb.argmax(axis=1), y_pred_ptbdb.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
FigH=plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.savefig("Confusion-matrix-for-PTBDB-Dataset", dpi=300)
# plt.clf()  # Clear the current plot for the next iteration
plt.tight_layout()
st.pyplot(FigH)

X_train_mitbih = train_df_mitbih.iloc[:,:186].values # Remember the train set is modified, resample balanced dataset.
X_test_mitbih = df_mitbih_test.iloc[:,:186].values # The test data is untouched! Very Important!

X_train_mitbih, y_train_mitbih = shuffle(X_train_mitbih, y_train_mitbih, random_state=0)
X_test_mitbih, y_test_mitbih = shuffle(X_test_mitbih, y_test_mitbih, random_state=0)

X_train_mitbih = X_train_mitbih.reshape(len(X_train_mitbih), X_train_mitbih.shape[1],1)
X_test_mitbih = X_test_mitbih.reshape(len(X_test_mitbih), X_test_mitbih.shape[1],1)

def network_mitbih(X_train, y_train, X_test, y_test):

    n_obs, feature, depth = X_train.shape;
    inp = Input(shape=(feature, depth));
    C = Conv1D(filters=32, kernel_size=5, strides=1)(inp) # 0th Conv1D layer

    C11 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C)
    A11 = Activation("relu")(C11)
    C12 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A11)
    S11 = Add()([C12, C])
    A12 = Activation("relu")(S11)
    M11 = MaxPooling1D(pool_size=5, strides=2)(A12)


    C21 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M11)
    A21 = Activation("relu")(C21)
    C22 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A21)
    S21 = Add()([C22, M11])
    A22 = Activation("relu")(S21) # A possible mistake! Replace S11 with S21.
    M21 = MaxPooling1D(pool_size=5, strides=2)(A22)


    C31 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M21)
    A31 = Activation("relu")(C31)
    C32 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A31)
    S31 = Add()([C32, M21])
    A32 = Activation("relu")(S31)
    M31 = MaxPooling1D(pool_size=5, strides=2)(A32)


    C41 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M31)
    A41 = Activation("relu")(C41)
    C42 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A41)
    S41 = Add()([C42, M31])
    A42 = Activation("relu")(S41)
    M41 = MaxPooling1D(pool_size=5, strides=2)(A42)


    C51 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(M41)
    A51 = Activation("relu")(C51)
    C52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(A51)
    S51 = Add()([C52, M41])
    A52 = Activation("relu")(S51)
    M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

    F1 = Flatten()(M51)

    D1 = Dense(32)(F1)
    A6 = Activation("relu")(D1)
    D2 = Dense(32)(A6)
    D3 = Dense(5)(D2)
    A7 = Softmax()(D3)

    model = Model(inputs=inp, outputs=A7)

    print(model.summary())

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.75)
    adam = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S");
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1, profile_batch = '500,520');

    callbacks = [EarlyStopping(monitor='val_loss', patience=8), ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True), tensorboard_callback]

    history = model.fit(X_train, y_train,
                    epochs=75,
                    batch_size=256, # 500
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)

    return(model, history);
model_mitbih, history_mitbih = network_mitbih(X_train_mitbih, y_train_mitbih, X_test_mitbih, y_test_mitbih)

evaluate_model(history_mitbih, X_test_mitbih, y_test_mitbih, model_mitbih)

y_pred_mitbih = model_mitbih.predict(X_test_mitbih, batch_size=1000)
results = model_mitbih.evaluate(X_test_mitbih, y_test_mitbih, batch_size=128)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

y_pred_mitbih = model_mitbih.predict(X_test_mitbih, batch_size = 1000);
cnf_matrix = confusion_matrix(y_test_mitbih.argmax(axis=1), y_pred_mitbih.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
figI= plt.figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix, classes=['N', 'S', 'V', 'F', 'Q'],normalize=True,
                      title='Confusion matrix, with normalization')
plt.savefig("Confusion-matrix-for-MITBIH-Dataset", dpi=300)
# plt.clf()  # Clear the current plot for the next iteration
plt.tight_layout()
st.pyplot(figI)
