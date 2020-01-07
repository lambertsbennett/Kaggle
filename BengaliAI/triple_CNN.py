import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import layers, models
from tensorflow import keras
import glob

#consonant diacritic model
num_outputs = 7
inputs = keras.Input(shape=(137, 236, 1), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_outputs, activation='softmax')(x)

model_consdia = keras.Model(inputs, outputs, name='bengali_consdia')
model_consdia.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=8)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.2)

train_labels = pd.read_csv("Data/train.csv")
def batch_train_preprocess_consdia():
    for filepath in glob.iglob(r'Data/train*.parquet'):
        train = pd.read_parquet(filepath)
        train_image_id = train.image_id
        train.drop(columns=['image_id'],inplace=True)

        train_image_id = pd.DataFrame(train_image_id, columns=['image_id'])
        labels = pd.merge(train_image_id,train_labels, on='image_id')
        y = labels['consonant_diacritic']
        train_data = train.values.reshape(train.shape[0],137,236,1)

        model_consdia.fit(train_data,np.array(y),
              epochs=80,validation_split=0.2,batch_size=32, shuffle=True,callbacks =[lr_reduction,es])

batch_train_preprocess_consdia()
model_consdia.save("consdia_model.h5")
print("Consonant diacritic model to disk")


num_outputs = 11

inputs = keras.Input(shape=(137, 236, 1), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_outputs, activation='softmax')(x)

model_vdia = keras.Model(inputs, outputs, name='bengali_vowel_dia')

model_vdia.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=8)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.2)

def batch_train_preprocess_vowdia():
    for filepath in glob.iglob(r'Data/train*.parquet'):
        train = pd.read_parquet(filepath)
        train_image_id = train.image_id
        train.drop(columns=['image_id'],inplace=True)

        train_image_id = pd.DataFrame(train_image_id, columns=['image_id'])
        labels = pd.merge(train_image_id,train_labels, on='image_id')
        y = labels['vowel_diacritic']
        train_data = train.values.reshape(train.shape[0],137,236,1)

        model_vdia.fit(train_data,np.array(y),
              epochs=80,validation_split=0.2,batch_size=32, shuffle=True,callbacks =[lr_reduction,es])

batch_train_preprocess_vowdia()
model_vdia.save("vdia_model.h5")
print("Vowel diacritic model to disk")

num_outputs = 168

inputs = keras.Input(shape=(137, 236, 1), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_outputs, activation='softmax')(x)

model_grapheme = keras.Model(inputs, outputs, name='bengali_grapheme_root')

model_grapheme.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=8)
lr_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.2)

def batch_train_preprocess_grapheme():
    for filepath in glob.iglob(r'Data/train*.parquet'):
        train = pd.read_parquet(filepath)
        train_image_id = train.image_id
        train.drop(columns=['image_id'],inplace=True)

        train_image_id = pd.DataFrame(train_image_id, columns=['image_id'])
        labels = pd.merge(train_image_id,train_labels, on='image_id')
        y = labels['grapheme_root']
        train_data = train.values.reshape(train.shape[0],137,236,1)

        model_grapheme.fit(train_data,np.array(y),
              epochs=80,validation_split=0.2,batch_size=32, shuffle=True,callbacks =[lr_reduction,es])

batch_train_preprocess_grapheme()
model_grapheme.save("grapheme_model.h5")
print("Grapheme model saved to disk")


grapheme = []
con_dia = []
vow_dia = []
label = []

for filepath in glob.iglob(r'Data/test*.parquet'):
    test = pd.read_parquet(filepath)
    test_image_id = test.image_id
    test.drop(columns=['image_id'],inplace=True)
    test_data = test.values.reshape(test.shape[0],137,236,1)
    label.append(test_image_id)

    con_dia.append(np.argmax(model_consdia.predict(test_data),axis=1))
    vow_dia.append(np.argmax(model_vdia.predict(test_data),axis=1))
    grapheme.append(np.argmax(model_grapheme.predict(test_data),axis=1))

res = pd.DataFrame({'row_id':label,'consonant_diacritic':con_dia,'grapheme_root':grapheme,'vowel_diacritic':vow_dia})
res.to_csv('results.csv',index=False)
