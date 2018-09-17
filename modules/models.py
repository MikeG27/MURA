#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:41:11 2018

@author: michal
"""


from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras.applications import VGG16


# =============================================================================
#                                   VGG16
# =============================================================================

def get_VGG16():
    
    model = Sequential()
    
    model.add(Conv2D(64,(3,3),activation="relu",padding="same",input_shape=(150,150,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
        
    model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D((2,2)))
        
        
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D((2,2)))
        
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D((2,2)))
        
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Conv2D(512,(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
        
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# =============================================================================
#                                 VGG16 pretrained
# =============================================================================

def get_pre_VGG16():
    
    conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
    conv_base.trainable = True

    set_trainable = False
    
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = Sequential()
    
    
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# =============================================================================
#                              Resnet50 pretrained
# =============================================================================
