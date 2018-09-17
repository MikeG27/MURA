# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

# =============================================================================
#                                 Libraries
# =============================================================================
import os
import shutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



from modules import models, my_utils, plots
from keras.applications import VGG16

from keras.models import Sequential
from keras.models import load_model
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization


# =============================================================================
#                           Create folder structure
# =============================================================================

os.chdir("/home/michal/Pulpit/MURA/MURA-v1.1") # Go to data directory
main_dir = os.getcwd()
main_structure = my_utils.get_main_f(os.listdir())


hand_path = os.path.join(main_dir,"hand")

os.mkdir(hand_path)

# Train
hand_train = os.path.join(hand_path,"train")
os.mkdir(hand_train)

hand_train_normal = os.path.join(hand_train,"normal")
os.mkdir(hand_train_normal)

hand_train_abnormal = os.path.join(hand_train,"abnormal")
os.mkdir(hand_train_abnormal)

# Validation
hand_valid = os.path.join(hand_path,"valid")
os.mkdir(hand_valid)
    
hand_valid_normal = os.path.join(hand_valid,"normal")
os.mkdir(hand_valid_normal)
    
hand_valid_abnormal = os.path.join(hand_valid,"abnormal")
os.mkdir(hand_valid_abnormal)


# =============================================================================
#                               Preprocessing
# =============================================================================

#CSV
os.chdir(main_structure[4])
csv_structure = my_utils.read_csv(os.listdir())
path_train,path_valid,label_valid,label_train = my_utils.read_csv(os.listdir())
os.chdir(main_dir)

# GET HAND XRAY

#Change columns name
path_train = path_train.rename(columns = {'MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png':'img'})
path_valid = path_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png':'img'})

# Use function and get only HAND directories
path_train = my_utils.get_selected_feature(path_train,"MURA-v1.1/train/XR_HAND")
path_valid = my_utils.get_selected_feature(path_valid,"MURA-v1.1/valid/XR_HAND")

# Get radiologist predictions
label_train = label_train.rename(columns = {"MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/":'img'})
label_valid = label_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/':'img'})
label_train = my_utils.get_selected_feature(label_train,"XR_HAND")
label_valid = my_utils.get_selected_feature(label_valid,"XR_HAND")

# Get normal/abnormal distribution
distr_hand_train = my_utils.get_data_distribution(path_train)
distr_hand_valid = my_utils.get_data_distribution(path_valid)


# Data separation

# 1.Rename pathes
path_train = my_utils.re_path(path_train)
path_valid = my_utils.re_path(path_valid)

# 2. Separate images to folder classes

my_utils.separate_images(path_train,hand_train_abnormal,hand_train_normal)
my_utils.separate_images(path_valid,hand_valid_abnormal,hand_valid_normal)


# =============================================================================
#                               Data Overview
# =============================================================================

#img_array = image.img_to_array(img)


print("\n1. Hand images : ")

print("\nTrain images : ",len(path_train))
print("Valid images : ",len(path_valid))

print("\n2. Hand images overview : \n")

print("Train abnormal img : ",len(os.listdir(hand_train_abnormal)))
print("Train normal img : ",len(os.listdir(hand_train_normal)))
print("Valid abnormal img : ",len(os.listdir(hand_valid_abnormal)))
print("Valid normal img : ",len(os.listdir(hand_valid_normal)))

#print("\n3. Image stats:\n")
#print("Image height : ",img_array.shape[0])
#print("Image width : ",img_array.shape[1])
#print("Image depth : ",img_array.shape[2])

#del img , img1 , img2 , img_path1 , img_path2 , img_path

#plot_sample_img()


# =============================================================================
#                                 Load Model 
# =============================================================================

# =============================================================================
# 
# =============================================================================

# Test Accuracy 0.7
# Test loss 0.78

from keras.applications import VGG16
from keras import optimizers

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        hand_train,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        hand_valid,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)


# Fine tuning 


conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history1 = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=125,
      validation_data=validation_generator,
      validation_steps=50)




# =============================================================================
#                         test model
# =============================================================================

test_generator = test_datagen.flow_from_directory(
        hand_valid,
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

print("Results")
print('test acc:', test_acc)
print('test loss:', test_loss)

plots.plot_training(history1,"/home/michal/Pulpit/MURA/MURA-v1.1","VGG16_fine_tuning")

# =============================================================================
#                           save model
# =============================================================================

