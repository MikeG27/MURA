# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
#                                 Libraries
# =============================================================================
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
#                              Auxiliary functions
# =============================================================================

def get_main_f(folder_list):
    
    """
    Get list of main folders of main directory to variable
    
    Input : list of directories
    
    Output : List of directories in variable
    
    """
    
    f_list = []
    
    for i in folder_list:
        f_list.append(os.path.join("/home/michal/Pulpit/MURA/MURA-v1.1",i))
        print(f_list)
        
    return f_list


def read_csv(csv_list):
    
    
    """
    Get variables contains csvs in Dataframe
    
    Input : list of csv
    
    Output : csv in variable
    
    """
    
    # Generator
    results = []
    i = 0
    while i < len(csv_list):
        results.append(pd.read_csv(csv_list[i]))
        i += 1
    return results



def get_selected_feature(img_path_array,feature_path):
    
    """
    Get hand data 
    
    """
    
    data = []
    
    for i in img_path_array["img"]:
        if feature_path in i:
            data.append(i)
            
    return data


# =============================================================================
#                           Create folder structure
# =============================================================================

os.chdir("/home/michal/Pulpit/MURA/MURA-v1.1") # Go to data directory
main_dir = os.getcwd()
main_structure = get_main_f(os.listdir())


hand_path = os.path.join(main_dir,"hand")

if os.path.exists(hand_path) == False:
   
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
os.chdir(main_structure[1])
csv_structure = read_csv(os.listdir())
path_train,path_valid,label_valid,label_train = read_csv(os.listdir())
os.chdir(main_dir)

# GET HAND XRAY

#Change columns name
path_train = path_train.rename(columns = {'MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png':'img'})
path_valid = path_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png':'img'})

# Use function and get only HAND directories
path_train = get_selected_feature(path_train,"MURA-v1.1/train/XR_HAND")
path_valid = get_selected_feature(path_valid,"MURA-v1.1/valid/XR_HAND")


# Get radiologist predictions  :)
#label_train = label_train.rename(columns = {"MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/":'img'})
#label_valid = label_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/':'img'})

#label_train = get_selected_feature(label_train,"MURA-v1.1/train/XR_HAND")
#label_valid = get_selected_feature(label_valid,"MURA-v1.1/valid/XR_HAND")


# =============================================================================
#                                   Plot Samples
# =============================================================================

from keras.preprocessing import image

plt.figure(1)
plt.suptitle("XR_HAND",fontsize=30)

plt.subplot(1,3,1)
img_path = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image1.png')
img = image.load_img(img_path)
plt.imshow(img)
plt.title("image1")

plt.subplot(1,3,2)
img_path1 = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image2.png')
img1 = image.load_img(img_path1)
plt.imshow(img1)
plt.title("image2")

plt.subplot(1,3,3)
img_path2 = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image4.png')
img2 = image.load_img(img_path2)
plt.imshow(img2)
plt.title("image3")



# =============================================================================
#                               Data Overview
# =============================================================================

img_array = image.img_to_array(img)

print("\nImage stats:\n")
print("Image height : ",img_array.shape[0])
print("Image width : ",img_array.shape[1])
print("Image depth : ",img_array.shape[2])
print("\nHand images : ")
print("\nTrain images : ",len(path_train))
print("Valid images : ",len(path_valid))

del img , img1 , img2 , img_path1 , img_path2 , img_path

