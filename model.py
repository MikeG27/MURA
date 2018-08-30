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
    Get hand data from all features 
    
    """
    
    data = []
    
    for i in img_path_array["img"]:
        if feature_path in i:
            data.append(i)
            
    return data

def get_data_distribution(data):
    

    distr = {}
    negative = []
    positive = []
    
    for i in path_train:
        if "negative" in i:
            negative.append(i)
        elif "study1_positive":
            positive.append(i)
    
    distr["positive"] = positive
    distr["negative"] = negative
    
    return distr


def re_path(patch_folder):
    
    """
    
    Return array of corrected pathes of images
    
    """
    
    repath = []
    
    for i in patch_folder:
        repath.append(str("/home/michal/Pulpit/MURA/"+i))
        
    return repath

def separate_images(img_path,abnormal_class,normal_class):
    
    '''
    Separate images to folder class
    
    '''
    
    n = 0 #negative index
    p = 0 #positive index
    
    for i in img_path:
        if os.path.exists(i):
            
            if "negative" in i:
                n += 1
                src = i
                dest = (normal_class)
                shutil.copy(src,dest+"/normal"+str(n)+".png")
            
            elif "positive" in i:
                p += 1
                src = i
                dest = (abnormal_class)
                shutil.copy(src,dest+"/abnormal"+str(p)+".png")


# =============================================================================
#                           Create folder structure
# =============================================================================

os.chdir("/home/michal/Pulpit/MURA/MURA-v1.1") # Go to data directory
main_dir = os.getcwd()
main_structure = get_main_f(os.listdir())


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
os.chdir(main_structure[2])
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
label_train = label_train.rename(columns = {"MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/":'img'})
label_valid = label_valid.rename(columns = {'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/':'img'})
label_train = get_selected_feature(label_train,"XR_HAND")
label_valid = get_selected_feature(label_valid,"XR_HAND")


# Get normal/abnormal distribution
distr_hand_train = get_data_distribution(path_train)
distr_hand_valid = get_data_distribution(path_valid)


# Data separation

# 1.Rename pathes
path_train = re_path(path_train)
path_valid = re_path(path_valid)

# 2. Separate images to folder classes

separate_images(path_train,hand_train_abnormal,hand_train_normal)
separate_images(path_valid,hand_valid_abnormal,hand_valid_normal)



# =============================================================================
#                                    Plots
# =============================================================================

from keras.preprocessing import image

plt.figure(1,figsize=(25,12))
plt.suptitle("XR_HAND",fontsize=30)

plt.subplot(1,3,1)
img_path = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image1.png')
img = image.load_img(img_path)
plt.imshow(img)
plt.title("image1")
plt.xlabel("Normal",fontsize=15)

plt.subplot(1,3,2)
img_path1 = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image2.png')
img1 = image.load_img(img_path1)
plt.imshow(img1)
plt.title("image2")
plt.xlabel("Normal",fontsize=15)


plt.subplot(1,3,3)
img_path2 = ('/home/michal/Pulpit/MURA/MURA-v1.1/train/XR_HAND/patient09900/study1_positive/image4.png')
img2 = image.load_img(img_path2)
plt.imshow(img2)
plt.title("image3")
plt.xlabel("Normal",fontsize=15)

plt.savefig("Sample.png")




# =============================================================================
#                               Data Overview
# =============================================================================

img_array = image.img_to_array(img)


print("\n1. Hand images : ")

print("\nTrain images : ",len(path_train))
print("Valid images : ",len(path_valid))

print("\n2. Hand images overview : \n")

print("Train abnormal img : ",len(os.listdir(hand_train_abnormal)))
print("Train normal img : ",len(os.listdir(hand_train_normal)))
print("Valid abnormal img : ",len(os.listdir(hand_valid_abnormal)))
print("Valid normal img : ",len(os.listdir(hand_valid_normal)))

print("\n3. Image stats:\n")
print("Image height : ",img_array.shape[0])
print("Image width : ",img_array.shape[1])
print("Image depth : ",img_array.shape[2])

del img , img1 , img2 , img_path1 , img_path2 , img_path


# =============================================================================
#                                Model 
# =============================================================================





