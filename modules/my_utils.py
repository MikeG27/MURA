# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd


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
    
    for i in data:
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