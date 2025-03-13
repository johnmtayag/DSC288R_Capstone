import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import yaml

from scripts.preprocessing import get_class_weights, get_value_weights
###

##### Import common.yaml
with open("config/common.yaml", "r") as common_params:
    common = yaml.safe_load(common_params)
    # unpack params
    conditions = common["conditions"]
    class_cols = common["class_cols"]
    n_classes = common["n_classes"]

def get_valid_paths(old_root):
    """
    The training dataset lists filepaths for the entire dataset, so this extracts only the filepaths available in the directory
    """
    valid_paths = []
    for patient in os.listdir(old_root):
        patient_path = old_root + patient
        studies = os.listdir(patient_path)
        for study in studies:
            study_path = patient_path + "/" + study
            images = os.listdir(study_path)
            for image in images:
                image_path = study_path + "/" + image
                valid_paths.append(image_path)

    return valid_paths

def get_valid_paths2(root):
    """
    The training dataset lists filepaths for the entire dataset, so this extracts only the filepaths available in the directory. Same as the original function, except in the flattened structure
    """
    valid_paths = []
    for img in os.listdir(root):
        img_path = root + img
        valid_paths.append(img_path)

    return valid_paths

def copy_directory_structure(source, destination):
    """Copy directory structure from the source path to the destination path"""
    if not os.path.exists(destination):
        os.makedirs(destination)
    for dir_path, dir_names, file_names in os.walk(source):
        full_path = os.path.join(destination, dir_path[len(source):])
        if os.path.isdir(full_path):
            continue
        else:
            os.mkdir(full_path)

def get_plot_histograms(df, class_cols, suptitle):

    fig,axes = plt.subplots(2,3, figsize=(7,5))
    for i,(col,ax) in enumerate(zip(class_cols, axes.flatten())):
        df[col].hist(ax=ax)
        ax.set_title(col)

    # Turn off unused axes
    for i in range(len(axes.flatten())):
        if i >= len(class_cols):
            axes.flatten()[i].axis("off")
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

##### Preprocessing dataframes ##############################################

def process_dataframes(df, source_root, old_root, output_root, dims):
    """
    Actions taken to process the data frames:
    1) Loop through the specified source_root folder to find all image paths that exist
        a. Store each path string in the valid_paths list
    2) For each image metadata row in df, add new path columns:
        a. floating_file_path: The image file path without the root folder (ex: "S:/Root/patientX/image.jpg" to "patientX/image.jpg")
        b. source_file_path: The file path to the original image saved from the blob
        c. base{dim}_file_path: The file path to the saved dim x dim processed image
        d. base{dim}_file_path2: The file path to the saved enhanced dim x dim processed image
    3) Filter out any images whose source_file_path is not contained in the valid_paths (created in step 1)
    4) Return the processed dataframes
    """
    # Create a copy of df
    df2 = df.copy()
    # Loop through the directories and get paths that exist
    valid_paths = get_valid_paths(source_root)
    # Separate the floating file paths from the old_root
    df2["floating_file_path"] = df2["Path"].str[len(old_root):]
    # Create a filepath to the image from the specified source_root
    df2["source_file_path"] = source_root + df2["floating_file_path"]
    # Add new file paths for processed images
    for dim in dims:
        dim = str(dim)
        col1, col2 = f"base{dim}_file_path", f"base{dim}_file_path2"
        df2[col1] = output_root + df2["floating_file_path"].str[:-4] + f"_{dim}.jpg"
        df2[col2] = df2[col1].str[:-4] + "_usm.jpg"
    # Filter the dfs for only valid paths from the source file paths
    df2 = df2[df2["source_file_path"].isin(valid_paths)]

    return df2

def resample(df, class_cols, n_samples):
    def get_class_and_value_weights(df, cols):
        """Gives a matrix of weights, one for each class value, with higher weights for class vlues with fewer instances"""
        # Count the frequency of each class value
        counts = np.zeros((3, len(cols)))
        for j,col in enumerate(cols):
            for i in range(0,3,1):
                counts[i,j] = df.loc[df[col]==i, col].count()
    
        # Get fractional counts, then remove the 0 instances
        class_value_probs = counts[1:,:]
    
        # Get weights based on inverse distributions
        class_value_probs = 1 / class_value_probs
        class_value_probs = class_value_probs / np.sum(class_value_probs)

        # print(class_value_probs.ravel())
        # print()
    
        return class_value_probs.ravel()
    
    def get_weighted_sampled_index(df, class_cols, n_samples):
        """
        Given the input parameters, resample from df with replacement with this criteria:
            1) Columns that are more imbalanced have a higher probability of being chosen
            2) Values that are more imbalanced have a higher probability of being chosen
        For example, sample 1 may be an image where "Pleural Other" is "2"
    
        For this, don't include the weights for 0 --> Only sample when the value==1 or value==2
    
        Return results as a batch of image indexes from df
        """
        batch = []
        df2 = df[class_cols].copy()
        df2_ind = list(df2.index)

        for i in range(1, n_samples+1):
            class_value_probs = get_class_and_value_weights(df2.loc[df2_ind,:], class_cols)
            samp_ind = np.random.choice(range(2 * len(class_cols)),
                                        size = 1,
                                        replace=False,
                                        p = class_value_probs)
            col_ind = (samp_ind) % len(class_cols)
            val = ((samp_ind) // len(class_cols)) + 1

            # print(samp_ind)
            # print(col_ind, val)

            subset_df = df[class_cols[col_ind[0]]]
            subset_df = subset_df[subset_df == val[0]]

            # print(subset_df)

            if len(subset_df) == 0:
                continue
            elif len(subset_df) == 1:
                row_idx = subset_df.index[0]
            else:
                row_idx = subset_df.sample(1).index[0]
            #
            batch.append(row_idx)
            df2_ind.append(row_idx)
            # print(len(batch), len(df2_ind), len(df2.loc[df2_ind,:]))
    
        return batch

    # def get_and_print_new_weights(df, class_cols, n_samples):
    #     ### Get class and value weights
    #     print(f"New class weights for {n_samples} samples")
    #     classWeights0 = get_class_weights(df, class_cols)
    #     for weight,class_col in zip(classWeights0, class_cols):
    #         print(f"{class_col}: {weight}")
            
    #     print()
        
    #     print(f"New value weights for {n_samples} samples")
    #     valueWeights0 = get_value_weights(df, class_cols)
    #     for weight,val in zip(valueWeights0, [0,1,2]):
    #         print(f"{val}: {weight}")
    
    #     return (classWeights0, valueWeights0)

    # Get a batch of indexes sampled based on class and value frequencies
    batch = get_weighted_sampled_index(df, 
                                       class_cols,
                                       n_samples)
    #
    # classWeights2, valueWeights2 = get_and_print_new_weights(output_df, class_cols, n_samples)

    # return (output_df, classWeights2, valueWeights2)
    return output_df
    
    
    