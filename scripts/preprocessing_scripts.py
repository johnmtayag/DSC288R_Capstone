# Standard python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dct, idct
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from skimage.measure import label, regionprops
import math
from PIL import Image
import datetime
import os

# Necessary installs
# --- PyTorch
import torch
from torchvision.transforms import v2
from torchvision.io import decode_image
# --- hilbert
from hilbert import decode, encode

##### Preprocessing pipeline ##############################################

def run_pipeline(output_df, dim, training=False, skip_if_image_exists=True, **kwargs):
    """ Wraps around the individual pipeline actions"""
    n_instances = len(output_df)
    
    input_paths = output_df["source_file_path"]
    output_paths = output_df[f"base{str(dim)}_file_path"]
    output_paths2 = output_df[f"base{str(dim)}_file_path"].str[:-4] + "_usm.jpg"
    #
    rotation90_angles = np.random.choice([0, 90, 180, 270], n_instances, replace=True)
    rotation_angles = scale_range(np.random.random(n_instances), -5, 5)
    
    for i,(input_path, output_path, output_path2) in enumerate(zip(input_paths, output_paths, output_paths2), start=1):
        if i % 1000 == 0:
            print(f"{datetime.datetime.now()}: Completed {i}/{n_instances}")
        # If processed image already exists, skip
        if skip_if_image_exists:
            if os.path.exists(output_path2): 
                continue
        rotation90_angle = rotation90_angles[i]
        rotation_angle = rotation_angles[i]
        #
        img, img_enhanced = pipeline(input_path, 
                                     dim,
                                     rotation90_angle=rotation90_angle, 
                                     rotation_angle=rotation_angle, 
                                     training=training, 
                                     **kwargs)
        #
        Image.fromarray(img).save(output_path, "JPEG", quality=95)
        Image.fromarray(img_enhanced).save(output_path2, "JPEG", quality=95)

def pipeline(path, dim, usm_weight, usm_sigma, he_sigma, scale_min, scale_max, rotation90_angle, rotation_angle, seed=None, training=False, **kwargs):
        """Contains the image transformations as a single pipeline, outputs a "raw" and "enhanced" variation"""
        def enhancement_pipeline(img, usm_sigma, usm_weight, he_sigma, scale_min, scale_max):
            img = unsharp_masking(img, usm_sigma, usm_weight, scale_min, scale_max)
            img = histogram_equalization(img.numpy().astype(np.uint8), scale_min, scale_max, he_sigma)
            img = torch.from_numpy(img).to(float)
            # img = v2.functional.equalize(img.to(torch.uint8)).to(float)
            return img
        #
        def prepare_for_model(img, dim, training, rotation90_angle, rotation_angle):
            if training:
                # Additional preprocessing for randomness was moved to the dataloader
                pass
            img = v2.Resize((dim,dim), interpolation=v2.InterpolationMode.BILINEAR)(img)
            # Convert back to images
            img = (scale_range(img.numpy(), scale_min, scale_max).astype(np.uint8))[0,:,:]
            return img
        #####
        # Load image as tensor
        img = decode_image(path).float()
        # Copy image and apply additional enhancements
        img_enhanced = enhancement_pipeline(img.clone(), usm_sigma, usm_weight, he_sigma, scale_min, scale_max, **kwargs)
        # Apply random transformations and prepare for the model
        img = prepare_for_model(img, dim, training, rotation90_angle, rotation_angle)
        img_enhanced = prepare_for_model(img_enhanced, dim, training, rotation90_angle, rotation_angle)
    
        return (img, img_enhanced)

#####

##### Rescale an array's range
def scale_range(X, new_min, new_max):
    """Scale all values in X to the range [new_min,new_max]"""
    Xmin, Xmax = X.min(), X.max()
    return ((X - Xmin) / (Xmax - Xmin)) * (new_max - new_min) + new_min

##### Histogram equalization for contrast adjustment
def get_hist(X, bins):
    hist = np.zeros(bins)
    for pixel_value in X: 
        hist[pixel_value] += 1
    return hist
#
def normalize_cumsum(cumsum, bins):
    cumsum_min = cumsum.min()
    numer = (cumsum - cumsum_min) * (bins - 1)
    denom = cumsum.max() - cumsum_min

    return (numer / denom).astype("uint8")
#
def histogram_equalization(X, scale_min, scale_max, sigma=2):
    """Increase image contrast by equalizing the histogram"""
    n_bins = scale_max - scale_min + 1
    X_flat = scale_range(X, scale_min, scale_max).flatten().astype(np.uint8)
    X_hist = get_hist(X_flat, n_bins)
    X_hist2 = np.cumsum(X_hist)
    X_hist2 = normalize_cumsum(X_hist2, n_bins)
    X_hist2 = gaussian_filter(X_hist2, sigma)
    return X_hist2[X_flat].reshape(X.shape)

##### Peak Signal-to-Noise Ratio
def get_psnr(img_arr, img_arr2, scale_min, scale_max):
    """ Calculates the peak signal-to-noise ratio in dB between an edited image (img_arr2) and the original (img_arr)"""
    img_arr = scale_range(img_arr, scale_min, scale_max)
    img_arr2 = scale_range(img_arr2, scale_min, scale_max)
    mse = np.mean((img_arr.astype(np.float64) - img_arr2.astype(np.float64))**2)
    if mse==0: 
        return np.inf
    else:
        return 20 * math.log10(scale_max) - 10 * math.log10(mse)

##### Unsharp Masking for Edge Enhancements 
def unsharp_masking(img_arr, sigma, weight, scale_min, scale_max):
    """Enhance the image by increasing edge contrasts"""
    img_arr = scale_range(img_arr, scale_min, scale_max)
    blurred = gaussian_filter(img_arr, sigma)
    return img_arr + (img_arr - blurred) * weight

##### Hilbert Curve encoding for PCA analysis of grad-CAM masks
def get_hilbert_index(hilbert_dim, n_dim, n_bits):
    """
    Get the index to ravel the img in a hilbert curve
    n_dim: The number of dimensions of the input
    n_bits: The number of times the hilbert pattern should repeat along each axis
    """
    return decode(np.array(np.arange(hilbert_dim**2)), n_dim, n_bits)
#
def hilbert_ravel(img, hilbert_dim, hilbert_idxs):
    """Rather than ravel by row, ravel as a hilbert curve to preserve spatial relations"""
    # Convert input img to size 256x256 (must be a power of 2 to use a hilbert curve)
    img = v2.Resize((hilbert_dim,hilbert_dim), interpolation=v2.InterpolationMode.BILINEAR)(torch.from_numpy(img).unsqueeze(0))
    img = img.numpy()[0,:,:]
    # 
    
    return np.array([img[ind[0], ind[1]] for ind in hilbert_idxs])

##### Getting class and value weights based on data frequencies
def get_class_weights(df, cols):
    """Gives a set of weights, one for each class, with higher weights for classes with fewer non-zero instances"""
    # Get the count of non-zero instances per class
    col_sums = df[cols].map(lambda x: 1 if x==1 else 1 if x==2 else 0).sum(axis=0)
    
    # Get the total count of ALL instances
    df_sum = len(df)
    
    # Get the fraction of non-zero instances for each class
    class_probs = col_sums / df_sum
    
    # Perform 1 / prob to amplify the smaller numbers and minimize the bigger numbers
    class_probs = 1 / class_probs
    
    # Adjust so that the probabilities add up to 1
    class_probs = class_probs / class_probs.sum()

    return class_probs
#
def get_value_weights(df, cols):
    """Gives a set of weights, one for each non-zero value, with higher weights for values with fewer instances"""

    ### Get the probability that one of the classes will be either (1) or (2)
    # Find the count of each instance of 1 and 2 across ALL classes
    counts = {}
    for col in cols:
        for i in range(0,3,1):
            counts[i] = counts.get(i, 0) + df.loc[df[col]==i ,col].count()

    # print(counts)
    
    # Find the total number of values across ALL classes
    sum_counts = len(df) * len(cols)
    
    # Find the fraction of all values that are either (1) or (2) --> This is the probability of occurrence
    class_value_probs = np.array([count/sum_counts for count in counts.values()])

    # print(class_value_probs)
    
    ### Given these probabilities, create a probability distribution of selecting either 1 or 2 which boosts the counts of the underrepresented value
    # Perform 1 / prob to amplify the smaller numbers and minimize the bigger numbers
    class_value_probs = 1 / class_value_probs
    
    # Adjust so that the probabilities add up to 1
    class_value_probs = class_value_probs / np.sum(class_value_probs)

    return class_value_probs