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

def run_pipeline(output_df, dims, skip_if_image_exists=True, **kwargs):
    """ Wraps around the individual pipeline actions"""
    n_instances = len(output_df)
    for i,(input_path) in enumerate(output_df["source_file_path"], start=1):
        if i % 1000 == 0:
            print(f"{datetime.datetime.now()}: Completed {i}/{n_instances}")
        # If processed image already exists, skip
        if skip_if_image_exists:
            all_images_exist = True
            for dim in dims:
                if not os.path.exists(output_df.loc[i - 1,f"base{str(dim)}_file_path"]): 
                    all_images_exist = False
                if not os.path.exists(output_df.loc[i - 1,f"base{str(dim)}_file_path2"]): 
                    all_images_exist = False
            if all_images_exist:
                continue
        #
        img_list, img_enhanced_list = pipeline(input_path, dims, **kwargs)
        # Save each image as needed
        for j,(img0,img_enhanced0,dim) in enumerate(zip(img_list, img_enhanced_list, dims)):
            output_path = output_df.loc[j,f"base{str(dim)}_file_path"]
            if not os.path.exists(output_path): 
                if not skip_if_image_exists:
                    Image.fromarray(img0).save(output_path, "JPEG", quality=95)     
            output_paths2 = output_df.loc[j,f"base{str(dim)}_file_path"][:-4] + "_usm.jpg"
            if not os.path.exists(output_paths2): 
                if not skip_if_image_exists:
                    Image.fromarray(img_enhanced0).save(output_path2, "JPEG", quality=95)    

def pipeline(path, dims, usm_weight=1.2, usm_sigma=10, he_sigma=5, scale_min=0, scale_max=255, **kwargs):
    """Contains the image transformations as a single pipeline, outputs a "raw" and "enhanced" variation"""
    def initial_pipeline(img, dims):
        output_imgs = []
        for dim in dims:
            img0 = v2.Resize((dim,dim), interpolation=v2.InterpolationMode.BILINEAR)(img)
            # Convert back to images
            img0 = (scale_range(img0.numpy(), scale_min, scale_max).astype(np.uint8))[0,:,:]
            output_imgs.append(img0)
        return output_imgs
    #
    def enhancement_pipeline(imgs, usm_sigma, usm_weight, he_sigma, scale_min, scale_max):
        output_imgs = []
        for img0 in imgs:
            img0 = scale_range(img0, scale_min, scale_max)
            img0 = unsharp_masking(img0, usm_sigma, usm_weight, scale_min, scale_max)
            img0 = histogram_equalization(img0, scale_min, scale_max, he_sigma).astype(np.uint8)
            output_imgs.append(img0)
        return output_imgs  
    #####
    # Load image as tensor
    img = decode_image(path).float()
    # Perform initial processsing (resizing and range scaling)
    output_imgs = initial_pipeline(img, dims)
    output_imgs_enhanced = [img0.copy() for img0 in output_imgs]
    # Copy image and apply additional enhancements
    output_imgs_enhanced = enhancement_pipeline(output_imgs_enhanced, usm_sigma, usm_weight, he_sigma, scale_min, scale_max, **kwargs)

    return (output_imgs, output_imgs_enhanced)

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
    
    # Find the total number of values across ALL classes
    sum_counts = len(df) * len(cols)
    
    # Find the fraction of all values that are either (1) or (2) --> This is the probability of occurrence
    class_value_probs = np.array([count/sum_counts for count in counts.values()])
    
    ### Given these probabilities, create a probability distribution of selecting either 1 or 2 which boosts the counts of the underrepresented value
    # Perform 1 / prob to amplify the smaller numbers and minimize the bigger numbers
    class_value_probs = 1 / class_value_probs
    
    # Adjust so that the probabilities add up to 1
    class_value_probs = class_value_probs / np.sum(class_value_probs)

    return class_value_probs