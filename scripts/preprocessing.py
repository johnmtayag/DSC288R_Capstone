import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dct, idct
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from skimage.measure import label, regionprops
import math

import os

##### Rescale an array's range

def scale_range(X, new_min, new_max):
    """Scale all values in X to the range [new_min,new_max]"""
    Xmin, Xmax = X.min(), X.max()
    return ((X - Xmin) / (Xmax - Xmin)) * (new_max - new_min) + new_min

##### Extended difference of Gaussians

def xdog(X, k, sigma, tau, ep, phi, scale_min, scale_max, truncate, threshold=None):
    """Perform an extended difference of gaussians for edge detection"""
    
    def threshold_binary(X, ep):
        """Set all values in X equal to or above the threshold to 1, set all other values to 0"""
        X[X < ep] = 0
        X[X != 0] = scale_max
        return X
    
    def threshold_tanh(X, ep, phi):
        """Set all values in X equal to or above the threshold to 1, use tanh to determine the other values"""
        X_shape = X.shape
        X = X.flatten()
        X[X <= ep] = (1 + np.tanh(phi * (X[X < ep] - ep))) * (scale_max - scale_min) + scale_min
        X[X > ep] = scale_max
        return X.reshape(X_shape)

    ###
    
    db1 = gaussian_filter(X, sigma=sigma, truncate=truncate)
    db2 = gaussian_filter(X, sigma=sigma * k, truncate=truncate)
    dog = ((1 + tau) * db1) - (tau * db2)
    dog = scale_range(dog, scale_min, scale_max)

    # Threshold
    if threshold == "binary":
        return threshold_binary(dog, ep)
    elif threshold == "tanh":
        return threshold_tanh(dog, ep, phi)
    else:
        return dog

##### Adaptive Mask: Filtering out the diaphragm

def get_thresholded_image(X, threshold=None):
    if threshold is None: threshold = 0.9
    input_min = X.min()
    T = input_min + threshold * (X.max() - input_min)
    X2 = X.copy()
    X2[X2 < T] = 0
    X2[X2 >=T] = 1

    return X2
    
def fill_gaps(X, sigma=None):
    if sigma is None: sigma = 1

    X2 = X.copy()
    # Fill in gaps
    X2 = binary_fill_holes(X2)
    # Add blur
    X2 = gaussian_filter(X2, sigma=sigma)

    # Binarize the output
    X2[X2 < X2.max()] = 0

    return X2

def get_largest_region_mask(X):
    # Get the largest region (that is not the background 0 region)
    region_sizes = np.bincount(X.ravel())
    largest_region_ind = region_sizes[1:].argmax() + 1   # 1 skips the background region
    mask = X.copy()
    mask[mask != largest_region_ind] = 0

    return mask

def filter_out_diaphragm(X, threshold=0.9, sigma=1):
    """
    Filter out the diaphragm adaptively --> Locate the largest region of the brightest value
    """
    ### Extract and label areas where threshold value is exceeded
    img_thresholded = get_thresholded_image(X, threshold)
    img_labeled = label(img_thresholded)
    ### Get the largest region, then fill in the gaps
    mask = get_largest_region_mask(img_labeled)
    mask = fill_gaps(mask, sigma)
    ### Remove the identified largest region
    X2 = X.copy()
    X2[mask > 0] = X.min()
    return mask

##### Adaptive cropping of border regions

def crop_borders(img_arr, threshold_range, crop_q1_threshold, crop_q3_threshold):
    """
    Remove borders by checking the difference between max/min values along edge rows/columns
    threshold_range: The range of values in a row/column must exceed the given threshold range
    crop_q1_threshold: The maximum of the row/column must exceed this threshold
    crop_q3_threshold: The minimum of the row/column must exceed this threshold
    """
    def check_conditions(min_val, max_val):
        if max_val < crop_q1_threshold: return False
        if min_val > crop_q3_threshold: return False
        if max_val - min_val < threshold_range: return False
        return True
    #####
    n_rows, n_cols = img_arr.shape
    # top
    for top_ind in np.arange(0, n_rows - 1, 1):
        row = img_arr[top_ind, :]
        row_min, row_max = row.min(), row.max()
        if check_conditions(row_min, row_max):
            break
    
    # bottom
    for bottom_ind in np.arange(n_rows - 1, 0, -1):
        row = img_arr[bottom_ind, :]
        row_min, row_max = row.min(), row.max()
        if check_conditions(row_min, row_max):
            break
    
    # left
    for left_ind in np.arange(0, n_cols - 1, 1):
        col = img_arr[:, left_ind]
        col_min, col_max = col.min(), col.max()
        if check_conditions(col_min, col_max):
            break
    
    # right
    for right_ind in np.arange(n_cols - 1, 0, -1):
        col = img_arr[:, right_ind]
        col_min, col_max = col.min(), col.max()
        if check_conditions(col_min, col_max):
            break
    
    return img_arr[top_ind:bottom_ind, left_ind:right_ind]

##### Find best 90-degree rotation

def get_best_rotation(img_arr, img_arr1, img_arr2):
    """Find the best 90 degree rotation of img where the harmonic mean of pixel-wise Euclidean distances between
    the img and two input images is minimized"""
    img_arrs = [img_arr] + [np.rot90(img_arr, k=k) for k in [1,2,3]]
    for j,img_arr_i in enumerate(img_arrs):
        d1 = np.mean((img_arr_i - img_arr1)**2)
        d2 = np.mean((img_arr - img_arr2)**2)
        hm_d = (2 / ((1/d1) + (1/d2)))
        if j == 0:
            best_img_i = img_arr_i.copy()
            best_hm_d = hm_d.copy()
        else:
            if hm_d < best_hm_d:
                best_img_i = img_arr_i.copy()
                best_hm_d = hm_d.copy()

    return best_img_i, best_hm_d

##### Histogram equalization for contrast adjustment

def get_hist(X, bins):
    hist = np.zeros(bins)
    for pixel_value in X: 
        hist[pixel_value] += 1
    return hist

def normalize_cumsum(cumsum, bins):
    cumsum_min = cumsum.min()
    numer = (cumsum - cumsum_min) * (bins - 1)
    denom = cumsum.max() - cumsum_min

    return (numer / denom).astype("uint8")

def histogram_equalization(X, scale_min, scale_max, sigma=2):
    """Increase image contrast by equalizing the histogram"""
    n_bins = scale_max - scale_min + 1
    X_flat = scale_range(X, scale_min, scale_max).astype(np.uint8).flatten()
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
