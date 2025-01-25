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

##### Extended difference of Gaussians

def scale_range(X, new_min, new_max):
    """Scale all values in X to the range [new_min,new_max]"""
    Xmin, Xmax = X.min(), X.max()
    return ((X - Xmin) / (Xmax - Xmin)) * (new_max - new_min) + new_min

def xdog(X, k, sigma, tau, ep, phi, scale_min, scale_max, threshold=None):
    """Perform an extended difference of gaussians for edge detection"""
    
    def threshold_binary(X, ep):
        """Set all values in X equal to or above the threshold to 1, set all other values to 0"""
        X[X < ep] = 0
        X[X != 0] = 1
        return X
    
    def threshold_tanh(X, ep, phi):
        """Set all values in X equal to or above the threshold to 1, use tanh to determine the other values"""
        X_shape = X.shape
        X = X.flatten()
        X[X < ep] = 1 + np.tanh(phi * (X[X < ep] - ep))
        X[X >= ep] = 1
        return X.reshape(X_shape)

    ###
    
    db1 = gaussian_filter(X, sigma=sigma)
    db2 = gaussian_filter(X, sigma=sigma * k)
    dog = ((1 + tau) * db1) - (tau * db2)
    dog = scale_range(dog, scale_min, scale_max)

    # Threshold
    if threshold == "binary":
        return threshold_binary(dog, ep)
    elif threshold == "tanh":
        return threshold_tanh(dog, ep, phi)
    else:
        return dog

##### Image compression: Discrete Cosine Transform

def dct2(X, type=None, norm=None):
    if type is None: type = 2
    if norm is None: norm = "ortho"
    return dct(dct(X.T, type=type, norm=norm).T, type=type, norm=norm)

def idct2(X, type=None, norm=None):
    if type is None: type = 2
    if norm is None: norm = "ortho"
    return idct(idct(X.T, type=type, norm=norm).T, type=type, norm=norm)

def compress_image(X, new_size, type=None, norm=None):
    height, width = new_size
    return idct2(dct2(X, type, norm)[:height, :width], type, norm)

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
    return X2

##### Adaptive cropping

def crop_borders(img_arr, threshold_range, crop_q1_threshold, crop_q3_threshold):
    """
    Remove borders by checking the difference between max/min values along edge rows/columns
    threshold_range: The range of values in a row/column must exceed the given threshold range
    crop_q1_threshold: The maximum of the row/column must exceed this threshold
    crop_q3_threshold: The minimum of the row/column must exceed this threshold
    """
    n_rows, n_cols = img_arr.shape
    # top
    for top_ind in np.arange(0, n_rows - 1, 1):
        row = img_arr[top_ind, :]
        row_min, row_max = row.min(), row.max()
        row_low_count = np.sum(row < threshold_range)
        row_high_count = np.sum(row > threshold_range)
        if (row_max - row_min > threshold_range) and (row_max > crop_q1_threshold) and (row_min < crop_q3_threshold):
            break
    
    # bottom
    for bottom_ind in np.arange(n_rows - 1, 0, -1):
        row = img_arr[bottom_ind, :]
        row_min, row_max = row.min(), row.max()
        row_low_count = np.sum(row < threshold_range)
        row_high_count = np.sum(row > threshold_range)
        if (row_max - row_min > threshold_range) and (row_max > crop_q1_threshold) and (row_min < crop_q3_threshold):
            break
    
    # left
    for left_ind in np.arange(0, n_cols - 1, 1):
        col = img_arr[:, left_ind]
        col_min, col_max = col.min(), col.max()
        col_low_count = np.sum(col < threshold_range)
        col_high_count = np.sum(col > threshold_range)
        if (col_max - col_min > threshold_range) and (col_max > crop_q1_threshold) and (col_min < crop_q3_threshold):
            break
    
    # right
    for right_ind in np.arange(n_cols - 1, 0, -1):
        col = img_arr[:, right_ind]
        col_min, col_max = col.min(), col.max()
        row_low_count = np.sum(row < threshold_range)
        row_high_count = np.sum(row > threshold_range)
        if (col_max - col_min > threshold_range) and (col_max > crop_q1_threshold) and (col_min < crop_q3_threshold):
            break
    
    return img_arr[top_ind:bottom_ind, left_ind:right_ind]

##### Find best 90-degree rotation

def get_best_rotation(img_arr, avg_img_arr):
    """Find the best 90 degree rotation of img where the mean of pixel-wise Euclidean distances between
    the img and avg_img is minimized"""
    img_arrs = [img_arr] + [np.rot90(img_arr, k=k) for k in [1,2,3]]
    for j,img_arr_i in enumerate(img_arrs):
        if j == 0:
            best_img_i = img_arr_i
            best_d = np.mean((img_arr_i - avg_img_arr)**2)
        else:
            d = np.mean((img_arr_i - avg_img_arr)**2)
            if d < best_d:
                best_img_i = img_arr_i
                best_d = np.mean((img_arr_i - avg_img_arr)**2)

    return best_img_i, best_d

