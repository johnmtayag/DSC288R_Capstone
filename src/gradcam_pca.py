import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.decomposition import PCA
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights
from torchvision.io import decode_image

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM

from src.model import Custom_ResNet50, Custom_DenseNet121, ModelWrapper
from src.preprocessing import get_hilbert_index, hilbert_ravel, scale_range

#####

def load_csv(csv_path, pca_sample_size):
    ### Load the CSV file
    try:
        df = pd.read_csv(csv_path).iloc[:pca_sample_size,:]
        if df.empty:
            raise ValueError("The CSV file is empty!")
        else:
            ### Extract conditions (excluding metadata columns)
            metadata_cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", 
                             "floating_file_path", "source_file_path", "base224_file_path",
                             "base224_file_path2", "base384_file_path", "base384_file_path2",
                             "base512_file_path", "base512_file_path2"]
            conditions = [col for col in df.columns if col not in metadata_cols and not col.endswith("_pred") and not col.endswith("_confidence")]
        
            return df, conditions
    except FileNotFoundError:
        raise ValueError(f"CSV file not found at: {csv_path}")

def load_model(model_type, model_path, n_classes, device):
    # Load Model from Checkpoint
    if model_type=="ResNet50":
        model = Custom_ResNet50(n_classes)
    elif model_type=="DenseNet121":
        model = Custom_DenseNet121(n_classes)
    else:
        raise ValueError("Invalid model_type input: Only `ResNet50` and `DenseNet121` are defined")
        
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))  # Load weights
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model

def get_image_labels(df, row_idx, conditions):
    row = df.iloc[row_idx]
    target_labels = [int(row[condition]) for condition in conditions]
    predicted_labels = [int(row[f"{condition}_pred"]) for condition in conditions]

    return target_labels, predicted_labels

def load_image(df, row_idx, image_path_column, dim, enhanced, usm_weight, usm_sigma, he_sigma, scale_min, scale_max, device):
    # -------------------------------
    # Step 1. Specify Image
    # -------------------------------
    image_path = df.loc[row_idx, image_path_column]
    
    # -------------------------------
    # Step 2. Load and Transform the Image
    # -------------------------------
    # Define transformation (Ensure the image matches model input size)
    img = scale_range(decode_image(image_path).float(), scale_min, scale_max)
    input_tensor = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).to(device)
    #
    img_np = scale_range(np.array(img), 0, 1).transpose(1, 2, 0)
    
    return input_tensor, img_np

def set_target_layer(model, model_type):
    # Ensure target layer exists
    if model_type=="ResNet50":
        try:
            target_layer = model.resnet_model.layer4[-1]  # Final ResNet block
        except AttributeError:
            raise ValueError("Could not find `model.resnet_model.layer4[-1]` in the model. Verify architecture.")
    elif model_type=="DenseNet121":
        try:
            target_layer = model.densenet_model.features.denseblock4.denselayer16.conv2  # Final ResNet block
        except AttributeError:
            raise ValueError("Could not find `model.densenet_model.features.denseblock4.denselayer16.conv2` in the model. Verify architecture.")
    else:
        raise ValueError("Invalid model_type")

    return target_layer

def get_gradcam_predictions(model, input_tensor, conditions):
    # -------------------------------
    # Step 3. Compute Model Predictions (Softmax for All Conditions)
    # -------------------------------
    num_conditions = len(conditions)

    # Get model predictions
    with torch.no_grad():
        # Forward pass: Compute predicted y by passing x to the model
        pred_i = torch.stack(model(input_tensor), dim=0)

        # Convert pred_i via Softmax and Argmax
        softmax_probabilities = torch.softmax(pred_i, dim=2)#.cpu().numpy()

        pred = torch.argmax(softmax_probabilities, dim=2)
        conf = torch.max(softmax_probabilities, dim=2).values

        softmax_probabilities = softmax_probabilities.cpu().numpy()

    return softmax_probabilities

def get_gradcam_masks(model, target_layer, input_tensor, target_labels, n_classes, device):
    gradcam_masks = []
    for cond_idx in range(n_classes):
        target = ClassifierOutputTarget(target_labels[cond_idx])  # Use predicted class for GradCAM
    
        wrapped_model = ModelWrapper(model, condition_index=cond_idx).to(device)
        cam = GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer])
    
        # Compute Grad-CAM heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())  # Normalize
        gradcam_masks.append(grayscale_cam)

    return gradcam_masks

def load_environment(csv_path, model_type, model_path, dim, device, enhanced, pca_sample_size, **kwargs):
    ### Load the csv of results
    df, conditions = load_csv(csv_path, pca_sample_size)
    n_classes = len(conditions)
    n_instances = len(df)

    ### Load the specified model
    model = load_model(model_type, model_path, n_classes, device)

    ### Set the target layer
    target_layer = set_target_layer(model, model_type)

    return (df, conditions, n_classes, n_instances, target_layer, model)
    
def gradcam_pca_analysis(model_type, dim, enhanced, n_pca_components, pca_sample_size, seed=None, **kwargs):
    ### Set the image path column
    image_path_column = "source_file_path"
    
    ### If not specified, build the csv and model file paths
    if enhanced:
        csv_path = f"results/{model_type}_BASE{dim}_ENHANCED_test_withPredictionsAndConfidences.csv"
        model_path = f"models/{model_type}_BASE{dim}_ENHANCED.pth" # << Change this to your actual model path
    else:
        csv_path = f"results/{model_type}_BASE{dim}_test_withPredictionsAndConfidences.csv"
        model_path = f"models/{model_type}_BASE{dim}.pth" # << Change this to your actual model path
    
    # -------------------------------
    # Define Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Set seed for reproducibility (optional)
    if not seed:
        seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ###### Load all items
    df, conditions, n_classes, n_instances, target_layer, model = load_environment(csv_path, model_type, model_path, dim, device, enhanced, pca_sample_size,**kwargs)

    ### Instantiate the gradcam analysis variables
    # Similarity is measured between gradcams of different conditions on the SAME x-ray
    # The similarity is then averaged across the set
    # Use dot product, L2, and cosine similarities
    gradcam_vars = dict(
        avg_dot = np.zeros((n_classes,n_classes)),
        avg_L2 = np.zeros((n_classes,n_classes)),
        avg_corr = np.zeros((n_classes,n_classes)),
        avg_explained_variance = 0,
        avg_explained_variance_ratios = np.zeros(n_pca_components)
    )

    ### Instantiate the hilbert curve index
    if dim > 224:
        hilbert_dim = 512
    else:
        hilbert_dim = 256
    n_dim = 2
    n_bits = 8
    hilbert_idxs = get_hilbert_index(hilbert_dim, n_dim, n_bits)

    ### Get a sample of n_instances row indexes from df
    # image_idxs = np.random.choice(df.index, pca_sample_size, replacement=False)
    img_idxs = df.index[:pca_sample_size]

    ### Instantiate matrices to hold data
    condition_img_arrays_mat = np.zeros((len(hilbert_idxs, pca_sample_size)))
    model_gradcamPCA_byCondition = np.zeros((n_classes, n_pca_components))

    ### Open the images and record them as a matrix
    imgs = np.zeros((pca_sample_size, dim, dim))
    for i,img_idx in enumerate(img_idxs):
        input_tensor, _ = load_image(df, image_idx, image_path_column, dim, enhanced, device)
        imgs[i,:,:] = input_tensor.cpu().numpy()
    
    ### Loop through all df images and track aggregate stats based on pca transformations
    # for counter,image_idx in enumerate(image_idxs, start=1):
    #     input_tensor, _ = load_image(df, image_idx, image_path_column, dim, enhanced, device)
    #     target_labels = [int(df.loc[image_idx, condition]) for condition in conditions]
    #     predicted_labels = [int(df.loc[image_idx, f"{condition}_pred"]) for condition in conditions]
    #     gradcam_masks = get_gradcam_masks(model, target_layer, input_tensor, target_labels, n_classes, device)
        
    #     gradcam_mat = np.zeros((n_classes, hilbert_dim**2))
    #     for cond_idx, gradcam_mask in enumerate(gradcam_masks):
    #         gradcam_mat[cond_idx, :] =  hilbert_ravel(gradcam_mask, hilbert_dim, hilbert_idxs)
        
    #     gradcam_vars = add_to_gradcam_vars(gradcam_vars, gradcam_mat, conditions, n_pca_components, counter, **kwargs)

    return gradcam_vars

def add_to_gradcam_vars(grad_vars, grad_mat, conditions, n_pca_components, counter, **kwargs):
    """
    Calculate the PCA transformation of the input gradcam masks, 
    Given:
        grad_vars: Dictionary containing aggregate variables for the analysis
        grad_mat: The array representing the input gradcam masks
        conditions: The list containing all possible conditions (classes)
        counter: The running count of gradcam images processed
    """
    # Get the PCA transformation
    pca = PCA(n_components=n_pca_components)
    pca_gradcam = pca.fit_transform(grad_mat)
    
    # Add the variance explained values to the running totals in grad_vars
    grad_vars["avg_explained_variance"] = (1 - (1/counter)) * pca.explained_variance_ratio_.sum() + (1/counter) * grad_vars["avg_explained_variance"] 
    grad_vars["avg_explained_variance_ratios"]  = (1 - (1/counter)) * pca.explained_variance_ratio_ + (1/counter) * grad_vars["avg_explained_variance_ratios"]
    
    # Calculate the pairwise correlations between the pca representations of each condition's gradcam mask
    corr = pd.DataFrame(pca_gradcam).T.corr().to_numpy()
    grad_vars["avg_corr"] = (1 - (1/counter)) * corr + (1/counter) * grad_vars["avg_corr"]
    
    # Calculate similarity values between conditions, then add to running totals
    for ind1,class_col1 in enumerate(conditions):
        for ind2, class_col2 in enumerate(conditions):
            dot_prod = np.dot(pca_gradcam[ind1,:], pca_gradcam[ind2,:])
            L2 = np.sqrt(np.sum((pca_gradcam[ind1,:] - pca_gradcam[ind2,:])**2))
            #
            grad_vars["avg_dot"][ind1, ind2] = (1 - (1/counter)) * dot_prod + (1/counter) * grad_vars["avg_dot"][ind1, ind2]
            grad_vars["avg_L2"][ind1, ind2] = (1 - (1/counter)) * L2 + (1/counter) * grad_vars["avg_L2"][ind1, ind2]
    return grad_vars

