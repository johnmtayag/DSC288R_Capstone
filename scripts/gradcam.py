import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.decomposition import PCA
from PIL import Image
import yaml

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

from scripts.model import Custom_ResNet50, Custom_DenseNet121, ModelWrapper
from scripts.preprocessing import get_hilbert_index, hilbert_ravel, scale_range, pipeline

##### Import common.yaml
with open("config/common.yaml", "r") as common_params:
    common = yaml.safe_load(common_params)
    # unpack params
    conditions = common["conditions"]
    class_cols = common["class_cols"]
    n_classes = common["n_classes"]
    
#####

def load_csv(csv_path, pca_sample_size):
    ### Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("The CSV file is empty!")
        if pca_sample_size:
            df = df.iloc[:pca_sample_size,:]
        return df
    except FileNotFoundError:
        raise ValueError(f"CSV file not found at: {csv_path}")

def load_model(model_type, model_path, device):
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
    
def load_image(df, row_idx, image_path_column, dim, enhanced, device, usm_weight=1.2, usm_sigma=10, he_sigma=5, scale_min=0, scale_max=255):
    # -------------------------------
    # Step 1. Specify Image
    # -------------------------------
    image_path = df.loc[row_idx, image_path_column]
    
    # -------------------------------
    # Step 2. Load and Transform the Image
    # -------------------------------
    if image_path_column == "source_file_path":
        if enhanced:
            _, imgs = pipeline(image_path, [dim], usm_weight=usm_weight, usm_sigma=usm_sigma, he_sigma=he_sigma, scale_min=scale_min, scale_max=scale_max)
        else:
            imgs, _ = pipeline(image_path, [dim], usm_weight=usm_weight, usm_sigma=usm_sigma, he_sigma=he_sigma, scale_min=scale_min, scale_max=scale_max)
        img = imgs[0]
        img = torch.tensor(np.array(Image.fromarray(img).convert("RGB"))).permute(2,0,1)
    else:
        # Define transformation (Ensure the image matches model input size)
        img = decode_image(image_path).float()
    img = scale_range(img, scale_min, scale_max)
    input_tensor = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0).to(device)
    #
    img_np = scale_range(np.array(img), 0, 1).transpose(1, 2, 0)
    
    return input_tensor, img_np

def load_environment(csv_path, row_idx, model_type, model_path, device, dim, enhanced, image_path_column, pca_sample_size=None, seed=None, scale_min=0, scale_max=255, usm_sigma=10, usm_weight=1.2, he_sigma=5, **kwargs):
    ### Load the csv of results
    df = load_csv(csv_path, pca_sample_size)

    ### Get the target and predicted image labels
    target_labels, predicted_labels = get_image_labels(df, row_idx)

    ### Load the specified model
    model = load_model(model_type, model_path, device)

    ### Set the target layer
    target_layer = set_target_layer(model, model_type)

    ### Load the specified image as a tensor (for gradcam) and numpy array (for plotting)
    input_tensor, img_np = load_image(df, row_idx, image_path_column, dim, enhanced, device, usm_weight, usm_sigma, he_sigma, scale_min, scale_max)

    return (df, target_labels, predicted_labels, target_layer, model, input_tensor, img_np)

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

def get_image_labels(df, row_idx):
    row = df.iloc[row_idx,:]
    target_labels = [int(row[condition]) for condition in conditions]
    predicted_labels = [int(row[f"{condition}_pred"]) for condition in conditions]

    return target_labels, predicted_labels

def generate_gradcams(model_type, dim, enhanced, row_idx, csv_path, model_path, device, **kwargs):
    
    ### Load all items
    df, target_labels, predicted_labels, target_layer, model, input_tensor, img_np = load_environment(csv_path, row_idx, model_type, model_path, device, dim, enhanced, **kwargs)
    # Get patient ID
    patientID = df.loc[row_idx, "floating_file_path"].split("/")[0]

    ### Get predictions
    softmax_probabilities = get_gradcam_predictions(model, input_tensor)

    ### Get gradcam masks
    gradcam_masks = get_gradcam_masks(model, target_layer, input_tensor, predicted_labels, device)

    ### Get overlayed gradcam masks
    gradcam_images = overlay_gradcam_masks(img_np, gradcam_masks)

    overlay_kwargs = dict(
        img_np=img_np,
        softmax_probabilities=softmax_probabilities,
        target_labels=target_labels,
        predicted_labels=predicted_labels,
    )

    return gradcam_masks, gradcam_images, patientID, overlay_kwargs

def wrapper_generate_gradcams(model_type, dim, enhanced, row_idx, csv_path=None, model_path=None, 
scale_min=0, scale_max=255, usm_sigma=10, usm_weight=1.2, he_sigma=5, **kwargs):
    
    ### Set the image path column
    image_path_column = "source_file_path"
    
    ### If not specified, build the csv and model file paths
    if not csv_path:
        if enhanced:
            csv_path = f"results/{model_type}_BASE{dim}_ENHANCED_test_withPredictionsAndConfidences.csv"
        else:
            csv_path = f"results/{model_type}_BASE{dim}_test_withPredictionsAndConfidences.csv"
    if not model_path:
        if enhanced:
            model_path = f"models/{model_type}_BASE{dim}_ENHANCED.pth" # << Change this to your actual model path
        else:
            model_path = f"models/{model_type}_BASE{dim}.pth" # << Change this to your actual model path
    
    # -------------------------------
    # Define Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------
    # Wrap all variables into kwargs dicts
    # -------------------------------
    gradcam_kwargs = dict(
        model_type=model_type, 
        dim=dim, 
        enhanced=enhanced,
        row_idx=row_idx,
        image_path_column=image_path_column,
        csv_path=csv_path, 
        model_path=model_path,
        device=device
    )
        
    pipeline_kwargs = dict(
        scale_min=scale_min,
        scale_max=scale_max,
        usm_sigma=usm_sigma,
        usm_weight=usm_weight,
        he_sigma=he_sigma,
    )

    gradcam_masks, gradcam_images, patientID, overlay_kwargs = generate_gradcams(**{**gradcam_kwargs, **pipeline_kwargs})

    ### Create the sup title for the figure
    fig_suptitle = (
        f"Model: {model_type}\n" + \
        f"{"Enhanced" if enhanced else "Raw"} Image Resolution: ({dim} x {dim})"
    )
    overlay_kwargs["fig_suptitle"] = fig_suptitle

    ### Save path for printed grad-CAM images
    if enhanced:
        overlay_kwargs["save_path"] = f"gradcam/ALL_ENHANCED_{model_type}_BASE{dim}_allGradcams.png"
    else:
        overlay_kwargs["save_path"] = f"gradcam/ALL_RAW_{model_type}_BASE{dim}_allGradcams.png"
    
    return gradcam_masks, gradcam_images, patientID, overlay_kwargs

def get_gradcam_predictions(model, input_tensor):
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

def get_gradcam_masks(model, target_layer, input_tensor, predicted_labels, device):
    gradcam_masks = []
    for cond_idx in range(n_classes):
        target = ClassifierOutputTarget(predicted_labels[cond_idx])  # Use predicted class for GradCAM
    
        wrapped_model = ModelWrapper(model, condition_index=cond_idx).to(device)
        cam = GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer])
    
        # Compute Grad-CAM heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())  # Normalize
        gradcam_masks.append(grayscale_cam)

    return gradcam_masks

#########################

##### Grad-CAM Overlay Functions

#########################

def overlay_gradcam_masks(img_np, gradcam_masks):
    gradcam_images = [show_cam_on_image(img_np, cam_mask * 0.9, use_rgb=True) for cam_mask in gradcam_masks]

    return gradcam_images

def plot_gradcam_overlays(img_np, gradcam_images, softmax_probabilities, target_labels, predicted_labels, save_path, fig_suptitle=None, **kwargs):
    # Prepare plots
    num_cols = 4
    num_rows = (n_classes // num_cols) + 1
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.75, num_rows * 2.25))
    ax = axes.flatten()
    gradcam_counter = 0

    for i in range(num_cols * num_rows):
        if i==0:
            ax[i].imshow(img_np, cmap="gray")
            ax[i].set_title("Original X-ray", fontsize=9)
        elif i > n_classes:
            pass
        else:
            ax[i].imshow(gradcam_images[gradcam_counter])
            probs_str = ", ".join([f"{p:.3f}" for p in softmax_probabilities[gradcam_counter][0]])  # Format probabilities
            title_text = (
                f"{conditions[gradcam_counter]}\n"
                f"Target: {target_labels[gradcam_counter]} | Pred: {predicted_labels[gradcam_counter]}\n"
                f"Probs: {probs_str}"
            )
            ax[i].set_title(title_text, fontsize=9)
            gradcam_counter += 1
        ax[i].axis("off")

    fig.suptitle(fig_suptitle)
    plt.tight_layout(pad=0.9)
    plt.savefig(save_path)
    
    return fig, axes

def plot_gradcam_overlays_gridspec(img_np, gradcam_images, softmax_probabilities, target_labels, predicted_labels, save_path, save_combined_gradcam_fig, fig_suptitle, **kwargs):
    # Prepare plots
    fig = plt.figure(figsize=(7,3.5))
    gs = fig.add_gridspec(5, 10, hspace=1.5, wspace=0.05,
                          top=0.925, bottom=0.001, left=0.005, right=0.995)
    axes = []
    
    # Add original image
    img_ax = fig.add_subplot(gs[0:3, 3:7])
    img_ax.imshow(img_np, cmap="gray")
    img_ax.set_title(fig_suptitle, fontsize=12)
    img_ax.axis("off")
    axes.append(img_ax)

    # Add 5 axes
    cond_axes = []
    for i in range(5):
        img_ax = fig.add_subplot(gs[3:5, 0+(2*i):2+(2*i)])
        cond_axes.append(img_ax)
        img_ax.axis("off")

    axes.extend(cond_axes)

    # Plot the data for each condition
    for i,(ax, gradcam_img, softmax_prob, cond, target_label, pred_label) in enumerate(zip(cond_axes,
                                                                                           gradcam_images,
                                                                                           softmax_probabilities,
                                                                                           conditions,
                                                                                           target_labels,
                                                                                           predicted_labels)):
        ax.imshow(gradcam_img)
        probs_str = ", ".join([f"{p:.2f}" for p in softmax_prob[0]])  # Format probabilities
        title_text = (
            f"{cond}\n"
            f"Target: {target_label} | Pred: {pred_label}\n"
            f"Probs: {probs_str}"
        )
        ax.set_title(title_text, fontsize=7)

    if save_combined_gradcam_fig:
        plt.savefig(save_path)
    print(f"Combined figure saved to {save_path}")
    
    return fig, axes

def save_separate_gradcams(gradcam_images, patientID, model_type, dim, **kwargs):
    for i,(gradcam_image, condition) in enumerate(zip(gradcam_images, conditions)):
        if kwargs["enhanced"]:
            save_path = f"results/gradcam/{patientID}_{model_type}_base{dim}_ENHANCED_gradcam_condition_{i}_{condition}.png"
        else:
            save_path = f"results/gradcam/{patientID}_{model_type}_base{dim}_RAW_gradcam_condition_{i}_{condition}.png"
        Image.fromarray(gradcam_image).save(save_path)
        print(f"Individual gradcam images saved to {save_path}")

#########################

##### Grad-CAM PCA analysis

#########################

def load_environment_forPCA(csv_path, model_type, model_path, device, dim, enhanced, pca_sample_size, **kwargs):
    ### Load the csv of results
    df = load_csv(csv_path, pca_sample_size)
    n_instances = len(df)

    ### Load the specified model
    model = load_model(model_type, model_path, device)

    ### Set the target layer
    target_layer = set_target_layer(model, model_type)

    return (df, n_instances, target_layer, model)

def get_gradcam_masks_forPCA(model, target_layer, input_tensor, predicted_label, cond_idx, device):
    
    target = ClassifierOutputTarget(predicted_label)  # Use predicted class for GradCAM

    wrapped_model = ModelWrapper(model, condition_index=cond_idx).to(device)
    cam = GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer])

    # Compute Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())  # Normalize

    return grayscale_cam

def gradcam_pca_analysis(model_type, dim, enhanced, n_pca_components, pca_sample_size, csv_path=None, model_path=None, seed=None, scale_min=0, scale_max=255, usm_sigma=10, usm_weight=1.2, he_sigma=5, **kwargs):
    ### Set the image path column
    image_path_column = "source_file_path"
    
    ### If not specified, build the csv and model file paths
    if not csv_path:
        if enhanced:
            csv_path = f"results/{model_type}_BASE{dim}_ENHANCED_test_withPredictionsAndConfidences.csv"
        else:
            csv_path = f"results/{model_type}_BASE{dim}_test_withPredictionsAndConfidences.csv"
    if not model_path:
        if enhanced:
            model_path = f"models/{model_type}_BASE{dim}_ENHANCED.pth" # << Change this to your actual model path
        else:
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
    df, n_instances, target_layer, model = load_environment_forPCA(csv_path, model_type, model_path, device, dim, enhanced, pca_sample_size,**kwargs)

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

    pipeline_kwargs = dict(
        scale_min=scale_min,
        scale_max=scale_max,
        usm_sigma=usm_sigma,
        usm_weight=usm_weight,
        he_sigma=he_sigma,
    )

    ### Instantiate the hilbert curve index
    if dim > 224:
        hilbert_dim = 512
    else:
        hilbert_dim = 256
    n_dim = 2
    n_bits = 8
    hilbert_idxs = get_hilbert_index(hilbert_dim, n_dim, n_bits)
    
    ### Get 
    
    ### Get a sample of n_instances row indexes from df
    # image_idxs = np.random.choice(df.index, pca_sample_size, replacement=False)
    img_idxs = df.index[:pca_sample_size]

    ### Instantiate matrices to hold data
    condition_img_arrays_mat = np.zeros((len(hilbert_idxs), pca_sample_size))
    model_gradcamPCA_byCondition = np.zeros((n_classes, n_pca_components))

    ### Open the images and record them as a matrix
    img_tensors = torch.zeros((pca_sample_size, 3, dim, dim))
    for i,img_idx in enumerate(img_idxs):
        input_tensor, _ = load_image(df, img_idx, image_path_column, dim, enhanced, device, **pipeline_kwargs)
        img_tensors[i,:,:,:] = input_tensor

    ## Loop through all df images and track aggregate stats based on pca transformations
    for counter,img_idx in enumerate(img_idxs, start=1):
        input_tensor = img_tensors[i,:,:,:].unsqueeze(0)
        target_labels = [int(df.loc[img_idx, condition]) for condition in conditions]
        predicted_labels = [int(df.loc[img_idx, f"{condition}_pred"]) for condition in conditions]
        gradcam_masks = get_gradcam_masks(model, target_layer, input_tensor, predicted_labels, device)
        
        gradcam_mat = np.zeros((n_classes, hilbert_dim**2))
        for cond_idx, gradcam_mask in enumerate(gradcam_masks):
            gradcam_mat[cond_idx, :] =  hilbert_ravel(gradcam_mask, hilbert_dim, hilbert_idxs)
        
        gradcam_vars = add_to_gradcam_vars(gradcam_vars, gradcam_mat, n_pca_components, counter, **kwargs)

    return gradcam_vars

def add_to_gradcam_vars(grad_vars, grad_mat, n_pca_components, counter, **kwargs):
    """
    Calculate the PCA transformation of the input gradcam masks
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