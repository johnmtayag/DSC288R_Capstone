import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime
import sys

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision.models import resnet50, densenet121, resnet101

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    from scripts.preprocessing import scale_range
except:
    # Replace S:/Chexpert with the correct root as needed
    sys.path.append("S:/Chexpert/scripts")
    from preprocessing import scale_range


##### Model classes

class Train_Image_Data(Dataset):
    def __init__(self, X, y, dim):
        self.X = X
        self.y = y
        self.rot90_k = np.random.choice(range(4), len(self.y), replace=True)
        self.random_float = np.random.randn(len(self.y), 2)
        self.dim = dim
        # self.center_crop_ind = np.random.rand(len(y))
        if len(self.X) != len(self.y):
            raise Exception("The length of X does not match the length of y")

    def random_preprocess(self, img, index):
        # Randomly rotate in a 90 degree interval
        # img = v2.functional.rotate(img, self.rotation90_angles[index], interpolation=v2.InterpolationMode.BILINEAR)
        img = torch.rot90(img, self.rot90_k[index], (1,2))
        # Randomly rotate less than 90 degrees
        img = v2.functional.rotate(img, self.random_float[index, 0] * 30, interpolation=v2.InterpolationMode.BILINEAR)
        # Random affine
        # img = v2.RandomAffine(degrees=0, translate=(0.0025, 0.025))(img)
        # Center crop and resize
        if np.abs(self.random_float[index, 1]) > 1.5:
            img = v2.CenterCrop(self.dim - 48)(img)
            img = v2.Resize((self.dim,self.dim), interpolation=v2.InterpolationMode.BILINEAR)(img)
        elif np.abs(self.random_float[index, 1]) > 1:
            img = v2.CenterCrop(self.dim - 32)(img)
            img = v2.Resize((self.dim,self.dim), interpolation=v2.InterpolationMode.BILINEAR)(img)
        elif np.abs(self.random_float[index, 1]) > 0:
            img = v2.CenterCrop(self.dim - 16)(img)
            img = v2.Resize((self.dim,self.dim), interpolation=v2.InterpolationMode.BILINEAR)(img)
        return img
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Output X values as a tensor of img values (n_rows, 3, H, W)
        # Output y values as an array of class values
        path = self.X[index]
        # bag_train_i = self.bag_train[index]
        img = decode_image(path).float()
        img = self.random_preprocess(img, index)
        img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, self.y[index,:].astype(np.float32), index

# --- Validation Loader
class Validate_Image_Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if len(self.X) != len(self.y):
            raise Exception("The length of X does not match the length of Y")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        # Output X values as a tensor of img values (n_rows, 3, H, W)
        # Output y values as an array of class values
        path = self.X[index]
        img = decode_image(path).float()
        img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, self.y[index,:].astype(np.float32), index

# --- Custom Resnet50 with 14 separate prediction layers
class Custom_ResNet50(torch.nn.Module):
    def __init__(self, n_classes):
        super(Custom_ResNet50, self).__init__()
        self.resnet_model = resnet50(weights="ResNet50_Weights.DEFAULT")
        self.num_fc_in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = torch.nn.Identity()
        #
        self.n_classes = n_classes
        # Set the final layer as 14 separate layers with 3 outputs each
        # These will individually predict 0/1/2 for a specific class
        # Each is fully connected to the previous layer
        # This setup allows class-based weighting instead of only overall weights
        self.fc = torch.nn.ModuleList([
                torch.nn.Linear(in_features=self.num_fc_in_features,
                                out_features=3,
                                bias=True
                                )
             for i in range(self.n_classes)]
            )
        
    def forward(self, x):
        x = self.resnet_model(x)
        # return a list of outputs, each containing one output from one classification layer
        outputs = [final_layer(x).to(torch.float64) for final_layer in self.fc]
        return outputs

class Custom_DenseNet121(torch.nn.Module):
    def __init__(self, n_classes):
        super(Custom_DenseNet121, self).__init__()
        self.densenet_model = densenet121(weights="DenseNet121_Weights.DEFAULT")
        self.num_fc_in_features = self.densenet_model.classifier.in_features
        self.densenet_model.classifier = torch.nn.Identity()
        #
        self.n_classes = n_classes
        # Set the final layer as 14 separate layers with 3 outputs each
        # These will individually predict 0/1/2 for a specific class
        # Each is fully connected to the previous layer
        # This setup allows class-based weighting instead of only overall weights
        self.classifier = torch.nn.ModuleList([
                            torch.nn.Linear(in_features=self.num_fc_in_features,
                                            out_features=3,
                                            bias=True
                                            )
                         for i in range(self.n_classes)]
            )
        
    def forward(self, x):
        x = self.densenet_model(x)
        # return a list of outputs, each containing one output from one classification layer
        outputs = [final_layer(x).to(torch.float64) for final_layer in self.classifier]
        return outputs

# --- Model wrapper to restrict outputs to a single condition for grad-CAM
class ModelWrapper(torch.nn.Module):
    """ Wraps a multi-condition model to return output for a single condition. """
    def __init__(self, model, condition_index=0):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.condition_index = condition_index

    def forward(self, x):
        output = self.model(x) 
        output = torch.stack(output, dim=0) # Expected shape: (B, num_conditions, num_classes)
        return output[self.condition_index, :, :]  # Extract output for one condition

##### --- Load the model (used for both training and validation)  
def create_model(device, model_type, n_classes, path=None):
    if model_type == "ResNet50":
        # Initialize the pretrained resnet model
        model = Custom_ResNet50(n_classes)
    elif model_type == "DenseNet121":
        # Initialize the pretrained densenet model
        model = Custom_DenseNet121(n_classes)

    if path:
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
    if device=="cuda":
        model = model.to(device)
        
    return model

# --- Custom loss function
class Custom_CrossEntropyLoss(torch.nn.Module):
    """
    Custom cross entropy loss to weight the losses by class
    """
    def __init__(self, weights=None):
        super(Custom_CrossEntropyLoss, self).__init__()
        if weights is not None:
            self.inner_criterion = torch.nn.CrossEntropyLoss(weights)
        else:
            self.inner_criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, predictions, actual, class_weight):
        
        loss = self.inner_criterion(predictions.double(), actual)
        # Add additional weighted mean loss for incorrect predictions for incorrect non-zero values
        loss += (loss * ((actual != 0).float()).mean()) * class_weight

        return loss

###### Model functions #########################################################################################################
def get_criterion(weights, use_weights):
    # Apply each class weight to its corresponding 
    criterion = []
    if use_weights:
        for weights_i in weights:
            criterion.append(torch.nn.CrossEntropyLoss(weights_i))
    else:
        for weights_i in weights:
            criterion.append(torch.nn.CrossEntropyLoss()) 
    # if use_weights:
    #     for weights_i in weights:
    #         criterion.append(Custom_CrossEntropyLoss(weights_i))
    # else:
    #     for weights_i in weights:
    #         criterion.append(Custom_CrossEntropyLoss()) 
                
    return criterion

def get_loss(criterion, pred, actual, classWeights_df):
    # Compute and print loss
    for ind,(output, criterion_i, class_weight) in enumerate(zip(pred, criterion, classWeights_df)):
        if ind==0:
            loss = criterion_i(output, actual[:,ind])# * class_weight
        else:
            loss += criterion_i(output, actual[:,ind])# * class_weight
    
    return loss
        
# --- Fit the model to some training data
def fit(device, model, train_dataloader, valid_dataloader, weights, min_epochs, max_epochs, classWeights_df, unfreeze_all_layers_at_epoch, update_learning_rate_at_epoch, freeze_all_but_final_layer, use_weights, ep, n_classes, save_file_name, lr1, lr2, **kwargs):
    # --- Run the model
    print(f"----- Training model...\n{datetime.datetime.now()}")
        
    # Set initial optimizer and loss functions
    criterion = get_criterion(weights, use_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
    
    t_start = time.time()
    model_losses = {}
    last_avg_valid_loss = torch.inf
    
    for epoch in range(max_epochs):
        # Update the learning rate at 8 epochs
        if epoch + 1 == update_learning_rate_at_epoch:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr2)
        if epoch + 1 == unfreeze_all_layers_at_epoch:
            for param in model.parameters():
                param.requires_grad == True
            # criterion = get_criterion(weights, use_weights)
        t0 = time.time()
        
        # Instantiate the loss sum
        loss_sum = torch.zeros(1)
        if device=="cuda":
            loss_sum = loss_sum.to(device)

        # Start looping through the dataloader
        n_batches = len(train_dataloader)
        print(f"Iterating over {n_batches} batches...")
        for i,(X_batch, y_batch, _) in enumerate(train_dataloader, start=1):
            if not i%((len(train_dataloader)//5)):
                print(f"Batch {i}/{n_batches}: Training...")
            if device=="cuda":
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).long()
                
            # Forward pass: Compute predicted y_batch by passing X_batch to the model
            y_pred = model(X_batch)
            
            # Architecture-specific actions to compute loss
            loss = get_loss(criterion, y_pred, y_batch, classWeights_df)
                
            # Add loss the the sum
            loss_sum += loss
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = loss_sum / n_batches
        torch.save(model.state_dict(), f"models/{save_file_name}_{epoch}.pth")
        
        # avg_valid_loss = validate(device, model, valid_dataloader, n_classes, criterion, classWeights_df)
        if (epoch + 1 >= min_epochs):
            avg_valid_loss = validate(device, model, valid_dataloader, n_classes, criterion, classWeights_df)

            if last_avg_valid_loss - avg_valid_loss < ep:
                print(f"Early Stop --> Difference in validation loss = {torch.abs(last_avg_valid_loss - avg_valid_loss)}")
                print(f"\tTraining complete after {epoch + 1} epochs")
                print(f"\tAvg Training Loss: {avg_train_loss.item()}")
                print(f"\tAvg Validation Loss: {avg_valid_loss.item()}")
                model_losses[epoch] = (avg_train_loss.item(), avg_valid_loss.item())
                break
            else:
                last_avg_valid_loss = avg_valid_loss.clone()
                print(f"{datetime.datetime.now()}:\tEpoch {epoch + 1} ---> Time elapsed since last update: {time.time() - t0}")
                print(f"\tAvg Training Loss: {avg_train_loss.item()}")
                print(f"\tAvg Validation Loss: {avg_valid_loss.item()}")
                print()
            model_losses[epoch] = (avg_train_loss.item(), avg_valid_loss.item())
        else:
            print(f"{datetime.datetime.now()}:\tEpoch {epoch + 1} ---> Time elapsed since last update: {time.time() - t0}")
            print(f"\tAvg Training Loss: {avg_train_loss.item()}")
            # print(f"\tAvg Validation Loss: {avg_valid_loss.item()}")
            print()
            model_losses[epoch] = (avg_train_loss.item(), None)
            
    t_model = time.time() - t_start
    print(f"----- Model complete\nTraining Time: {t_model}")
            
    return model, t_model, model_losses
    
# --- Getting weights for the loss functions
def get_loss_weights(device, class_value_counts, n_instances, n_classes):
    """
    Parameters
    ----------
    class_value_counts : TYPE
        An array containing the counts for each value, for each class
    n_instances : TYPE
        The total count of instances being inputted into the model

    Returns
    -------
    weights : TYPE
        A set of weights for the loss criterion
        If using multioutput architecture, a set of weights are outputted for each class

    """
    # If any count < 0, add the minimum + 1 to all counts to prevent value errors
    if class_value_counts.min() < 0:
        class_value_counts -= class_value_counts.min()
    # Add 1 to each count to avoid zero
    class_value_counts += np.ones(class_value_counts.shape)
    
    # instantiate the output list
    weights_output = []

    # Loop through class columns and append weight vectors to weights_output
    for i in range(n_classes):
        weights_vec = (np.ones(3) * n_instances) / (3 * class_value_counts[:,i])
        # Make weights sum to one
        weights_vec = weights_vec / weights_vec.sum()
        weights_output.append(weights_vec)
        
    if device=="cuda":
        if isinstance(weights_output, list):
            for i in np.arange(len(weights_output)):
                weights_output[i] = torch.from_numpy(weights_output[i]).to(device)
        else:
            weights_output = torch.from_numpy(weights_output).to(device)
    else:
        weights_output = torch.from_numpy(weights_output)
            
    return weights_output

##### VALIDATION FUNCTIONS
# --- Validate within an epoch for early stopping
def validate(device, model, dataloader, n_classes, criterion, classWeights_df, **kwargs):
    print("Validating...")
    model.eval()
    valid_loss_sum = 0

    # Start looping through the dataloader
    n_batches = len(dataloader)
    with torch.no_grad():
        for i,(X_batch, y_batch, _) in enumerate(dataloader, start=1):
            if not (i % (n_batches // 5)):
                print(f"Batch {i}/{n_batches}: Validating...")
            if device=="cuda":
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).long()
                    
            # Forward pass: Compute predicted y_batch by passing X_batch to the model
            y_pred = model(X_batch)
            
            # Architecture-specific actions to compute loss
            loss = get_loss(criterion, y_pred, y_batch, classWeights_df)
                
            # Add loss the the sum
            valid_loss_sum += loss
                
        avg_valid_loss = valid_loss_sum / n_batches
    
    return avg_valid_loss
    
# --- Get predictions on the validation sets
def predict(device, model, dataloader, n_classes, **kwargs):
    predictions = []
    actual = []
    indexes = []
    confidences = []
    
    # Start looping through the dataloader
    print(f"Iterating over {len(dataloader)} batches...")
    for i,(X_batch, y_batch, index_batch) in enumerate(dataloader, start=1):
        n_batches = len(dataloader)
        batch_size_i = len(y_batch)
        if not (i % (n_batches // 5)):
            print(f"Batch {i}/{n_batches}: Validating...")
            
        with torch.no_grad():
            # Load to GPU
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(X_batch)

            # Convert y_pred via Softmax and Argmax
            preds = torch.zeros((batch_size_i, n_classes))
            confs = torch.zeros((batch_size_i, n_classes))
            if device=="cuda":
                preds = preds.to(device)
                confs = confs.to(device)
            for ind,output in enumerate(y_pred):
                output = torch.softmax(output, dim=1)
                preds[:,ind] = torch.argmax(output, dim=1)
                confs[:,ind] = torch.max(output, dim=1).values
            predictions.extend([preds[ind,:] for ind in range(batch_size_i)])
            actual.extend(y_batch)
            indexes.extend(index_batch)
            confidences.extend([confs[ind,:] for ind in range(batch_size_i)])
    return (actual, predictions, indexes, confidences)

# --- Get overall accuracy and class-based accuracy values
def get_stats(pred, actual, class_cols):
    results_all = []
    results_byClass = torch.zeros(len(class_cols))
    for i,(a,p) in enumerate(zip(actual,pred)):
        results = (a==p).to("cpu")
        results_all.extend(results)
        results_byClass += results
    
    results_all = torch.Tensor(results_all)
    acc = (torch.sum(results_all) / len(results_all)).item()
    acc_byClass = (results_byClass / len(actual))

    return (acc, acc_byClass)
            
def plot_confusion_matrices(confmat_fig_name, save_file_name, acc_byClass, actual_arr, predictions_arr, class_cols):
    # --- Create plots of class-wise confusion matrices
    fig,ax = plt.subplots(3,5, figsize=(16,8))
    fig.suptitle(save_file_name)
    
    conf_mats = {}
    # Overall confusion matrices
    conf_mats["Overall"] = {}
    for i,col,acc_byClass_i in zip(range(len(class_cols)),class_cols,acc_byClass):
        conf_mat = confusion_matrix(actual_arr[:,i], predictions_arr[:,i])
        conf_mats["Overall"][col] = conf_mat
        disp = ConfusionMatrixDisplay(conf_mat)
        disp.plot(ax=ax[0,i])
        ax[0,i].set_title(f"{col}:\n{round(acc_byClass_i.item(), 3)}")

    # Only 0 and 1
    conf_mats["No_Uncertain"] = {}
    for j,col in enumerate(class_cols):
        class_actual, class_predictions = [], []
        acc = []
        for i in range(len(actual_arr)):
            if (actual_arr[i,j] == 2) or (predictions_arr[i,j] == 2):
                continue
            else:
                class_actual.append(actual_arr[i,j])
                class_predictions.append(predictions_arr[i,j])
                if actual_arr[i,j]==predictions_arr[i,j]:
                    acc.append(1)
        class_actual = torch.tensor(class_actual)
        class_predictions = torch.tensor(class_predictions)
        acc2 = len(acc) / len(class_actual)
        #
        conf_mat = confusion_matrix(class_actual, class_predictions)
        conf_mats["No_Uncertain"][col] = conf_mat
        disp = ConfusionMatrixDisplay(conf_mat)
        disp.plot(ax=ax[1,j])
        ax[1,j].set_title(f"Only Certain Classifications:\n{round(acc2, 3)}")

    # Only 0 and 1
    conf_mats["Certain_vs_Uncertain"] = {}
    for j,col in enumerate(class_cols):
        class_actual, class_predictions = actual_arr[:,j].copy(), predictions_arr[:,j].copy()
        class_actual[class_actual != 2] = 0
        class_predictions[class_predictions != 2] = 0
        acc = []
        for i in range(len(actual_arr)):
            if class_actual[i]==class_predictions[i]:
                acc.append(1)
        class_actual = torch.tensor(class_actual)
        class_predictions = torch.tensor(class_predictions)
        acc2 = len(acc) / len(class_actual)
        #
        conf_mat = confusion_matrix(class_actual, class_predictions)
        conf_mats["Certain_vs_Uncertain"][col] = conf_mat
        disp = ConfusionMatrixDisplay(conf_mat)
        disp.plot(ax=ax[2,j])
        ax[2,j].set_title(f"Certain vs Uncertain:\n{round(acc2, 3)}")

    plt.tight_layout()
    plt.show()
    fig.savefig(confmat_fig_name)
    
    return conf_mats

def get_predictions(device, model_path, df_path, path_col, predict_type, batch_size, sample_frac, confmat_fig_name, class_cols, save_file_name, model_type, **model_params):
    # --- Load the data, then reduce to the first (sample_frac * len(df)) instances
    df = pd.read_csv(df_path)
    df = df[:int(round(sample_frac * len(df)))]
    X = list(df[path_col])
    y = df[class_cols].to_numpy()
    n_instances = len(y)
    n_classes = len(class_cols)
    #
    actual = torch.Tensor(y)
    if device=="cuda":
        actual = actual.to(device)
    #
    print()
    print(f"--- {predict_type}\n{datetime.datetime.now()}")
    print(f"Number of {predict_type} samples loaded: {len(df)}")
    print()
    
    # --- Instantiate the dataloader
    dataloader = DataLoader(Validate_Image_Data(X,y), 
                            batch_size=batch_size, 
                            shuffle=False)
    
    # --- Instantiate the base model
    print(f"Using model saved at {model_path}")
    print()
    
    # --- Get predictions from the ensemble model
    predictions, confidences = torch.zeros((n_instances, n_classes)), torch.zeros((n_instances, n_classes))
    if device=="cuda":
        predictions = predictions.to(device)
        confidences = confidences.to(device)
        
    # --- Load the current model and obtain prediction-actual pairings
    model = create_model(device, model_type, n_classes, model_path)
    actual_i, predictions_i, indexes_i, confidences_i  = predict(device, model, dataloader, n_classes)
    
    # --- Append the score in the prediction dictionary
    for j,(a,p,c,ind) in enumerate(zip(actual_i, predictions_i, confidences_i, indexes_i)):
        predictions[ind.item()] = p
        confidences[ind.item()] = c
            
    # Get stats
    actual_arr = np.array([a.cpu() for a in actual])
    predictions_arr = np.array([p.cpu() for p in predictions])
    confidence_arr = np.array([c.cpu() for c in confidences])
    
    acc, acc_byClass = get_stats(predictions, actual, class_cols)
    
    # Generate confusion matrices
    conf_mats = plot_confusion_matrices(confmat_fig_name,
                                        save_file_name,
                                        acc_byClass,
                                        actual_arr,
                                        predictions_arr,
                                        class_cols)
    
    # Append the predictions and confidences to df
    predictions = pd.DataFrame({col+"_pred":predictions_arr[:,i] for i,col in enumerate(class_cols)})
    confidences = pd.DataFrame({col+"_confidence":confidence_arr[:,i] for i,col in enumerate(class_cols)})
    df = pd.concat([df, predictions], axis=1)
    df = pd.concat([df, confidences], axis=1)
    
    outputs = {"actual_arr":actual_arr,
               "predictions_arr":predictions_arr,
               "confidence_arr":confidence_arr,
               "acc":acc,
               "acc_byClass":acc_byClass,
               "df":df,
               "conf_mats":conf_mats}
    
    return outputs

def print_accuracies(outputs, class_cols):
    print()
    print("Overall Accuracy =", outputs["acc"])
    print("---")
    for label,label_acc in zip(class_cols, outputs["acc_byClass"]):
        print(f"{label_acc} --> Label: {label}")



