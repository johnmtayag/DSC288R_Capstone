
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import datetime

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision.models import resnet50

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

################################################################################

### Set initial variables

################################################################################

# --- Use the GPU
use_cuda = True

# --- Use enhanced training images (additional transforms applied)
# enhanced_images = True
enhanced_images_list = [False, True]
# enhanced_images_list = [True]

# --- Select the image resolution value: 224, 384, or 512
# dim = 224
dim_list = [224, 224]
# dim_list = [224]

n_samples_list = [150000, 150000]

# --- Save file name
# save_file_name = f"ResNet50_BASE{dim}_fullDataWithOversampling" 
# save_file_name = f"ResNet50_BASE{dim}"
# if enhanced_images: 
#     save_file_name = save_file_name + "_ENHANCED"

##### Set the class columns
class_cols = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices', 'No Finding']

# class_col_index = [2, 4, 5, 7, 8, 9, 12, 13]
class_col_index = [i for i in range(len(class_cols))]
class_cols = [class_cols[i] for i in class_col_index]
n_classes = len(class_cols)

# ##### Set the path column based on dim and enhanced_images selection
# path_col = f"base{dim}_file_path"
# if enhanced_images:
#     path_col += "2"
    
# ##### MAIN CONTROLS ############################################################
# # --- Train the model
# train_model = True
# #
# train_batch_size = 96
# train_sample_frac = 1

# model_filepath = f"models/{save_file_name}_filenames.txt"

# # --- Predict on the validation set
# validate_model = False
# valid_model_paths = f"models/{save_file_name}_filenames.txt"
# save_valid_model_path = f"models/{save_file_name}_valid_filenames.txt"
# #
# valid_batch_size = 64
# valid_sample_frac = 1

# # --- Predict on the test set
# test_model = False
# test_model_paths = f"models/{save_file_name}_filenames.txt"
# save_test_model_path = f"models/{save_file_name}_test_filenames.txt"
# #
# test_batch_size = 64
# test_sample_frac = 1

# # --- Predict on the validation rad set
# validate_rad_model = False
# valid_rad_model_paths = f"models/{save_file_name}_filenames.txt"
# save_valid_rad_model_path = f"models/{save_file_name}_validrad_filenames.txt"
# #
# valid_rad_batch_size = 64
# valid_rad_sample_frac = 1

# # --- Predict on the test rad set
# test_rad_model = False
# test_rad_model_paths = f"models/{save_file_name}_filenames.txt"
# save_test_rad_model_path = f"models/{save_file_name}_testrad_filenames.txt"
# #
# test_rad_batch_size = 64
# test_rad_sample_frac = 1

# --- Extract a subset from the training set to use for validation
validate_from_training_set = False

# --- Resample the dataset: Use one of these if not using an already-sampled version
use_CustomSampler = False ### 
use_RandomResample = False ### 
#
with_replacement = True
n_samples = 30000

# --- Apply weights based on class values (0, 1, 2)
use_weights = True
# Apply different weights based on individual class distributions
weights_byClass = True

# --- Use the custom loss function to add additional losses based on incorrect predictions for (1) and/or (2)
use_customLoss = False

# --- Set the number of epochs
n_epochs = 8
n_models = 1


##### File variables ###########################################################
root = "S:/CheXpert/"

# --- Training model architecture variables
### RESNET
model_type = "resnet"
n_model_layers = 50
#
n_workers = 4

##### Dataset/Model params #####################################################
# --- Image enhancement variables (Not used)
he_sigma = 5
usm_sigma = 10
usm_weight = 1.2
scale_min = 0
scale_max = 255

# --- Class weights
class_probs = [0.06608382755578038,
                0.041322407816882005,
                0.01359448345359668,
                0.15074868241942074,
                0.020497934798456904,
                0.035359570514560616,
                0.061903826415203915,
                0.021455397623643135,
                0.06530314033111131,
                0.014286771552752263,
                0.2499316321871095,
                0.1669507903403583,
                0.011981385757781403,
                0.08058014923334274]

class_probs = [class_probs[i] for i in class_col_index]
if len(class_probs) < 14:
    class_probs = np.array(class_probs)
    class_probs = class_probs / class_probs.sum()

class_value_probs = [0.1910498186148265,
                     0.8089501813851734] 

##### Training variables #######################################################
train_root = f"{root}train/"
train_filepath = f"{root}train_df_30000.csv"
train_filepaths = [f"{root}train_df_oversampled.csv",
                   f"{root}train_df_oversampled.csv"]

##### Validation variables #####################################################
valid_root = f"{root}valid/"
valid_filepath = f"{root}valid_df.csv"

##### Test variables #####################################################
test_root = f"{root}test/"
test_filepath = f"{root}test_df.csv"

##### Validation rad variables #####################################################
valid_rad_root = f"{root}valid_rad/"
valid_rad_filepath = f"{root}valid_rad_df.csv"

##### Validation variables #####################################################
test_rad_root = f"{root}test_rad/"
test_rad_filepath = f"{root}test_rad_df.csv"

################################################################################

### Defining classses and functions

################################################################################
# %%

##### --- Dataset Loaders and preprocessing steps
# - Get training data loader
class Train_Image_Data(Dataset):
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
        # bag_train_i = self.bag_train[index]
        img = decode_image(path).float()
        img = random_preprocess(img)
        img = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, self.y[index,:].astype(np.float32), index
# --- Get validation loader 
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
# --- Custom sampler to oversample images with underrepresented values
class CustomSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, df, batch_size, n_samples, class_cols, class_probs, class_value_probs):
        self.df = df
        self.batch_size = batch_size
        self.n_samples = n_samples
        #
        # (1 - Probability) of each class being 1/2 in the dataset, total adds to 1
        # This gives the probability of a class being sampled with a higher chance of
        # underrepresented columns being selected
        #
        # These are calculated in the preprocessing notebook
        self.class_probs = class_probs
        #
        # (1 - Probability) of each class value (1 or 2) occuring in the dataset
        # across all classes
        # This gives the probability of the selected sample in a class being 1 or 2
        #
        # These are calculated in the preprocessing notebook
        self.class_value_probs = class_value_probs
        self.class_cols = class_cols
        self.n_classes = len(self.class_cols)
        
    def __iter__(self):
        # print("Iter called")
        batch = []
        # For each sample index, get the column and value to create a df subset
        col_sample = np.random.choice(np.arange(self.n_classes),
                                      size=self.n_samples,
                                      replace=True,
                                      p=self.class_probs)
        # Probabilities here are the (1-freq) of each class value occuring in train_df
        val_sample = np.random.choice([1,2],
                                      size=self.n_samples,
                                      replace=True,
                                      p=self.class_value_probs)

        # Get the indexes
        for i,(col_samp, val_samp) in enumerate(zip(col_sample, val_sample)):
            # Cover edge case of no uncertain "No Finding"
            if col_samp == self.n_classes - 1:
                val_samp = 1
            # Retrieve a random instance where the specified column has the specified value
            subset_df = self.df[self.class_cols[col_samp]]
            subset_df = subset_df[subset_df == val_samp]
            
            # # If no samples exist for the given value, add a random index
            # if len(subset_df) == 0:
            #     batch.append(self.df.sample(1).index[0])
            
            if len(subset_df) == 1:
                batch.append(subset_df.index[0])
            else:
                batch.append(subset_df.sample(1).index[0])
                    
            # Yield from batch if enough have been accumulated
            if len(batch) == self.batch_size:
                # print(i, "yielding")
                yield from batch 
                batch = []
        # if batch:
        #     yield batch
                
    def __len__(self):
        return self.n_samples

# --- Random preprocessing steps applied to bagged examples
random_preprocess = v2.Compose([
    v2.RandomRotation((-10,10), interpolation=v2.InterpolationMode.BILINEAR),
    ])
# --- Pipeline if performing all image transformations
class image_enhancements(torch.nn.Module):
    """
    Apply unsharp masking to sharpen the input image
    Then apply histogram equalization
    """
    def __init__(self, he_sigma, usm_sigma, usm_weight):
        super().__init__()
        self.he_sigma = he_sigma
        self.usm_sigma = usm_sigma
        self.usm_weight = usm_weight
        
    def forward(self, sample):
        sample = v2.ToImage()(sample).numpy()
        # sample = unsharp_masking(sample, self.usm_sigma, self.usm_weight, scale_min, scale_max)
        # sample = histogram_equalization(sample, scale_min, scale_max, self.he_sigma)
        sample = torch.from_numpy(sample).float()
        return sample

##### Training classes and functions
# --- New resnet class creation with 14 separate final classification layers
class Custom_ResNet_SeparateOutputLayers(torch.nn.Module):
    def __init__(self, resnet_model, n_features, n_classes):
        super(Custom_ResNet_SeparateOutputLayers, self).__init__()
        self.resnet_model = resnet_model
        self.n_features = n_features
        self.n_classes = n_classes
        # Set the final layer as 14 separate layers with 3 outputs each
        # These will individually predict 0/1/2 for a specific class
        # Each is fully connected to the previous layer
        # This setup allows class-based weighting instead of only overall weights

        self.fc = torch.nn.ModuleList([
                torch.nn.Linear(in_features=self.n_features,
                                out_features=3,
                                bias=True
                                )
             for i in range(self.n_classes)]
            )
    def forward(self, x):
        x = self.resnet_model(x)
        # return a list of outputs, each containing one output from one classification layer
        # x = self.fc(x)
        outputs = [final_layer(x).to(torch.float64) for final_layer in self.fc]
        return outputs
    
# --- Custom loss function
class Custom_CrossEntropyLoss(torch.nn.Module):
    """
    Custom cross entropy loss to manually increase loss for misclassifications for class values (1) and (2)
    """
    def __init__(self, weights):
        super(Custom_CrossEntropyLoss, self).__init__()
        self.weights = weights
        
    def forward(self, predictions, actual, class_value_probs):
        # actual = torch.LongTensor(actual) ### This is done in the main fit loop
        inner_criterion = torch.nn.CrossEntropyLoss(self.weights)
        # print(predictions.dtype, actual.dtype)
        loss = inner_criterion(predictions.double(), actual)
        # Add the mean loss for incorrect predictions for nonzero values
        loss += (loss * ((actual != 0).float()).mean())

        return loss
    
# --- Model functions
def get_criterion(weights=None):
    if use_weights:
        criterion = []
        for weights_i in weights:
            if use_customLoss:
                criterion.append(Custom_CrossEntropyLoss(weights_i))
            else:
                criterion.append(torch.nn.CrossEntropyLoss(weights_i))
    else:
        if use_customLoss:
            criterion = [Custom_CrossEntropyLoss() for i in range(n_classes)]
        else:
            criterion = [torch.nn.CrossEntropyLoss() for i in range(n_classes)]
                
    return criterion
def get_loss(criterion, pred, actual):
    # Compute and print loss
    for ind,(output, criterion_i, class_prob) in enumerate(zip(pred, criterion, class_probs)):
        # print(output.dtype, actual.dtype)
        # output = output.to(torch.float64)
        # print(output.dtype, actual.dtype)
        if use_customLoss:
            if ind==0:
                loss = criterion_i(output, actual[:,ind], class_value_probs) * class_prob
            else:
                loss += criterion_i(output, actual[:,ind], class_value_probs) * class_prob
        else:
            if ind==0:
                loss = criterion_i(output, actual[:,ind]) * class_prob
            else:
                loss += criterion_i(output, actual[:,ind]) * class_prob
        
    return loss
        
# --- Fit the model to some training data
def fit(model, train_dataloader, weights=None):
    # --- Run the model
    print(f"----- Training model...\n{datetime.datetime.now()}")
        
    # Set initial optimizer and loss functions
    criterion = get_criterion(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    t_start = time.time()
    model_losses = np.zeros(n_epochs)
    
    for epoch in range(n_epochs):
        # Update the learning rate halfway through training
        if epoch == n_epochs//2:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
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
            loss = get_loss(criterion, y_pred, y_batch)
                
            # Add loss the the sum
            loss_sum += loss
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = loss_sum / n_batches
        model_losses[epoch] = avg_loss.item()
        
        print(f"{datetime.datetime.now()}:\tEpoch {epoch + 1} ---> Time elapsed since last update: {time.time() - t0}")
        print(f"\tAvg Loss: {avg_loss.item()}")
        print()
            
    t_model = time.time() - t_start
    print(f"----- Model complete\nTraining Time: {t_model}")
            
    return model, t_model, model_losses
# --- Getting weights for the loss functions
def get_loss_weights(class_value_counts, n_instances):
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
    # Add 1 to each count
    class_value_counts += np.ones(class_value_counts.shape)
    
    # Get the weights for each class as a list
    weights_output = []

    # Get each count as the fraction of all values
    weights = class_value_counts / n_instances
    # Nonlinearly transform the weights
    #    1 / weights --> Flips the representation
    #    sqrt(weights) --> Squeezes the value range non-linearly toward zero
    weights = (1 / weights)
    # weights = np.log(weights + 0.25)
    # Make the sum of each class's weights add up to 1
    if weights_byClass:
        # Ensure rowsums for each column add up to one
        weights = weights / weights.sum(axis=0)
        # Split into a list of weights by class
        weights_output = [weights[:,i] for i in range(n_classes)]
    else:
        # Compress to total class value sums (0/1/2)
        weights = weights.sum(axis=1)
        # Ensure the sum of weights adds up to one
        weights = weights / weights.sum(axis=0)

        # Split into a list of weights
        weights_output = [weights for i in range(n_classes)]
        
    if device=="cuda":
        if isinstance(weights_output, list):
            for i in np.arange(len(weights_output)):
                weights_output[i] = torch.from_numpy(weights_output[i]).to(device)
        else:
            weights_output = torch.from_numpy(weights_output).to(device)
    else:
        weights_output = torch.from_numpy(weights_output)
            
    return weights_output

##### --- Load the model (used for both training and validation)  
def create_model(model_type, n_model_layers, n_classes, path=None):
    # Initialize the pretrained resnet model
    output_model = resnet50(weights=f"ResNet{n_model_layers}_Weights.DEFAULT")
    n_features = output_model.fc.in_features
    output_model.fc = torch.nn.Identity()
    model = Custom_ResNet_SeparateOutputLayers(output_model, n_features, n_classes)
    #
    if path:
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
    if device=="cuda":
        model = model.to(device)

        
    return model

##### VALIDATION FUNCTIONS
# --- Get predictions on the validation set
def predict(model, dataloader):
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
            pass
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
            
def plot_confusion_matrices(acc_byClass, actual_arr, predictions_arr):
    # --- Create plots of class-wise confusion matrices
    fig,axes = plt.subplots(3,5, figsize=(16,8))
    fig.suptitle(save_file_name)
    
    conf_mats = {}
    for i,col,ax,acc_byClass_i in zip(range(len(class_cols)),class_cols,axes.flatten(), acc_byClass):
        conf_mat = confusion_matrix(actual_arr[:,i], predictions_arr[:,i])
        conf_mats[col] = conf_mats
        disp = ConfusionMatrixDisplay(conf_mat)
        disp.plot(ax=ax)
        ax.set_title(f"{col}:\n{round(acc_byClass_i.item(), 3)}")
    
    axes[2,4].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"models/{save_file_name}.png")
    
    return conf_mats

def get_predictions(model_path, df_path, path_col, predict_type, batch_size, sample_frac):
    # --- Load the data, then reduce to the first (sample_frac * len(df)) instances
    df = pd.read_csv(df_path)
    df = df[:int(round(sample_frac * len(df)))]
    X = list(df[path_col])
    y = df[class_cols].to_numpy()
    n_instances = len(y)
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
    model = create_model(model_type, n_model_layers, n_classes, model_path)
    actual_i, predictions_i, indexes_i, confidences_i  = predict(model, dataloader)
    
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
    conf_mats = plot_confusion_matrices(acc_byClass,
                                        actual_arr,
                                        predictions_arr)
    
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

def print_accuracies(outputs):
    print()
    print("Overall Accuracy =", outputs["acc"])
    print("---")
    for label,label_acc in zip(class_cols, outputs["acc_byClass"]):
        print(f"{label_acc} --> Label: {label}")
        
# %%

################################################################################

### Running the model(s)

################################################################################

for dim,enhanced_images,train_filepath,n_samples_i in zip(dim_list, enhanced_images_list, train_filepaths, n_samples_list):
# for dim,enhanced_images in zip(dim_list, enhanced_images_list):
    save_file_name = f"ResNet50_BASE{dim}" 
    # save_file_name = f"ResNet50_BASE{dim}"
    if enhanced_images: 
        save_file_name = save_file_name + "_ENHANCED"
    save_file_name = save_file_name + f"_{n_samples_i}samples"
    model_filepath = f"models/{save_file_name}.pth"
        
    ##### Set the path column based on dim and enhanced_images selection
    path_col = f"base{dim}_file_path"
    if enhanced_images:
        path_col += "2"
        
    n_instances = n_samples
        
    ##### MAIN CONTROLS ############################################################
    # --- Train the model
    train_model = True
    #
    train_batch_size = 48
    train_sample_frac = 1


    # --- Predict on the validation set
    validate_model = True
    #
    valid_batch_size = 96
    valid_sample_frac = 1

    # --- Predict on the test set
    test_model = True
    #
    test_batch_size = 96
    test_sample_frac = 1

    # --- Predict on the validation rad set
    validate_rad_model = True
    #
    valid_rad_batch_size = 20
    valid_rad_sample_frac = 1

    # --- Predict on the test rad set
    test_rad_model = True
    #
    test_rad_batch_size = 50
    test_rad_sample_frac = 1
        
    ### --- Fit the model
    if __name__=="__main__":
        ### --- Checking for the GPU
        print("Is CUDA available:", torch.cuda.is_available())
        print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
        if use_cuda:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            device = "cpu"
        print(f"Device --> {device}")
        print()
        
        if train_model:
            t_models, model_losses = [], []
            print(f"\n----- Starting model {model_filepath}\n{datetime.datetime.now()}")
            train_df = pd.read_csv(train_filepath)
            if validate_from_training_set:
                cutoff_ind = int(round(valid_sample_frac * len(train_df)))
                valid_df = train_df.iloc[:cutoff_ind,:].copy().reset_index(drop=True)
                train_df = train_df.iloc[cutoff_ind:,:].reset_index(drop=True)
            train_df = train_df.sample(frac=train_sample_frac)
            X_train = list(train_df[path_col])
            y_train = train_df[class_cols].to_numpy()
            n_samples = len(X_train)

            print(f"\n# training samples: {n_samples}")
            if (use_RandomResample) or (use_RandomResample):
                print(f"# Samples with replacement: {n_samples}")
                
            ### --- Initializing model class --> Load the pre-trained model and replace the final class layer
            model = create_model(model_type, n_model_layers, n_classes)
            model_name = f"{model.__class__.__name__}{n_model_layers}" 
            print(f"Model name: {model_name}")
            
            ### --- Define the file name to save results
            if len(save_file_name)==0:
                save_file_name = f"Model_{datetime.datetime.now()}"
            print(f"Files will be saved with the name {save_file_name}")
            print()
            
            ### --- Load the data and fit the model
            train_data = Train_Image_Data(X_train, y_train)
            
            ### --- Instantiate the sampler (optional) and dataloader
            if use_RandomResample:
                sampler = RandomSampler(train_data, 
                                        replacement=with_replacement,
                                        num_samples=n_samples)
                train_dataloader = DataLoader(train_data, 
                                              batch_size=train_batch_size, 
                                              num_workers=n_workers,
                                              persistent_workers=True,
                                              sampler=sampler,
                                              drop_last=True
                                             )
                n_instances = n_samples
            elif use_CustomSampler:
                sampler = CustomSampler(train_df,
                                        train_batch_size,
                                        n_samples,
                                        class_cols,
                                        class_probs,
                                        class_value_probs)
                train_dataloader = DataLoader(train_data, 
                                              batch_size=train_batch_size,
                                              num_workers=n_workers,
                                              persistent_workers=True,
                                              sampler=sampler,
                                             )
                n_instances = n_samples
            else:
                sampler = None
                train_dataloader = DataLoader(train_data, 
                                              batch_size=train_batch_size, 
                                              shuffle=True,
                                              num_workers=n_workers,
                                              persistent_workers=True
                                             )
            print(f"Using sampler: {sampler}")
            print(f"Using dataloader: {train_dataloader}")
            print()
                
            ### Get loss weights (optional)
            if use_weights:
                if sampler:
                    # Weights must be calculated according to the samples
                    print("Getting weights for loss function")
                    ### Count the number of 0/1/2 values per class
                    class_value_counts = torch.zeros((3, n_classes))
                    for i,(_, y_batch_i, _) in enumerate(train_dataloader):
                        batch_sums = torch.stack([torch.sum(torch.where(y_batch_i==0, 1, 0), axis=0),
                                                  torch.sum(torch.where(y_batch_i==1, 1, 0), axis=0),
                                                  torch.sum(torch.where(y_batch_i==2, 1, 0), axis=0)], dim=0)
                        class_value_counts += batch_sums
                    class_value_counts = class_value_counts.numpy().astype(np.float32)
                    ### Use the value counts to extract weights
                    weights = get_loss_weights(class_value_counts, n_instances)
                else:
                    # Weights can be calculated directly from the df
                    print("Getting weights for loss function")
                    ### Count the number of 0/1/2 values per class
                    class_value_counts = np.zeros((3,n_classes))
                    for j,col in enumerate(class_cols):
                        counts = train_df[col].value_counts()
                        class_value_counts[0,j] = counts[0]
                        if 1 in counts.index:
                            class_value_counts[1,j] = counts[1]
                        if 2 in counts.index:
                            class_value_counts[2,j] = counts[2]
                    # class_value_counts = class_value_counts.astype(np.float32)
                    weights = get_loss_weights(class_value_counts, n_instances)

                
                ### Print information for the user
                if weights_byClass:
                    print("Value weights per class")
                    for i,col in enumerate(class_cols):
                        print(f"{col}:\n" + \
                              f"   0: {weights[i][0].item():.6f}" + \
                              f"   1: {weights[i][1].item():.6f}" + \
                              f"   2: {weights[i][2].item():.6f}")
                else:
                    print("Value weights per class")
                    for i,col in enumerate(class_cols):
                        print(f"{col}:\n" + \
                              f"   0: {weights[i][0].item():.6f}" + \
                              f"   1: {weights[i][1].item():.6f}" + \
                              f"   2: {weights[i][2].item():.6f}")
                print()
            else:
                weights = None
                
            ### Fit the model
            model, t_model, model_loss = fit(model, train_dataloader, weights)
        
            ### Save the results
            torch.save(model.state_dict(), model_filepath)
            print(f"Saved PyTorch Model State to {model_filepath}")
            
            t_models.append(t_model)
            model_losses.append(model_loss)
        # Save model data
        with open(f"models/{save_file_name}_times.txt", "w") as file:
            file.write(str(t_models))
        np.savetxt(f"models/{save_file_name}_losses.txt", model_losses)
            
        ###
        model_outputs = {}
        if validate_model:
            predict_type = "valid"
            outputs = get_predictions(model_filepath, 
                                      valid_filepath, 
                                      path_col, 
                                      predict_type, 
                                      valid_batch_size, 
                                      valid_sample_frac)
            print_accuracies(outputs)
            model_outputs[predict_type] = outputs
            outputs["df"].to_csv(f"models/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv")
    
        if test_model:
            predict_type = "test"
            outputs = get_predictions(model_filepath, 
                            test_filepath, 
                            path_col, 
                            predict_type, 
                            test_batch_size, 
                            test_sample_frac)
            print_accuracies(outputs)
            model_outputs[predict_type] = outputs
            outputs["df"].to_csv(f"models/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv")
            
        if validate_rad_model:
            predict_type = "valid_rad"
            outputs = get_predictions(model_filepath, 
                                      valid_rad_filepath, 
                                      path_col, 
                                      predict_type, 
                                      valid_rad_batch_size, 
                                      valid_rad_sample_frac)
            print_accuracies(outputs)
            model_outputs[predict_type] = outputs
            outputs["df"].to_csv(f"models/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv")
            
        if test_rad_model:
            predict_type = "test_rad"
            outputs = get_predictions(model_filepath, 
                                      test_rad_filepath, 
                                      path_col, 
                                      predict_type, 
                                      test_rad_batch_size, 
                                      test_rad_sample_frac)
            print_accuracies(outputs)
            model_outputs[predict_type] = outputs
            outputs["df"].to_csv(f"models/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv")
            
            
    
    
            