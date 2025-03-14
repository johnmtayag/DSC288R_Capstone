import pandas as pd
import numpy as np
import datetime 
import torch
import sys
import yaml
from torch.utils.data import DataLoader

##### Common actions
with open("config/common.yaml", "r") as params:
    common = yaml.safe_load(params)
    # unpack params
    conditions = common["conditions"]
    class_cols = common["class_cols"]
    n_classes = common["n_classes"]

################################################################################

### Set variables

################################################################################

# --- SET THE PATH TO THE SRC FOLDER
path = "S:/Chexpert/src"

# --- CHOOSE THE MODEL PARAMETERS (Options listed below)
"""
ResNet50_BASE224_RAW
ResNet50_BASE224_ENHANCED
ResNet50_BASE384_RAW
ResNet50_BASE384_ENHANCED
ResNet50_BASE512_RAW
ResNet50_BASE512_ENHANCED
DenseNet121_BASE224_RAW
DenseNet121_BASE224_ENHANCED
DenseNet121_BASE384_RAW
DenseNet121_BASE384_ENHANCED
DenseNet121_BASE512_RAW
DenseNet121_BASE512_ENHANCED
"""
model_params = "ResNet50_BASE224_RAW"

# --- CONFIRM IF IMAGES HAVE BEEN PREPROCESSED 
# If EnvironmentPrep_Testing.ipynb was run, set this to FALSE
# If EnvironmentPrep_TrainingAndTest.ipynb was run, set this to TRUE
using_preprocessed_images = True

##### MAIN CONTROLS ############################################################

# --- Set train_model to True to train a new model
train_model = False

# --- Set these to True to test the model on test_df, valid_rad_df, and/or test_df 
test_model = True
validate_rad_model = False
test_rad_model = False

# --- Model parameters (Ensure that the dataset file paths are correct)
"""
Batch size: Affects the number of images loaded for mini-batch SGD
Sample frac: Affects the total number of images loaded
"""
train_batch_size = 32
train_sample_frac = 1
train_filepath = "data/processed_data/train_df_oversampled.csv"
#
valid_batch_size = 64
valid_sample_frac = 1
valid_filepath = "data/processed_data/valid_df.csv"
#
test_batch_size = 64
test_sample_frac = 1
test_filepath = "data/processed_data/test_df.csv"
#
valid_rad_batch_size = 20
valid_rad_sample_frac = 1
valid_rad_filepath = "data/processed_data/valid_rad_df.csv"
#
test_rad_batch_size = 50
test_rad_sample_frac = 1
test_rad_filepath = "data/processed_data/test_rad_df.csv"

##### OTHER VARIABLES ##########################################################
# --- Use the GPU
use_cuda = True
# --- Set the number of workers for the data loader (Typically use the # of GPUs * 4)
n_workers = 4



################################################################################

### Model

################################################################################

# Load model scripts
try:
    import src.model as model_scripts
except:
    # Replace S:/Chexpert with the correct root as needed
    sys.path.append(path)
    import src.model as model_scripts
    
# Load preset model parameters
with open(f"config/{model_params}.yaml", "r") as params:
    model_params = yaml.safe_load(params)
    # Unpack params
    model_type = model_params["model_type"]
    dim = model_params["dim"]
    enhanced = model_params["enhanced"]
    #
    use_weights = model_params["use_weights"]
    freeze_all_but_final_layer=model_params["freeze_all_but_final_layer"]
    
# Set the class column index
class_col_index = [i for i in range(len(class_cols))]

# --- Wrap these into a dictionary for easier parameter entry
train_params = {"n_workers":n_workers,
                "min_epochs":model_params["min_epochs"],
                "max_epochs":model_params["max_epochs"],
                "freeze_all_but_final_layer":model_params["freeze_all_but_final_layer"],
                "unfreeze_all_layers_at_epoch":model_params["unfreeze_all_layers_at_epoch"],
                "update_learning_rate_at_epoch":model_params["update_learning_rate_at_epoch"],
                "use_weights":use_weights,
                "ep":model_params["ep"],
                "n_classes":n_classes,
                "lr1":model_params["lr1"],
                "lr2":model_params["lr2"],
                }

predict_params = {"class_cols":class_cols,
                  "min_epochs":model_params["min_epochs"],
                  "max_epochs":model_params["max_epochs"],
                  "freeze_all_but_final_layer":model_params["freeze_all_but_final_layer"],
                  "unfreeze_all_layers_at_epoch":model_params["unfreeze_all_layers_at_epoch"],
                  "update_learning_rate_at_epoch":model_params["update_learning_rate_at_epoch"],
                  "ep":model_params["ep"],
                  "n_classes":n_classes,
                  "using_preprocessed_images":using_preprocessed_images,
                  "dim":dim
                  }


################################################################################

### Running the model(s)

################################################################################
    
save_file_name = f"{model_type}_BASE{dim}" 
if enhanced: 
    save_file_name = save_file_name + "_ENHANCED"
model_filepath = f"models/{save_file_name}.pth"
    
##### Set the path column based on dim and enhanced_images selection
path_col = f"base{dim}_file_path"
if enhanced:
    path_col += "2"

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
        train_df = pd.read_csv(train_filepath).sample(frac=train_sample_frac)
        valid_df = pd.read_csv(valid_filepath)
        #
        X_train = list(train_df[path_col])
        y_train = train_df[class_cols].to_numpy()
        n_samples = len(X_train)
        #
        X_valid = list(valid_df[path_col])
        y_valid = valid_df[class_cols].to_numpy()
        n_valid_samples = len(X_valid)

        print(f"\n# training samples: {n_samples}")
        
        ### --- Initializing model class --> Load the pre-trained model and replace the final class layer
        if model_type == "ResNet50":
            model = model_scripts.create_model(device, model_type, n_classes)
            if train_model:
                if freeze_all_but_final_layer:
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.fc.parameters():
                        param.requires_grad = True
        else:
            model = model_scripts.create_model(device, model_type, n_classes)
            if train_model:
                if freeze_all_but_final_layer:
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.classifier.parameters():
                        param.requires_grad = True
        print(f"Model name: {model.__class__.__name__}")
        
        ### --- Define the file name to save results
        if len(save_file_name)==0:
            save_file_name = f"Model_{datetime.datetime.now()}"
        print(f"Files will be saved with the base {save_file_name}")
        print()
        train_params["save_file_name"] = save_file_name
        
        ### --- Load the data
        train_data = model_scripts.Train_Image_Data(X_train, y_train, using_preprocessed_images, dim)
        valid_data = model_scripts.Validate_Image_Data(X_valid, y_valid, using_preprocessed_images, dim)
        
        ### --- Instantiate the sampler (optional) and dataloader
        sampler = None
        train_dataloader = DataLoader(train_data, 
                                      batch_size=train_batch_size, 
                                      shuffle=True,
                                      num_workers=n_workers,
                                      persistent_workers=True
                                     )
        valid_dataloader = DataLoader(valid_data, 
                                      batch_size=valid_batch_size,
                                      num_workers=n_workers,
                                      persistent_workers=True
                                     )
        print(f"Using train dataloader: {train_dataloader}")
        print(f"Using valid dataloader: {valid_dataloader}")
        print()
            
        ### Get loss weights
        # ---  Count the number of 0/1/2 values per class
        # --- For this, use the counts from the raw train df to amplify 1/2 counts
        if use_weights:
            class_value_counts = np.zeros((3,n_classes))
            for j,col in enumerate(class_cols):
                counts = train_df[col].value_counts()
                class_value_counts[0,j] = counts[0]
                if 1 in counts.index:
                    class_value_counts[1,j] = counts[1]
                if 2 in counts.index:
                    class_value_counts[2,j] = counts[2]
            weights = model_scripts.get_loss_weights(device, class_value_counts, n_samples, n_classes)
            
            ### Print information for the user
            print("Value weights per class for calculating class losses")
            for i,col in enumerate(class_cols):
                print(f"{col}:\n" + \
                      f"   0: {weights[i][0].item():.6f}" + \
                      f"   1: {weights[i][1].item():.6f}" + \
                      f"   2: {weights[i][2].item():.6f}")
        else: 
            weights = [1 for col in class_cols]
            
        ### Fit the model
        model, t_model, model_loss = model_scripts.fit(device, model, train_dataloader, valid_dataloader, weights, **train_params)
    
        ### Save the results
        torch.save(model.state_dict(), model_filepath)
        print(f"Saved PyTorch Model State to {model_filepath}")
        
        t_models.append(t_model)
        model_losses.append(model_loss)
        # Save model data
        with open(f"results/model_training_times/{save_file_name}_times.txt", "w") as file:
            file.write(str(t_models))
        with open(f"results/model_losses/{save_file_name}_losses.txt", "w") as file:
            file.write(str(model_losses))
        
    ###
    predict_params["save_file_name"] = save_file_name
    predict_params["path_col"] = path_col
    predict_params["model_path"] = model_filepath
    predict_params["device"] = device
    #
    model_outputs = {}

    if test_model:
        predict_type = "test"
        predict_params["confmat_fig_name"] = f"results/confusion_matrix_images/{save_file_name}_{predict_type}.png"
        outputs = model_scripts.get_predictions(df_path = test_filepath, 
                                                predict_type = predict_type,
                                                model_type = model_type,
                                                batch_size = test_batch_size, 
                                                sample_frac = test_sample_frac,
                                                **predict_params)
        model_scripts.print_accuracies(outputs, class_cols)
        model_outputs[predict_type] = outputs
        outputs["df"].to_csv(f"results/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv", index=False)
        
    if validate_rad_model:
        predict_type = "valid_rad"
        predict_params["confmat_fig_name"] = f"results/confusion_matrix_images/{save_file_name}_{predict_type}.png"
        outputs = model_scripts.get_predictions(df_path = valid_rad_filepath, 
                                                predict_type = predict_type, 
                                                model_type = model_type,
                                                batch_size = valid_rad_batch_size, 
                                                sample_frac = valid_rad_sample_frac,
                                                **predict_params)
        model_scripts.print_accuracies(outputs, class_cols)
        model_outputs[predict_type] = outputs
        outputs["df"].to_csv(f"results/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv", index=False)
        
    if test_rad_model:
        predict_type = "test_rad"
        predict_params["confmat_fig_name"] = f"results/confusion_matrix_images/{save_file_name}_{predict_type}.png"
        outputs = model_scripts.get_predictions(df_path = test_rad_filepath, 
                                                predict_type = predict_type, 
                                                model_type = model_type,
                                                batch_size = test_rad_batch_size, 
                                                sample_frac = test_rad_sample_frac,
                                                **predict_params)
        model_scripts.print_accuracies(outputs, class_cols)
        model_outputs[predict_type] = outputs
        outputs["df"].to_csv(f"results/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv", index=False)
        
            
    
    
                