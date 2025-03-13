import pandas as pd
import numpy as np
import datetime 
import torch
import sys
from torch.utils.data import DataLoader

try:
    import scripts.model as model_scripts
except:
    # Replace S:/Chexpert with the correct root as needed
    sys.path.append("S:/Chexpert/scripts")
    import model as model_scripts

################################################################################

### Set initial variables

################################################################################


# # --- Use enhanced training images (additional transforms applied)
# enhanced_images_list = [False, True, False, True, False, True,
#                    False, True, False, True, False, True]

# # --- Select the image resolution value(s): 224, 384, or 512
# dim_list = [224, 224, 384, 384, 512, 512,
#        224, 224, 384, 384, 512, 512]

# # --- Choose ResNet50 or DenseNet121
# model_type_list = ["DenseNet121", "DenseNet121", "DenseNet121",
#                    "DenseNet121", "DenseNet121", "DenseNet121",
#                   "ResNet50", "ResNet50", "ResNet50",
#                   "ResNet50", "ResNet50", "ResNet50"]

# --- Use enhanced training images (additional transforms applied)
enhanced_images_list = [False, True]

# --- Select the image resolution value(s): 224, 384, or 512
dim_list = [512, 512]

# --- Choose ResNet50 or DenseNet121
model_type_list = ["DenseNet121", "DenseNet121"]

##### MAIN CONTROLS ############################################################
# --- Train the model
train_model = True

# --- Test the model on different validation/test sets
validate_model = False
test_model = True
validate_rad_model = True
test_rad_model = True

# --- Model parameters
train_batch_size = 32
train_sample_frac = 1
raw_train_filepath = "data/train_df.csv"
train_filepath = "data/train_df_oversampled2.csv"
# train_filepath = "data/train_df_60000.csv"
#
valid_batch_size = 64
valid_sample_frac = 1
valid_filepath = "data/valid_df2.csv"
#
test_batch_size = 64
test_sample_frac = 1
test_filepath = "data/test_df2.csv"
#
valid_rad_batch_size = 20
valid_rad_sample_frac = 1
valid_rad_filepath = "data/valid_rad_df2.csv"
#
test_rad_batch_size = 50
test_rad_sample_frac = 1
test_rad_filepath = "data/test_rad_df2.csv"

##### Set the class variables ##################################################
class_cols = ["Cardiomegaly", "Consolidation", "Edema", "Atelectasis", "Pleural Effusion"]

class_col_index = [i for i in range(len(class_cols))]
n_classes = len(class_cols)

# --- Class and value weights
classWeights_filepath = "data/classWeights.csv"
valueWeights_filepath = "data/valueWeights.csv"

##### OTHER VARIABLES ##########################################################
# --- Use the GPU
use_cuda = True
# --- Training model architecture variables
n_workers = 4
# --- Set the min/max number of epochs
min_epochs = 8
max_epochs = 50
# --- If True, then all layers but the classification layer will be frozen
freeze_all_but_final_layer = True
# --- If above is True, all layers will be unfrozen at epoch __
unfreeze_all_layers_at_epoch = 100
# --- If above is true, then learning rate will be lowered at epoch __
update_learning_rate_at_epoch = 5
# Early stopping cutoff for loss difference per epoch
ep = 5e-4
# --- Use class weights by value for the loss function
use_weights = True

# --- Wrap these into a dictionary for easier parameter entry
train_params = {"n_workers":n_workers,
                "min_epochs":min_epochs,
                "max_epochs":max_epochs,
                "freeze_all_but_final_layer":freeze_all_but_final_layer,
                "unfreeze_all_layers_at_epoch":unfreeze_all_layers_at_epoch,
                "update_learning_rate_at_epoch":update_learning_rate_at_epoch,
                "use_weights":use_weights,
                "ep":ep,
                "n_classes":n_classes,
                }

predict_params = {"class_cols":class_cols,
                  "min_epochs":min_epochs,
                  "max_epochs":max_epochs,
                  "freeze_all_but_final_layer":freeze_all_but_final_layer,
                  "unfreeze_all_layers_at_epoch":unfreeze_all_layers_at_epoch,
                  "update_learning_rate_at_epoch":update_learning_rate_at_epoch,
                  "ep":ep,
                  "n_classes":n_classes
                  }


################################################################################

### Running the model(s)

################################################################################

for model_type, dim, enhanced_images in zip(model_type_list, dim_list, enhanced_images_list):
    
    save_file_name = f"{model_type}_BASE{dim}" 
    # save_file_name = f"ResNet50_BASE{dim}"
    if enhanced_images: 
        save_file_name = save_file_name + "_ENHANCED"
    # if use_weights:
    #     save_file_name = save_file_name + "_WEIGHTED"
    model_filepath = f"models/{save_file_name}.pth"
        
    ##### Set the path column based on dim and enhanced_images selection
    path_col = f"base{dim}_file_path"
    if enhanced_images:
        path_col += "2"
    
    if dim==224:
        # --- Starting learning rate
        train_params["lr1"] = 0.001
        # --- Second learning rate
        train_params["lr2"] = 0.000001
    elif dim==384:
        # --- Starting learning rate
        train_params["lr1"] = 0.0025
        # --- Second learning rate
        train_params["lr2"] = 0.0000025
    elif dim==512:
        if model_type=="DenseNet121":
            # --- Starting learning rate
            train_params["lr1"] = 0.0025
            # --- Second learning rate
            train_params["lr2"] = 0.0000001
        elif model_type=="ResNet50":
            # --- Starting learning rate
            train_params["lr1"] = 0.0025
            # --- Second learning rate
            train_params["lr2"] = 0.0000025
            
        
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
            raw_train_df = pd.read_csv(raw_train_filepath)
            train_df = pd.read_csv(train_filepath).sample(frac=train_sample_frac)
            valid_df = pd.read_csv(valid_filepath)#.sample(7500)
            #
            X_train = list(train_df[path_col])
            y_train = train_df[class_cols].to_numpy()
            n_samples = len(X_train)
            #
            X_valid = list(valid_df[path_col])
            y_valid = valid_df[class_cols].to_numpy()
            n_valid_samples = len(X_valid)
    
            print(f"\n# training samples: {n_samples}")
            
            ### Set the class and value weights according to the data sample
            # classWeights_df = pd.read_csv(classWeights_filepath)[str(n_samples_i)].tolist()
            # classWeights_df = (pd.read_csv(classWeights_filepath)["train_df"] + pd.read_csv(classWeights_filepath)[str(n_instances)]) / 2
            # classWeights_df = pd.read_csv(classWeights_filepath)[str(n_samples)].tolist()
            # valueWeights_df = pd.read_csv(valueWeights_filepath)[str(n_samples_i)].tolist()
            train_params["classWeights_df"] = [None] * n_classes
            
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
            print(f"Files will be saved with the name {save_file_name}")
            print()
            train_params["save_file_name"] = save_file_name
            
            ### --- Load the data
            train_data = model_scripts.Train_Image_Data(X_train, y_train, dim)
            valid_data = model_scripts.Validate_Image_Data(X_valid, y_valid)
            
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
            print(f"Using sampler: {sampler}")
            print(f"Using train dataloader: {train_dataloader}")
            print(f"Using valid dataloader: {valid_dataloader}")
            print()
                
            ### Get loss weights
            # ---  Count the number of 0/1/2 values per class
            # --- For this, use the counts from the raw train df to amplify 1/2 counts
            if use_weights:
                class_value_counts = np.zeros((3,n_classes))
                for j,col in enumerate(class_cols):
                    # counts = raw_train_df[col].value_counts()
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
                # print()
                # print("Class weights to get a weighted average loss")
                # for i,(col,val) in enumerate(zip(class_cols, classWeights_df)):
                #     print(f"{col}: {val}")
                # print()
            else: 
                weights = [1 for col in class_cols]
                
            # print()
            # print("Class weights to get a weighted average loss")
            # for i,(col,val) in enumerate(zip(class_cols, classWeights_df)):
            #     print(f"{col}: {val}")
            # print()
                
            ### Fit the model
            model, t_model, model_loss = model_scripts.fit(device, model, train_dataloader, valid_dataloader, weights, **train_params)
        
            ### Save the results
            torch.save(model.state_dict(), model_filepath)
            print(f"Saved PyTorch Model State to {model_filepath}")
            
            t_models.append(t_model)
            model_losses.append(model_loss)
            # Save model data
            with open(f"results/{save_file_name}_times.txt", "w") as file:
                file.write(str(t_models))
            with open(f"results/{save_file_name}_losses.txt", "w") as file:
                file.write(str(model_losses))
            
        ###
        predict_params["save_file_name"] = save_file_name
        predict_params["path_col"] = path_col
        predict_params["model_path"] = model_filepath
        predict_params["device"] = device
        #
        model_outputs = {}
        if validate_model:
            predict_type = "valid"
            predict_params["confmat_fig_name"] = f"results/{save_file_name}_{predict_type}.png"
            outputs = model_scripts.get_predictions(df_path = valid_filepath, 
                                                    predict_type = predict_type, 
                                                    model_type = model_type,
                                                    batch_size = valid_batch_size, 
                                                    sample_frac = valid_sample_frac,
                                                    **predict_params)
            model_scripts.print_accuracies(outputs, class_cols)
            model_outputs[predict_type] = outputs
            outputs["df"].to_csv(f"results/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv", index=False)
    
        if test_model:
            predict_type = "test"
            predict_params["confmat_fig_name"] = f"results/{save_file_name}_{predict_type}.png"
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
            predict_params["confmat_fig_name"] = f"results/{save_file_name}_{predict_type}.png"
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
            predict_params["confmat_fig_name"] = f"results/{save_file_name}_{predict_type}.png"
            outputs = model_scripts.get_predictions(df_path = test_rad_filepath, 
                                                    predict_type = predict_type, 
                                                    model_type = model_type,
                                                    batch_size = test_rad_batch_size, 
                                                    sample_frac = test_rad_sample_frac,
                                                    **predict_params)
            model_scripts.print_accuracies(outputs, class_cols)
            model_outputs[predict_type] = outputs
            outputs["df"].to_csv(f"results/{save_file_name}_{predict_type}_withPredictionsAndConfidences.csv", index=False)
            
            
    
    
                