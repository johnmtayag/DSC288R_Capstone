import os
from shutil import copyfile
###

def get_valid_paths(old_root):
    """
    The training dataset lists filepaths for the entire dataset, so this extracts only the filepaths available in the directory
    """
    valid_paths = []
    for patient in os.listdir(old_root):
        patient_path = old_root + patient
        studies = os.listdir(patient_path)
        for study in studies:
            study_path = patient_path + "/" + study
            images = os.listdir(study_path)
            for image in images:
                image_path = study_path + "/" + image
                valid_paths.append(image_path)

    return valid_paths

def get_valid_paths2(root):
    """
    The training dataset lists filepaths for the entire dataset, so this extracts only the filepaths available in the directory. Same as the original function, except in the flattened structure
    """
    valid_paths = []
    for img in os.listdir(root):
        img_path = root + img
        valid_paths.append(img_path)

    return valid_paths

def prepare_datasets(old_root, new_root, dataset_type):
    """ Move files contained in the old training structure into a flattened representation in a single directory"""
    # Define new base variables
    if dataset_type == "train":
        new_root += "train"
    elif dataset_type == "valid":
        new_root += "valid"
    else:
        print("Error - invalid dataset_type selection")
        return None
    
    # Create output directories if needed
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    ### Copy files to the new folder
    # Get a list of file paths available in temp_base_folder
    valid_paths =  get_valid_paths(old_root)
    
    # Copy all valid files to the new directory
    for file_path in valid_paths:
        new_filename = file_path[len(old_root):].replace("/", "_")
        copyfile(file_path, f"{new_root}/{new_filename}")

def copy_directory_structure(source, destination):
    """Copy directory structure from the source path to the destination path"""
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    for dir_path, dir_names, file_names in os.walk(source):
        full_path = os.path.join(destination, dir_path[len(source):])
        if os.path.isdir(full_path):
            continue
        else:
            os.mkdir(full_path)
    