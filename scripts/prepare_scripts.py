import os
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

##### Preprocessing dataframes ##############################################

def process_dataframes(df, source_root, old_root, output_root, dims):
    """
    Actions taken to process the data frames:
    1) Loop through the specified source_root folder to find all image paths that exist
        a. Store each path string in the valid_paths list
    2) For each image metadata row in df, add new path columns:
        a. floating_file_path: The image file path without the root folder (ex: "S:/Root/patientX/image.jpg" to "patientX/image.jpg")
        b. source_file_path: The file path to the original image saved from the blob
        c. base{dim}_file_path: The file path to the saved dim x dim processed image
        d. base{dim}_file_path2: The file path to the saved enhanced dim x dim processed image
    3) Filter out any images whose source_file_path is not contained in the valid_paths (created in step 1)
    4) Return the processed dataframes
    """
    # Create a copy of df
    df2 = df.copy()
    # Loop through the directories and get paths that exist
    valid_paths = get_valid_paths(source_root)
    # Separate the floating file paths from the old_root
    df2["floating_file_path"] = df2["Path"].str[len(old_root):]
    # Create a filepath to the image from the specified source_root
    df2["source_file_path"] = source_root + df2["floating_file_path"]
    # Add new file paths for processed images
    for dim in dims:
        dim = str(dim)
        col1, col2 = f"base{dim}_file_path", f"base{dim}_file_path2"
        df2[col1] = output_root + df2["floating_file_path"].str[:-4] + f"_{dim}.jpg"
        df2[col2] = df2[col1].str[:-4] + "_usm.jpg"
    # Filter the dfs for only valid paths from the source file paths
    df2 = df2[df2["source_file_path"].isin(valid_paths)]

    return df2
    
    
    