import json
import yaml
import open3d as o3d
import numpy as np
import cv2
import scipy.io as sio
import os
import sys
import shutil
import pandas as pd
from multiprocessing.pool import ThreadPool

def count_labels_in_real_data(real_data_converted_folder, train_files):
    image_label_count = {}

    # Iterate over all label files
    for file in os.listdir(os.path.join(real_data_converted_folder, "label")):
        # Get the frame number from the file name
        frame_num = file.split(".")[0]

        # Check how many labels are in the file
        with open(os.path.join(real_data_converted_folder, "label", file), "r") as f:
            lines = f.readlines()
            
        num_labels = len(lines)

        # Check which split the file belongs to and add the number of labels
        if frame_num in train_files:
            image_label_count[frame_num] = num_labels
        
    # Return the amounts
    return image_label_count

def move_frame(
    iterator_with_row,
    splat_fresh_folder, real_fresh_folder, output_fresh_folder,
):
    ## Part 0: Set up input and output name
    iterator_row = iterator_with_row[1]
    combined_idx = iterator_row["combined_idx"]
    original_idx = iterator_row["origin_idx"]
    data_origin = iterator_row["origin"]
    if data_origin == "splat":
        input_folder = splat_fresh_folder
    elif data_origin == "real":
        input_folder = real_fresh_folder
    else:
        raise ValueError(f"Unknown data origin: {data_origin}")

    ## Part 1: Copy the calib data
    input_calib_path = os.path.join(input_folder, "calib", original_idx + ".txt")
    output_calib_path = os.path.join(output_fresh_folder, "calib", combined_idx + ".txt")
    shutil.copyfile(input_calib_path, output_calib_path)

    ## Part 2: Copy the depth data
    input_depth_path = os.path.join(input_folder, "depth", original_idx + ".mat")
    output_depth_path = os.path.join(output_fresh_folder, "depth", combined_idx + ".mat")
    shutil.copyfile(input_depth_path, output_depth_path)

    ## Part 3: Copy the image data
    input_image_path = os.path.join(input_folder, "image", original_idx + ".jpg")
    output_image_path = os.path.join(output_fresh_folder, "image", combined_idx + ".jpg")
    shutil.copyfile(input_image_path, output_image_path)

    ## Part 4: Copy the label data
    input_label_path = os.path.join(input_folder, "label", original_idx + ".txt")
    output_label_path = os.path.join(output_fresh_folder, "label", combined_idx + ".txt")
    shutil.copyfile(input_label_path, output_label_path)

    return None

def main():
    params = yaml.safe_load(open("params.yaml"))
    train_trees = params["convert_randwijk_data_to_fresh"]["train_trees"]
    real_fraction = params["combine_real_and_splat_randwijk_data"]["real_fraction"]
    
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython combine_real_and_splat_randwijk_data.py splat-fresh-folder real-fresh-folder output-fresh-folder \n")
        sys.exit(1)

    splat_fresh_folder = sys.argv[1]
    real_fresh_folder = sys.argv[2]
    output_fresh_folder = sys.argv[3]

    # # Define folders for testing
    # splat_fresh_folder = "data_disk/dvc_data/randwijk_papple_splat"
    # real_fresh_folder = "data_disk/dvc_data/real_randwijk_papple"
    # output_fresh_folder = "data_disk/dvc_data/combined_randwijk_papple"

    # Create the output directories
    os.makedirs(output_fresh_folder, exist_ok=True)
    os.makedirs(os.path.join(output_fresh_folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_fresh_folder, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_fresh_folder, "calib"), exist_ok=True)
    os.makedirs(os.path.join(output_fresh_folder, "label"), exist_ok=True)

    # Get train idx from the splat folder
    splat_train_idx = np.loadtxt(
        os.path.join(splat_fresh_folder, "train_data_idx.txt"), dtype=str
    )
    # Get real train idx from the real folder
    real_train_idx = np.loadtxt(
        os.path.join(real_fresh_folder, "train_data_idx.txt"), dtype=str
    )
    # Get val and test idx only from the real folder
    val_idx = np.loadtxt(
        os.path.join(real_fresh_folder, "val_data_idx.txt"), dtype=str
    )
    test_idx = np.loadtxt(
        os.path.join(real_fresh_folder, "test_data_idx.txt"), dtype=str
    )
    
    df = pd.DataFrame(columns=["combined_idx", "origin", "origin_idx", "split"])

    # Add the train data to the dataframe
    i = 0
    for i, idx in enumerate(splat_train_idx):
        df.loc[len(df)] = {
            "combined_idx": str(i),
            "origin": "splat",
            "origin_idx": idx,
            "split": "train",
        }
    for j, idx in enumerate(real_train_idx):
        df.loc[len(df)] = {
            "combined_idx": str(i + j + 1),
            "origin": "real",
            "origin_idx": idx,
            "split": "train",
        }

    # Randomly shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Determine the number of labels in the real data
    real_image_label_count = count_labels_in_real_data(
        real_fresh_folder, real_train_idx
    )
    real_sum_labels = sum(real_image_label_count.values())

    # Get the correct number of real data to add
    real_data_to_add = int(real_sum_labels * real_fraction)
    real_data_added = 0

    for i in range(len(df)):
        if df.loc[i, "origin"] == "real":
            # If the row is from real data, check if we need to remove it
            if (real_data_added + real_image_label_count[df.loc[i, "origin_idx"]]) <= real_data_to_add and real_image_label_count[df.loc[i, "origin_idx"]] > 0:
                # If we still need to add real data, keep it
                real_data_added += real_image_label_count[df.loc[i, "origin_idx"]]
            else:
                # If we have added enough real data, remove this row
                df.drop(i, inplace=True)

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    # Create the combined_idx column
    for tree in train_trees:
        tree_id = f"{int(tree.split('_')[1]):02d}"

        tree_df = df[df["origin_idx"].str.startswith(tree_id)]
        # Create the correct combined_idx column
        for i in range(len(tree_df)):
            df.loc[tree_df.index[i], "combined_idx"] = tree_id + f"{i:04d}"

    # Iterate val and test idx
    for val_id in val_idx:
        df.loc[len(df)] = {
            "combined_idx": val_id,
            "origin": "real",
            "origin_idx": val_id,
            "split": "val",
        }
    for test_id in test_idx:
        df.loc[len(df)] = {
            "combined_idx": test_id,
            "origin": "real",
            "origin_idx": test_id,
            "split": "test",
        }


    n_cores = int(os.cpu_count()) 
    with ThreadPool(n_cores) as pool:
        tree_used_frames = list(pool.starmap(
            move_frame, 
            [
                (
                    iterator_with_row,
                    splat_fresh_folder, real_fresh_folder, output_fresh_folder,
                ) for iterator_with_row in df.iterrows()
            ]
        ))
        
    # Make seperate df for train, val and test
    train_combined_idx = df[df["split"] == "train"]["combined_idx"]
    with open(os.path.join(output_fresh_folder, "train_data_idx.txt"), "a") as f:
        for f_id in train_combined_idx:
            f.write(f"{f_id}\n")

    val_combined_idx = df[df["split"] == "val"]["combined_idx"]
    with open(os.path.join(output_fresh_folder, "val_data_idx.txt"), "a") as f:
        for f_id in val_combined_idx:
            f.write(f"{f_id}\n")

    test_combined_idx = df[df["split"] == "test"]["combined_idx"]
    with open(os.path.join(output_fresh_folder, "test_data_idx.txt"), "a") as f:
        for f_id in test_combined_idx:
            f.write(f"{f_id}\n")

if __name__ == "__main__":
    main()