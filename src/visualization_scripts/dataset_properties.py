#%% Imports
import open3d as o3d
import numpy as np
import scipy.io as sio
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import matplotlib as mpl
import pandas as pd
import yaml
import cv2

os.chdir("/home/gs-lfd/dvc-5d-apple-pose-estimation")

# Set the fonttype to TrueType
mpl.rcParams["text.usetex"] = True

# Font sizes
SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# Set background as white
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
# Set grid style
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = '--'
# Set axis color
plt.rcParams['axes.edgecolor'] = 'black'

# Define a color palette
color_palette = sns.color_palette("Paired").as_hex()

#%% Load the test labels of generated and original data

generated_label_folder = "data_disk/dvc_data/randwijk_papple_test_labels/label"
original_label_folder = "data_disk/dvc_data/real_randwijk_papple_test_labels/label"

# Load train/val/test trees
all_params = yaml.safe_load(open("params.yaml"))
params = all_params["convert_randwijk_data_to_fresh"]
train_trees = params["train_trees"]
val_trees = params["val_trees"]
test_trees = params["test_trees"]

all_labels = []
for label_folder in [generated_label_folder, original_label_folder]:
    result_type = "Generated" if label_folder == generated_label_folder else "Original"
    for label_file in os.listdir(label_folder):
        with open(os.path.join(label_folder, label_file), "r") as f:
            for line in f:
                all_elements = line[:-1].split(" ")
                area = (int(all_elements[2]) - int(all_elements[1])) * (int(all_elements[4]) - int(all_elements[3])) 
                if label_file.split(".")[0].startswith("01"):
                    dataset = "Train"
                    tree_id = "tree_01"
                elif label_file.split(".")[0].startswith("02"):
                    dataset = "Train"
                    tree_id = "tree_02"
                elif label_file.split(".")[0].startswith("03"):
                    dataset = "Train"
                    tree_id = "tree_03"
                elif label_file.split(".")[0].startswith("04"):
                    dataset = "Train"
                    tree_id = "tree_04"
                elif label_file.split(".")[0].startswith("05"):
                    dataset = "Val"
                    tree_id = "tree_05"
                elif label_file.split(".")[0].startswith("06"):
                    dataset = "Train"
                    tree_id = "tree_06"
                elif label_file.split(".")[0].startswith("07"):  
                    dataset = "Test"
                    tree_id = "tree_07"
                elif label_file.split(".")[0].startswith("08"):
                    dataset = "Val"
                    tree_id = "tree_08"
                elif label_file.split(".")[0].startswith("09"):
                    dataset = "Train"
                    tree_id = "tree_09"
                elif label_file.split(".")[0].startswith("10"):
                    dataset = "Train"
                    tree_id = "tree_10"
                elif label_file.split(".")[0].startswith("11"):
                    dataset = "Train"
                    tree_id = "tree_11"
                elif label_file.split(".")[0].startswith("12"):
                    dataset = "Train"
                    tree_id = "tree_12"
                label_data = {
                    "result_type": result_type,
                    "dataset": dataset,
                    "tree": tree_id,
                    "label_file": label_file,
                    "category": all_elements[0],
                    "x_min": int(all_elements[1]),
                    "x_max": int(all_elements[2]),
                    "y_min": int(all_elements[3]),
                    "y_max": int(all_elements[4]),
                    "bounding_box_area": area,
                    "coord_x": float(all_elements[5]),
                    "coord_y": float(all_elements[6]),
                    "coord_z": float(all_elements[7]),
                    "size_x": float(all_elements[8]),
                    "size_y": float(all_elements[9]),
                    "size_z": float(all_elements[10]),
                    "roll": float(all_elements[11]),
                    "pitch": float(all_elements[12]),
                    "yaw": float(all_elements[13]),
                    "visibility": float(all_elements[15]),
                    "fruit_id": int(all_elements[16]),
                }
                all_labels += [label_data]

df_all_labels = pd.DataFrame(all_labels)

#%% Additional processing

df_all_labels["roll_deg"] = df_all_labels["roll"] * 180 / np.pi
df_all_labels["pitch_deg"] = df_all_labels["pitch"] * 180 / np.pi
df_all_labels["yaw_deg"] = df_all_labels["yaw"] * 180 / np.pi

# Rotate yaw by -90 degrees to align with camera coordinate system
df_all_labels["yaw_deg"] -= 90.0

# Remap below -180
too_lows_roll = df_all_labels["roll_deg"] < -180.0
df_all_labels.loc[too_lows_roll, "roll_deg"] += 360.0
too_lows_pitch = df_all_labels["pitch_deg"] < -180.0
df_all_labels.loc[too_lows_pitch, "pitch_deg"] += 360.0
too_lows_yaw = df_all_labels["yaw_deg"] < -180.0
df_all_labels.loc[too_lows_yaw, "yaw_deg"] += 360.0

df_all_labels["vector_x"] = np.sin(np.deg2rad(df_all_labels["yaw_deg"])) * np.cos(np.deg2rad(df_all_labels["pitch_deg"]))
df_all_labels["vector_y"] = np.sin(np.deg2rad(df_all_labels["yaw_deg"])) * np.sin(np.deg2rad(df_all_labels["pitch_deg"]))
df_all_labels["vector_z"] = np.cos(np.deg2rad(df_all_labels["yaw_deg"]))

# KDE of orientation vectors
from scipy.stats import gaussian_kde

# KDE of original data
original_vectors = df_all_labels[df_all_labels["result_type"] == "Original"][["vector_x", "vector_y", "vector_z"]].to_numpy().T
kde_original = gaussian_kde(original_vectors)
kde_original.set_bandwidth(bw_method=kde_original.factor * 3.)
density_original = kde_original(original_vectors)
df_all_labels.loc[df_all_labels["result_type"] == "Original", "density"] = density_original
# KDE of generated data
generated_vectors = df_all_labels[df_all_labels["result_type"] == "Generated"][["vector_x", "vector_y", "vector_z"]].to_numpy().T
kde_generated = gaussian_kde(generated_vectors)
kde_generated.set_bandwidth(bw_method=kde_generated.factor * 3.)
density_generated = kde_generated(generated_vectors)
df_all_labels.loc[df_all_labels["result_type"] == "Generated", "density"] = density_generated

# Create a melted dataframe for orientations
df_original_orientations = df_all_labels[df_all_labels["result_type"] == "Original"][["roll_deg", "pitch_deg", "yaw_deg"]].melt(var_name="Orientation", value_name="Angle")
df_generated_orientations = df_all_labels[df_all_labels["result_type"] == "Generated"][["roll_deg", "pitch_deg", "yaw_deg"]].melt(var_name="Orientation", value_name="Angle")
df_original_orientations["Result_type"] = "Original"
df_generated_orientations["Result_type"] = "Generated"
df_all_orientations = pd.concat([df_original_orientations, df_generated_orientations], ignore_index=True)

# Save to CSV
df_all_labels.to_csv("data_disk/testing/test_labels/all_labels.csv", index=False)
df_all_orientations.to_csv("data_disk/testing/test_labels/all_orientations.csv", index=False)

#%% Load from CSV
df_all_labels = pd.read_csv("data_disk/testing/test_labels/all_labels.csv")
df_all_orientations = pd.read_csv("data_disk/testing/test_labels/all_orientations.csv")

# Determine occlusion_rate
df_all_labels["occlusion_rate"] = 1.0 - df_all_labels["visibility"]

# Determine angle_dist
df_all_labels["angle_dist"] = np.sqrt(df_all_labels.pitch_deg**2 + df_all_labels.yaw_deg**2)

#%% Plot orientations

for dataset_selector in df_all_labels["result_type"].unique():
    for split_iterator, dataset_split in enumerate(["Train", "Val", "Test"]):
        # Alternative plot
        g1 = sns.JointGrid(data=df_all_labels[(df_all_labels["result_type"] == dataset_selector) & (df_all_labels["dataset"] == dataset_split)], x="pitch_deg", y="yaw_deg", xlim=(-90,90), ylim=(-180,180), height=4)
        g1.plot_joint(sns.histplot, element="step", color=color_palette[1+split_iterator*2])
        g1.plot_marginals(sns.histplot, element="step", color=color_palette[1+split_iterator*2])
        g1.set_axis_labels("Pitch (degrees)", "Yaw (degrees)")

        # Save both figures
        g1.savefig(f"data_disk/testing/test_labels/{dataset_selector.lower()}_{dataset_split.lower()}_pitch_yaw_distribution.pdf", bbox_inches='tight', dpi=300)
        plt.show()

#%% Plot onto sphere

for dataset_selector in df_all_labels["result_type"].unique():
    fig1 = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_box_aspect([1,1,1])  # Equal aspect ratio   
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    # Plot points
    ax1.scatter(
        df_all_labels[df_all_labels["result_type"] == dataset_selector]["vector_x"], 
        df_all_labels[df_all_labels["result_type"] == dataset_selector]["vector_y"],
        df_all_labels[df_all_labels["result_type"] == dataset_selector]["vector_z"],
        s=0.2,
        c=df_all_labels[df_all_labels["result_type"] == dataset_selector]["density"],
        cmap='flare_r',
    )

    # Remove axis labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    # Plot sphere as gridlines
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x, y, z, color='gray', alpha=0.4)

    # Save as png
    image_dpi = 300
    fig1.savefig(f"data_disk/testing/test_labels/{dataset_selector.lower()}_data_orientations_on_sphere.png", bbox_inches='tight', dpi=image_dpi)

    # Import image
    image_fig = cv2.imread(f"data_disk/testing/test_labels/{dataset_selector.lower()}_data_orientations_on_sphere.png", cv2.IMREAD_COLOR_RGB)

    # Create new figure
    fig2 = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax2 = fig2.add_subplot(111)

    ax2.imshow(image_fig[100:-90, 110:-110])
    
    # Remove axis ticks and lines
    ax2.axis("off")


    # Add a colorbar
    mappable = plt.cm.ScalarMappable(cmap='flare_r')
    mappable.set_array(df_all_labels[df_all_labels["result_type"] == dataset_selector]["density"])
    cbar = plt.colorbar(mappable, ax=ax2, fraction=0.05, pad=0.1)
    cbar.set_label('Density')

    # Save figure
    fig2.savefig(f"data_disk/testing/test_labels/{dataset_selector.lower()}_data_orientations_on_sphere.pdf", bbox_inches='tight', dpi=image_dpi)
    plt.show()

# %% Plot the visibility of the apples

for dataset_selector in df_all_labels.result_type.unique():
    fig, axs = plt.subplots(3, 1, 
        figsize=(5, 5), 
        constrained_layout=True,
        sharex=True,
    )
    for split_iterator, dataset_split in enumerate(["Train", "Val", "Test"]):
        sns.histplot(df_all_labels[(df_all_labels["result_type"] == dataset_selector) & (df_all_labels["dataset"] == dataset_split)], x="occlusion_rate", kde=True, bins=100, element="step", stat="count", ax=axs[split_iterator], color=color_palette[1+split_iterator*2])

        # Invert y-axis
        # axs[split_iterator].yaxis.set_inverted(True)
        axs[split_iterator].set_title(dataset_split)

    axs[2].set_xlabel("Occlusion rate")

    fig.savefig(f"data_disk/testing/test_labels/{dataset_selector.lower()}_visibility_split.pdf")
    
    plt.show()

# %% Plot cropped image of examples of visibilities

original_df_labels = df_all_labels[(df_all_labels["result_type"] == "Original") & (df_all_labels["dataset"] == "Val") & (df_all_labels["bounding_box_area"] >= 5000)]

visibility_examples = [0.985, 0.8, 0.6, 0.4, 0.2, 0.16]

for visibility_example in visibility_examples:
    visibility_df_labels = original_df_labels[(original_df_labels["occlusion_rate"] >= (visibility_example - 0.01)) & (original_df_labels["occlusion_rate"] <= (visibility_example + 0.01))]

    for i in range(10):
        row = visibility_df_labels.iloc[i+10]
        if row.label_file.startswith("13"):
            print(f"Skipping iter {i}")
            continue
        img_file_name = row.label_file.replace(".txt", ".jpg")

        # Load image
        img = cv2.imread("data_disk/dvc_data/combined_randwijk_papple/image/"+img_file_name, cv2.IMREAD_COLOR_RGB)

        
        plt.imshow(img[row.y_min:row.y_max,row.x_min:row.x_max])
        plt.show()

        cropped_image = img[row.y_min:row.y_max,row.x_min:row.x_max]
        # Save
        cv2.imwrite(f"data_disk/testing/test_labels/example_{int(visibility_example*100)}.jpg", cropped_image[:,:,::-1])
