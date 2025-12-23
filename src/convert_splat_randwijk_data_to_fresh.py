import json
import open3d as o3d
import numpy as np
import cv2
import scipy.io as sio
import os
import yaml
import sys
import transforms3d as t3d
import copy
from multiprocessing.pool import ThreadPool

def count_labels_in_real_data(real_data_converted_folder):
    # Get files in each split
    train_files = np.loadtxt(os.path.join(real_data_converted_folder, "train_data_idx.txt"), dtype=int)
    val_files = np.loadtxt(os.path.join(real_data_converted_folder, "val_data_idx.txt"), dtype=int)
    test_files = np.loadtxt(os.path.join(real_data_converted_folder, "test_data_idx.txt"), dtype=int)

    real_train_labels, real_val_labels, real_test_labels = 0, 0, 0

    # Iterate over all label files
    for file in os.listdir(os.path.join(real_data_converted_folder, "label")):
        # Get the frame number from the file name
        frame_num = int(file.split(".")[0])

        # Check how many labels are in the file
        with open(os.path.join(real_data_converted_folder, "label", file), "r") as f:
            lines = f.readlines()
            
        num_labels = len(lines)

        # Check which split the file belongs to and add the number of labels
        if frame_num in train_files:
            real_train_labels += num_labels
        elif frame_num in val_files:
            real_val_labels += num_labels
        elif frame_num in test_files:
            real_test_labels += num_labels
        
    # Return the amounts
    return real_train_labels, real_val_labels, real_test_labels

def load_apples_pcd(directory, tree_bbs):
    # Initialize empty dict for apples
    individual_apples = {}

    # Rotate the point clouds 180 degrees around X to align with the camera frame
    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    for files in os.listdir(directory):

    # Load the segmented apple point cloud with RGB colors
        file_path = os.path.join(directory, files)
        
        parts = files.replace('.ply', '').split('_')
        apple_id = int(parts[1])    # Extracts the 1
        calyx_idx = int(parts[2])
 
        point_cloud = o3d.io.read_point_cloud(file_path)
        points_3d = np.asarray(point_cloud.points)  # XYZ coordinates
        
        # Change color to apple_id
        colors = np.full((points_3d.shape[0], 3), float(apple_id)/255.0)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Add points to the total point cloud
        fruit_points = o3d.t.geometry.PointCloud.from_legacy(point_cloud) # Add the current point cloud to the total point cloud
        
        # Add the calyx point to the calyx point cloud
        calyx_points = o3d.t.geometry.PointCloud.from_legacy(point_cloud.select_by_index([calyx_idx])) # Add the calyx point to the calyx point cloud

        individual_apples[apple_id] = [
            fruit_points.rotate(rotation_matrix, center=(0, 0, 0)), 
            calyx_points.rotate(rotation_matrix, center=(0, 0, 0))
        ]

    trees_dict = {}
    # Seperate by tree using the bounding boxes
    for tree_name, dict_bb in tree_bbs.items():
        trees_dict[tree_name] = {k:individual_apples[k] for k in dict_bb['fruits'] if k in individual_apples}

    return trees_dict

def load_frame_pcd(
        depth_images_folder, images_folder,
        tree, frame, K, max_background_depth,
        color_threshold):
    img_filename = frame["file_path"]
            
    # Load the depth image
    depth_image = o3d.io.read_image(os.path.join(depth_images_folder, img_filename.replace(".jpg", ".png")))

    # Load the color image
    color_image = o3d.io.read_image(os.path.join(images_folder, img_filename))
    
    # Combine to RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_scale=65535.0/max_background_depth, depth_trunc=0.96*max_background_depth, convert_rgb_to_intensity=False)

    # Convert to point cloud using the camera intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, K)

    # Find points with color
    valid_points_red = np.asarray(pcd.colors)[:, 0] >= color_threshold 
    valid_points_green = np.asarray(pcd.colors)[:, 1] >= color_threshold
    valid_points_blue = np.asarray(pcd.colors)[:, 2] >= color_threshold
    valid_points = np.logical_or(valid_points_red, np.logical_or(valid_points_green, valid_points_blue))

    pcd = pcd.select_by_index(np.where(valid_points)[0])

    rotated_pcd = copy.deepcopy(pcd)
    rot_mat = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    rot_center = np.asarray([0.0, 0.0, 0.0])
    rotated_pcd.rotate(R=rot_mat, center=rot_center)

    # Return the point cloud
    return rotated_pcd

def convert_frame(
    iterator_w_frame, tree, K, 
    cloud_cloud_dist, min_mask_size, min_view_fraction,
    input_images_folder, input_depth_folder, output_folder,
    tree_fruits_dict,
    split_label_target, split_label_count,
    max_background_depth, color_threshold,
):
    ## Part 0: Set up output numerator and other variables
    frame_i, frame = iterator_w_frame
    tree_num = int(tree.split("_")[1])
    output_numerator = f"{tree_num:02d}" + f"{frame_i:04d}"

    cam_transform = o3d.core.Tensor(frame["transform_matrix"])

    ## Part 1: Load the PC data and make sure it's not empty
    try:
        pc_frame = load_frame_pcd(
            input_depth_folder, input_images_folder,
            tree, frame, K,
            max_background_depth, color_threshold,
        )
    except:
        return None, 0

    # Chec if the pc data is empty
    if len(pc_frame.points) == 0:
        return None, 0

    ## Part 3: Load the image data
    image = cv2.imread(os.path.join(input_images_folder, frame["file_path"]))

    ## Part 4: Load the annotation data
    label_data = []

    # Rotate for openGL standard
    rot_mat = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    rot_center = np.asarray([0.0, 0.0, 0.0])

    for fruit_id, fruit_pcs in tree_fruits_dict.items():
        # Get the fruit points and calyx points
        fruit_points = o3d.t.geometry.PointCloud.from_legacy(fruit_pcs[0].to_legacy())
        calyx_points = o3d.t.geometry.PointCloud.from_legacy(fruit_pcs[1].to_legacy())

        # Check if the fruit points are empty
        if len(fruit_points.point.positions) == 0:
            continue

        # Check if the calyx points are empty
        if len(calyx_points.point.positions) == 0:
            continue

        # Project fruit points to the camera frame
        fruits_in_cam = fruit_points.project_to_rgbd_image(
            K.width, K.height, K.intrinsic_matrix, cam_transform.inv(), depth_max=10
        )

        fruit_colors = np.unique(np.asarray(fruits_in_cam.color))

        if len(fruit_colors) == 1: # Skip if just background in the frame
            continue

        for color_value in fruit_colors:
            if color_value == 0: # Skip the background
                continue

            mask = np.asarray(fruits_in_cam.color) == color_value
            mask = mask[:,:,0]
            fruit_mask_size = np.sum(mask)
            
            # Get the upper and lower bounds of the mask
            mask_bounds = np.argwhere(mask)
            y0, x0 = mask_bounds.min(axis=0)
            y1, x1 = mask_bounds.max(axis=0)

            ## Make sure the fruit is visible compared to the depth image
            # Convert the point cloud frame to a depth image
            pc_rotated = copy.deepcopy(pc_frame)
            pc_rotated.rotate(R=rot_mat, center=rot_center)
            pc_depth = o3d.t.geometry.PointCloud.from_legacy(pc_rotated).project_to_depth_image(
                K.width, K.height, K.intrinsic_matrix, depth_max=10
            ).as_tensor().numpy()[:,:,0]
            fruits_in_cam_depth = fruits_in_cam.depth.as_tensor().numpy()[:,:,0]

            # Get the depth values within the mask
            pc_behind_fruit = pc_depth > (fruits_in_cam_depth - 1000 * cloud_cloud_dist)
            pc_before_fruit = pc_depth < (fruits_in_cam_depth + 1000 * cloud_cloud_dist)

            # Mask the depth values
            masked_pc_in_plane_with_fruit = np.logical_and(
                mask, 
                np.logical_and(
                    pc_behind_fruit,
                    pc_before_fruit))

            # Count number of pixels in the mask
            mask_size = np.sum(masked_pc_in_plane_with_fruit)
            
            # Check if the mask is too small
            if mask_size < min_mask_size:
                continue

            # Check if the mask is too small
            view_fraction = mask_size / fruit_mask_size
            if view_fraction < min_view_fraction:
                continue

            # Convert the fruit points to the camera frame
            fruit_points.transform(cam_transform.inv()).rotate(R=rot_mat, center=rot_center)
            calyx_points.transform(cam_transform.inv()).rotate(R=rot_mat, center=rot_center)

            # Get center of fruit point cloud
            center_coord = fruit_points.get_center().numpy()
            extents = fruit_points.get_axis_aligned_bounding_box().get_extent().numpy()
            unit_vector = center_coord - calyx_points.get_center().numpy()
            unit_vector = unit_vector / np.linalg.norm(unit_vector)
            
            zero_rot_vector = np.array([1.0, 0.0, 0.0])

            rot_axis = np.cross(zero_rot_vector, unit_vector)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            rot_angle = np.arccos(np.dot(zero_rot_vector, unit_vector))

            euler_angles = t3d.euler.axangle2euler(
                rot_axis, 
                rot_angle, 
                axes="sxyz"
            )

            apple_label = [
                "apple", x0, x1, y0, y1, 
                center_coord[0], center_coord[1], center_coord[2],
                extents[0], extents[1], extents[2], 
                0.0, euler_angles[1], euler_angles[2], 
                1.0, view_fraction, fruit_id
            ]

            label_data.append(apple_label)

    # Check if number of label will exceed the target
    if (split_label_count + len(label_data)) > split_label_target:
        return None, 0
    # If it won't, store the result
    else:
        # Save the point cloud data
        pc_data_xyz = np.asarray(pc_frame.points)
        pc_data_rgb = np.asarray(pc_frame.colors)
        pc_data = np.hstack([pc_data_xyz, pc_data_rgb])
        dense_pc_file = os.path.join(output_folder, "depth", output_numerator + ".mat")
        sio.savemat(dense_pc_file, {"data": pc_data})

        # Save the image data
        image_file = os.path.join(output_folder, "image", output_numerator + ".jpg")
        cv2.imwrite(image_file, image)

        # Save the camera data
        calib_rt = np.eye(3)
        calib = np.vstack([calib_rt.flatten(), K.intrinsic_matrix.T.flatten()])
        calib_file = os.path.join(output_folder, "calib", output_numerator + ".txt")
        np.savetxt(calib_file, calib)

        label_file = os.path.join(output_folder, "label", output_numerator + ".txt")
        with open(label_file, "w") as f:
            for label in label_data:
                f.write(" ".join(str(x) for x in label) + "\n")

        return output_numerator, len(label_data)
    
def bulk_convert_frames(
    iterator_w_frame, trees_dict, per_tree_transforms,
    cloud_cloud_dist, min_mask_size, min_view_fraction,
    input_images_folder, input_depth_folder, output_folder,
    max_background_depth, color_threshold,
):
    frame_i, frame = iterator_w_frame
                
    tree = frame["file_path"].split("/")[0]

    # Get the fruit and calyx points
    tree_fruits_dict = trees_dict[tree]
    
    # Get the camera intrinsics from the tree_transforms
    w = per_tree_transforms[tree]["w"]
    h = per_tree_transforms[tree]["h"]
    fl_x = per_tree_transforms[tree]["fl_x"]
    fl_y = per_tree_transforms[tree]["fl_y"]
    cx = per_tree_transforms[tree]["cx"]
    cy = per_tree_transforms[tree]["cy"]

    # Create the camera intrinsics matrix
    K = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

    output_numerator, len_label_data = convert_frame(
        (frame_i, frame), tree, K, 
        cloud_cloud_dist, min_mask_size, min_view_fraction,
        input_images_folder, input_depth_folder, output_folder,
        tree_fruits_dict,
        1000, 0,
        max_background_depth, color_threshold,
    )

    if output_numerator is not None:
        return output_numerator
    else:
        return

def main():
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["convert_randwijk_data_to_fresh"]
    min_mask_size = params["min_mask_size"]
    min_view_fraction = params["min_view_fraction"]
    cloud_cloud_dist = params["cloud_cloud_dist"]
    train_trees = params["train_trees"]
    val_trees = params["val_trees"]
    test_trees = params["test_trees"]

    splat_params = all_params["convert_splat_randwijk_data_to_fresh"]
    real_fraction = splat_params["real_fraction"]

    depth_params = all_params["convert_depth_data"]
    max_background_depth = depth_params["max_background_depth"]
    color_threshold = depth_params["color_threshold"]

    tree_bbs = all_params["tree_bounding_boxes"]

    if len(sys.argv) != 6:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython convert_splat_randwijk_data_to_fresh.py real-data-converted-folder image-folder input-depth-folder apple-pc-folder data-output-folder\n")
        sys.exit(1)

    real_data_converted_folder = sys.argv[1]
    input_images_folder = sys.argv[2]
    input_depth_folder = sys.argv[3]
    input_apples_folder = sys.argv[4]
    output_folder = sys.argv[5]

    # # Testing folders
    # real_data_converted_folder = "data_disk/dvc_data/real_randwijk_papple"
    # input_images_folder = "data_disk/dvc_data/randwijk_row_4_images_from_splat"
    # input_depth_folder =  "data_disk/dvc_data/randwijk_row_4_depth_from_splat"
    # input_apples_folder = "data_disk/dvc_data/randwijk_row_4_individual_apples"
    # output_folder = "data_disk/testing/gs_dataset"

    # Count the number of labels in the real data
    real_train_labels, real_val_labels, real_test_labels = count_labels_in_real_data(real_data_converted_folder)
    
    # Load input apples
    trees_dict = load_apples_pcd(input_apples_folder, tree_bbs)

    # Make the output directory if it doesn't exist
    output_labels = os.path.join(output_folder, "label")
    output_images = os.path.join(output_folder, "image")
    output_depth = os.path.join(output_folder, "depth")
    output_calib = os.path.join(output_folder, "calib")

    # Create the output directories
    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_depth, exist_ok=True)
    os.makedirs(output_calib, exist_ok=True)

    # List the files in the depth images folder
    images_list = os.listdir(input_images_folder)
    per_tree_transforms = {}

    # If there is a json file, load it
    for tree in images_list:
        # List files in the tree folder
        tree_files = os.listdir(os.path.join(input_images_folder, tree))
        # If there is a json file, load it
        if "transforms.json" in tree_files:
            with open(os.path.join(input_images_folder, tree, "transforms.json"), "r") as f:
                per_tree_transforms[tree] = json.load(f)

    # Combine frames per split
    train_frames = []
    val_frames = []
    test_frames = []
    for tree in per_tree_transforms:
        tree_frames = per_tree_transforms[tree]["frames"]
        for tree_frame in tree_frames:
            tree_frame["file_path"] = tree + "/" + tree_frame["file_path"]

            if tree in train_trees:
                train_frames += [tree_frame]
            elif tree in val_trees:
                val_frames += [tree_frame]
            elif tree in test_trees:
                test_frames += [tree_frame]

    splits = {
        "train": (train_frames, int(real_fraction * real_train_labels)),
        "val": (val_frames, int(real_fraction * real_val_labels)),
        "test": (test_frames, int(real_fraction * real_test_labels)),
    }

    for data_split in splits:
        split_frames, split_label_target = splits[data_split]
        split_label_count = 0

        split_output_numerators = []

        # Shuffle the list of split_frames with a determined seed
        np.random.seed(42)
        np.random.shuffle(split_frames)

        if real_fraction < 10.0:
        # Make while loop iterating while target not reached
            frame_i = 0
            while split_label_count < split_label_target:
                frame = split_frames[frame_i]
                
                tree = frame["file_path"].split("/")[0]

                # Get the fruit and calyx points
                tree_fruits_dict = trees_dict[tree]
                
                # Get the camera intrinsics from the tree_transforms
                w = per_tree_transforms[tree]["w"]
                h = per_tree_transforms[tree]["h"]
                fl_x = per_tree_transforms[tree]["fl_x"]
                fl_y = per_tree_transforms[tree]["fl_y"]
                cx = per_tree_transforms[tree]["cx"]
                cy = per_tree_transforms[tree]["cy"]

                # Create the camera intrinsics matrix
                K = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

                output_numerator, len_label_data = convert_frame(
                    (frame_i, frame), tree, K, 
                    cloud_cloud_dist, min_mask_size, min_view_fraction,
                    input_images_folder, input_depth_folder, output_folder,
                    tree_fruits_dict,
                    split_label_target, split_label_count,
                    max_background_depth, color_threshold,
                )

                if output_numerator is not None:
                    split_output_numerators += [output_numerator]

                split_label_count += len_label_data

                frame_i += 1

        else:            
            n_cores = int(os.cpu_count())
            with ThreadPool(n_cores) as pool:
                split_output_numerators = list(pool.starmap(
                    bulk_convert_frames, 
                    [
                        (
                            iterator_w_frame, trees_dict, per_tree_transforms,
                            cloud_cloud_dist, min_mask_size, min_view_fraction,
                            input_images_folder, input_depth_folder, output_folder,
                            max_background_depth, color_threshold,
                        ) for iterator_w_frame in enumerate(split_frames)
                    ]
                ))

        # Make split based on variables
        with open(os.path.join(output_folder, data_split+"_data_idx.txt"), "a") as f:
            for f_id in split_output_numerators:
                if f_id is not None:
                    f.write(f"{f_id}\n")

        

if __name__ == "__main__":
    main()