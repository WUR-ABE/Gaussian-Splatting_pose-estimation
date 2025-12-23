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
    if tree == "global":
        depth_image = o3d.io.read_image(os.path.join(depth_images_folder, img_filename.replace(".jpg", ".png")))
    else:
        depth_image = o3d.io.read_image(os.path.join(depth_images_folder, tree, img_filename.replace(".jpg", ".png")))

    # Load the color image
    color_img_filename = img_filename.replace(".png", ".jpg")
    if tree == "global":
        color_image = o3d.io.read_image(os.path.join(images_folder, color_img_filename))
    else:
        color_image = o3d.io.read_image(os.path.join(images_folder, tree, color_img_filename))

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
    max_background_depth, color_threshold
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
        return

    # Chec if the pc data is empty
    if len(pc_frame.points) == 0:
        return

    # Save the point cloud data
    pc_data_xyz = np.asarray(pc_frame.points)
    pc_data_rgb = np.asarray(pc_frame.colors)
    pc_data = np.hstack([pc_data_xyz, pc_data_rgb])
    dense_pc_file = os.path.join(output_folder, "depth", output_numerator + ".mat")
    sio.savemat(dense_pc_file, {"data": pc_data})

    ## Part 2: Load the camera data
    calib_rt = np.eye(3)
    calib = np.vstack([calib_rt.flatten(), K.intrinsic_matrix.T.flatten()])
    calib_file = os.path.join(output_folder, "calib", output_numerator + ".txt")
    np.savetxt(calib_file, calib)

    ## Part 3: Load the image data
    image = cv2.imread(os.path.join(input_images_folder, tree, frame["file_path"]))
    image_file = os.path.join(output_folder, "image", output_numerator + ".jpg")
    cv2.imwrite(image_file, image)

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

    label_file = os.path.join(output_folder, "label", output_numerator + ".txt")
    with open(label_file, "w") as f:
        for label in label_data:
            f.write(" ".join(str(x) for x in label) + "\n")

    return output_numerator

def main():
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["convert_randwijk_data_to_fresh"]
    min_mask_size = params["min_mask_size"]
    min_view_fraction = params["min_view_fraction"]
    cloud_cloud_dist = params["cloud_cloud_dist"]
    train_trees = params["train_trees"]
    val_trees = params["val_trees"]
    test_trees = params["test_trees"]

    depth_params = all_params["convert_depth_data"]
    max_background_depth = depth_params["max_background_depth"]
    color_threshold = depth_params["color_threshold"]
    

    tree_bbs = yaml.safe_load(open("params.yaml"))["tree_bounding_boxes"]

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython convert_randwijk_data_to_fresh.py image-folder input-depth-folder apple-pc-folder data-output-folder\n")
        sys.exit(1)

    input_images_folder = sys.argv[1]
    input_depth_folder = sys.argv[2]
    input_apples_folder = sys.argv[3]
    output_folder = sys.argv[4]

    # # Testing folders
    # input_images_folder = "data_disk/dvc_data/randwijk_row_4_patched_images_tree"
    # input_depth_folder =  "data_disk/dvc_data/randwijk_row_4_patched_depth_images_tree"
    # input_apples_folder = "data_disk/dvc_data/randwijk_row_4_individual_apples"
    # output_folder = "data_disk/testing"

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
    all_transforms = {}

    # If there is a json file, load it
    if "transforms.json" in images_list:
        with open(os.path.join(input_images_folder, "transforms.json"), "r") as f:
            all_transforms["global"] = json.load(f)
    else:
        for tree in images_list:
            # List files in the tree folder
            tree_files = os.listdir(os.path.join(input_images_folder, tree))
            # If there is a json file, load it
            if "transforms.json" in tree_files:
                with open(os.path.join(input_images_folder, tree, "transforms.json"), "r") as f:
                    all_transforms[tree] = json.load(f)


    tree_used_frames = {}
    
    for tree in all_transforms:
        if not ((tree in train_trees) or (tree in val_trees) or (tree in test_trees)):
            print(f"Skipping {tree}")
            continue
        tree_transforms = all_transforms[tree]

        # Get the fruit and calyx points
        tree_fruits_dict = trees_dict[tree]

        try:
            # Get the camera intrinsics from the tree_transforms
            w = tree_transforms["w"]
            h = tree_transforms["h"]
            fl_x = tree_transforms["fl_x"]
            fl_y = tree_transforms["fl_y"]
            cx = tree_transforms["cx"]
            cy = tree_transforms["cy"]

            # Create the camera intrinsics matrix
            K = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

            n_cores = int(os.cpu_count()) 
            with ThreadPool(n_cores) as pool:
                tree_used_frames[tree] = list(pool.starmap(
                    convert_frame, 
                    [
                        (
                            iterator_w_frame, tree, K, 
                            cloud_cloud_dist, min_mask_size, min_view_fraction,
                            input_images_folder, input_depth_folder, output_folder,
                            tree_fruits_dict,
                            max_background_depth, color_threshold,
                        ) for iterator_w_frame in enumerate(tree_transforms["frames"])
                    ]
                ))
        except KeyError:
            print(f"Camera intrinsics different per frame for {tree}")
            
            # Get the camera intrinsics from the tree_transforms
            w = tree_transforms["w"]
            h = tree_transforms["h"]
            
            n_cores = int(os.cpu_count())
            with ThreadPool(n_cores) as pool:
                tree_used_frames[tree] = list(pool.starmap(
                    convert_frame, 
                    [
                        (
                            iterator_w_frame, tree, o3d.camera.PinholeCameraIntrinsic(
                                w, h, np.asarray(iterator_w_frame[1]["intrinsic_matrix"])
                            ), 
                            cloud_cloud_dist, min_mask_size, min_view_fraction,
                            input_images_folder, input_depth_folder, output_folder,
                            tree_fruits_dict,
                            max_background_depth, color_threshold,
                        ) for iterator_w_frame in enumerate(tree_transforms["frames"])
                    ]
                ))


    if "global" in tree_used_frames.keys():
        # Make a simple train/val/test split
        frame_ids = tree_used_frames["global"]
        num_images = len(frame_ids)
        num_train = int(num_images * 0.8)
        num_val = int(num_images * 0.1) # Remaining 10% for test

        # Put train ids in a .txt file
        with open(os.path.join(output_folder, "train_data_idx.txt"), "w") as f:
            for i in range(num_train):
                f.write(f"{frame_ids[i]}\n")
        
        # Put val ids in a .txt file
        with open(os.path.join(output_folder, "val_data_idx.txt"), "w") as f:
            for i in range(num_train, num_train+num_val):
                f.write(f"{frame_ids[i]}\n")
        
        # Put test ids in a .txt file
        with open(os.path.join(output_folder, "test_data_idx.txt"), "w") as f:
            for i in range(num_train+num_val, num_images):
                f.write(f"{frame_ids[i]}\n")
    else: 
        # Make split based on variables

        for tree_id in tree_used_frames.keys():
            frame_ids = tree_used_frames[tree_id]

            # Put train ids in a .txt file, which can already exist
            if tree_id in train_trees:
                with open(os.path.join(output_folder, "train_data_idx.txt"), "a") as f:
                    for f_id in frame_ids:
                        if f_id is not None:
                            f.write(f"{f_id}\n")
            
            # Put val ids in a .txt file
            elif tree_id in val_trees:
                with open(os.path.join(output_folder, "val_data_idx.txt"), "a") as f:
                    for f_id in frame_ids:
                        if f_id is not None:
                            f.write(f"{f_id}\n")


            # Put test ids in a .txt file
            elif tree_id in test_trees:
                with open(os.path.join(output_folder, "test_data_idx.txt"), "a") as f:
                    for f_id in frame_ids:
                        if f_id is not None:
                            f.write(f"{f_id}\n")
            else:
                print(f"Invalid tree_id: {tree_id}")

if __name__ == "__main__":
    main()