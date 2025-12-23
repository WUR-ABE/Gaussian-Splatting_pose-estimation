
import numpy as np
import torch
import open3d as o3d
import yaml
import sys
import os
import cv2
import json
from multiprocessing.pool import ThreadPool
import copy

def convert_depth(
        depth_images_folder, images_folder, 
        point_cloud_folder, xyz_image_folder,
        tree, frame, K, max_background_depth,
        xyz_img_min_z, xyz_img_max_z,
        xyz_img_min_x, xyz_img_max_x,
        xyz_img_min_y, xyz_img_max_y,
        color_threshold):
    img_filename = frame["file_path"]
            
    # Load the depth image
    if tree == "global":
        depth_image = o3d.io.read_image(os.path.join(depth_images_folder, img_filename))
    else:
        depth_image = o3d.io.read_image(os.path.join(depth_images_folder, tree, img_filename))

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

    # Save the point cloud
    ply_filename = img_filename.replace(".png", ".ply")
    if tree == "global":
        ply_path = os.path.join(point_cloud_folder, ply_filename)
    else:
        ply_path = os.path.join(point_cloud_folder, tree, ply_filename)
    if not os.path.exists(os.path.dirname(ply_path)):
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    o3d.io.write_point_cloud(ply_path, rotated_pcd)

    # # Convert depth to point cloud
    # pcd_o3d_depth = o3d.geometry.PointCloud.create_from_depth_image(
    # depth_image, K, project_valid_depth_only=False, depth_scale=65535.0/max_background_depth, depth_trunc=0.96*max_background_depth)
    # xyz_image = np.asarray(pcd_o3d_depth.points).reshape(K.height, K.width, 3).astype(np.float32)

    # # Replace nan values with 0
    # xyz_image = np.nan_to_num(xyz_image)

    # # Find valid pixels
    # valid_pixels_red = np.asarray(color_image)[:,: ,0] >= color_threshold
    # valid_pixels_green = np.asarray(color_image)[:,: ,1] >= color_threshold
    # valid_pixels_blue = np.asarray(color_image)[:,: ,2] >= color_threshold
    # valid_pixels = np.logical_or(valid_pixels_red, np.logical_or(valid_pixels_green, valid_pixels_blue))

    # # Set invalid pixels to 0
    # xyz_image[~valid_pixels] = np.asarray([xyz_img_min_x, xyz_img_min_y, xyz_img_min_z])


    # # normalize depth map
    # zxy_image_normalized = np.zeros_like(xyz_image, dtype=np.uint16)
    # # Z values
    # zxy_image_normalized[:,:,0] = ((xyz_image[:,:,2] - xyz_img_min_z) / (xyz_img_max_z - xyz_img_min_z) * 65535).astype(np.uint16)
    # # X values
    # zxy_image_normalized[:,:,1] = ((xyz_image[:,:,0] - xyz_img_min_x) / (xyz_img_max_x - xyz_img_min_x) * 65535).astype(np.uint16)
    # # Y values
    # zxy_image_normalized[:,:,2] = ((xyz_image[:,:,1] - xyz_img_min_y) / (xyz_img_max_y - xyz_img_min_y) * 65535).astype(np.uint16)

    # # Save the XYZ image
    # xyz_filename = "depth_map_" + img_filename
    # if tree == "global":
    #     xyz_path = os.path.join(xyz_image_folder, xyz_filename)
    # else:
    #     xyz_path = os.path.join(xyz_image_folder, tree, xyz_filename)
    # # Make sure the folder exists
    # if not os.path.exists(os.path.dirname(xyz_path)):
    #     os.makedirs(os.path.dirname(xyz_path), exist_ok=True)
    # cv2.imwrite(xyz_path, zxy_image_normalized)

def main():

    # Load the parameters
    params = yaml.safe_load(open("params.yaml"))["convert_depth_data"]
    max_background_depth = params["max_background_depth"]
    xyz_img_min_z = params["xyz_img_z_min"]
    xyz_img_max_z = params["xyz_img_z_max"]
    xyz_img_min_x = params["xyz_img_x_min"]
    xyz_img_max_x = params["xyz_img_x_max"]
    xyz_img_min_y = params["xyz_img_y_min"]
    xyz_img_max_y = params["xyz_img_y_max"]
    color_threshold = params["color_threshold"]

    # Parameters
    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython convert_depth_data.py depth-images-folder images-folder point-cloud-folder xyz-image-folder\n")
        sys.exit(1)

    depth_images_folder = sys.argv[1]
    images_folder = sys.argv[2]
    point_cloud_folder = sys.argv[3]
    xyz_image_folder = sys.argv[4]

    if not os.path.exists(point_cloud_folder):
        os.makedirs(point_cloud_folder, exist_ok=True)
    if not os.path.exists(xyz_image_folder):
        os.makedirs(xyz_image_folder, exist_ok=True)

    # Set random seed
    torch.manual_seed(42)
    device = "cuda"

    # List the files in the depth images folder
    depth_images = os.listdir(depth_images_folder)
    all_transforms = {}

    # If there is a json file, load it
    if "transforms.json" in depth_images:
        with open(os.path.join(depth_images_folder, "transforms.json"), "r") as f:
            all_transforms["global"] = json.load(f)
    else:
        for tree in depth_images:
            # List files in the tree folder
            tree_files = os.listdir(os.path.join(depth_images_folder, tree))
            # If there is a json file, load it
            if "transforms.json" in tree_files:
                with open(os.path.join(depth_images_folder, tree, "transforms.json"), "r") as f:
                    all_transforms[tree] = json.load(f)
    
    for tree in all_transforms:
        tree_transforms = all_transforms[tree]

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
            
            n_cores = int(os.cpu_count()) - 2
            with ThreadPool(n_cores) as pool:
                pool.starmap(
                    convert_depth, 
                    [
                        (
                            depth_images_folder, images_folder, 
                            point_cloud_folder, xyz_image_folder,
                            tree, frame, K, max_background_depth,
                            xyz_img_min_z, xyz_img_max_z,
                            xyz_img_min_x, xyz_img_max_x,
                            xyz_img_min_y, xyz_img_max_y,
                            color_threshold
                        ) for frame in tree_transforms["frames"]
                    ]
                )
        except KeyError:
            print(f"Camera intrinsics different per frame for {tree}")
            
            # Get the camera intrinsics from the tree_transforms
            w = tree_transforms["w"]
            h = tree_transforms["h"]
            
            n_cores = int(os.cpu_count()) - 2
            with ThreadPool(n_cores) as pool:
                pool.starmap(
                    convert_depth, 
                    [
                        (
                            depth_images_folder, images_folder, 
                            point_cloud_folder, xyz_image_folder,
                            tree, frame, o3d.camera.PinholeCameraIntrinsic(
                                w, h, np.asarray(frame["intrinsic_matrix"])
                            ), max_background_depth,
                            xyz_img_min_z, xyz_img_max_z,
                            xyz_img_min_x, xyz_img_max_x,
                            xyz_img_min_y, xyz_img_max_y,
                            color_threshold
                        ) for frame in tree_transforms["frames"]
                    ]
                )



if __name__ == "__main__":
    main()
