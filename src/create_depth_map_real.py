import numpy as np
import open3d as o3d
import cv2
import os
import json
import pandas as pd
from scipy.ndimage import minimum_filter
import sys
import yaml




# Function to create a transformation matrix
def create_transformation_matrix(translate, rotate, scale):
    tx, ty, tz = translate
    sx, sy, sz = scale
    rx, ry, rz = rotate

    # Translation matrix
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Combine rotation and scale into a single matrix
    RS = np.eye(4)
    RS[:3, :3] = R @ np.diag([sx, sy, sz])

    # Final transformation matrix
    return T @ RS


# Function to check if points are within a unit cube
def is_within_unit_cube(points):
    return np.all((points >= -1) & (points <= 1), axis=1)


def main():
    params = yaml.safe_load(open("params.yaml"))["create_depth_map_real"]
    min_dist = params["min_dist"]
    max_dist = params["max_dist"]

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython create_depth_map_real.py image-folder complete-pc-folder output-depth-folder\n")
        sys.exit(1)

    image_folder = sys.argv[1]
    complete_pc_folder = sys.argv[2]
    output_depth_folder = sys.argv[3]


    # Load transforms.json
    with open(image_folder + '/transforms.json') as f:
        data = json.load(f)

    # Camera intrinsic parameters from transforms.json
    w, h = data["w"], data["h"]
    fl_x, fl_y = data["fl_x"], data["fl_y"]
    cx, cy = data["cx"], data["cy"]
    transform = np.eye(4)  # Initialize transformation matrix
    transform_1 = np.asarray(data['applied_transform'])  # Transformation matrix
    transform[:3, :4] = transform_1  # Update transformation matrix

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud(complete_pc_folder + "/dense_pc_full_colour.ply")
    points_3d = np.asarray(point_cloud.points)  # 3D points

    # Output directory for depth maps
    os.makedirs(output_depth_folder, exist_ok=True)

    # Generate depth map for each camera
    for frame in data['frames']:

        img_path = frame["file_path"]
        img_idx = img_path.split('_')[1].split('.')[0]
        # Initialize the depth map with infinity (representing farthest depth initially)
        depth_map = np.full((h, w, 3), np.inf)


        transform_matrix_image = np.array(frame["transform_matrix"])
        transform_matrix = transform @ transform_matrix_image  # Update the transformation matrix

        transform_matrix = np.linalg.inv(transform_matrix)
        # Transform points to camera coordinates using the extrinsic matrix
        ones = np.ones((points_3d.shape[0], 1))
        point_3d_hom = np.hstack([points_3d, ones])

        # print('converting 3d points to camera coordinates')
        points_cam = (transform_matrix @ point_3d_hom.T).T  # Transform points
        x_proj = (points_cam[:, 0] / -points_cam[:, 2] * fl_x + cx).astype(int)
        y_proj = h - (points_cam[:, 1] / -points_cam[:, 2] * fl_y + cy).astype(int)
        points_cam[:, 2] = -points_cam[:, 2]
        points_cam = points_cam[:, :3]  # Remove the homogeneous coordinate
        # print('filtering points in front of the camera')

        points_in_frame = (x_proj >= 0) & (x_proj <= w-1) & \
                            (y_proj >= 0) & (y_proj <= h-1) & \
                            (points_cam[:, 2] >= min_dist) & (points_cam[:, 2] <= max_dist)  # Filter points in front of the camera
        # Project each point onto the image plane
        filtered_points = np.column_stack([
                                            x_proj[points_in_frame],         # Filtered x coordinates in pixels
                                            y_proj[points_in_frame],         # Filtered y coordinates in pixels
                                            points_cam[points_in_frame, 2],  # Filtered depth values
                                            points_cam[points_in_frame, 0],  # Filtered x coordinates in meters
                                            points_cam[points_in_frame, 1]   # Filtered y coordinates in meters
                                            ]) 


        # Convert to DataFrame for grouping
        df = pd.DataFrame(filtered_points, columns=['col0', 'col1', 'depth', 'x_cam', 'y_cam'])
        # print('removing points with the same x and y coordinates')
        # print('original points', df.shape)
        # Group by columns 0 and 1, keeping the row with the minimum value in column 2
        # Sort by 'col2' to ensure that the minimum value is at the top for each duplicate group
        df = df.sort_values(by='depth')

    # Drop duplicates based on columns 0 and 1, keeping the first occurrence (which has the minimum col2 after sorting)
        df = df.drop_duplicates(subset=["col0", "col1"], keep="first")

    # Convert back to a NumPy array if needed
        result = df.to_numpy()
        # print('start creating depth map')
        # print('filtered_points', result.shape)

        if result.shape[0] > 0:
            x, y, depth, x_m, y_m = result[:, 0].astype(int), result[:, 1].astype(int), result[:, 2], result[:, 3], result[:, 4]

                # Update depth map with the closest point at each pixel
            depth_map[y, x, 0] = depth
            depth_map[y, x, 1] = x_m
            depth_map[y, x, 2] = y_m
            #
            valid_x = depth_map[:, :, 1] != np.inf
            valid_y = depth_map[:, :, 2] != np.inf
            # Replace infinite values with zero or maximum depth for visualization
            depth_map[depth_map == np.inf] = 0
            x_min = np.min(depth_map[valid_x,1])
            x_max = np.max(depth_map[valid_x,1])
            y_min = np.min(depth_map[valid_y,2])
            y_max = np.max(depth_map[valid_y,2])

        
            # Save or display the depth map
            # normalize depth map z from 0 to 10 meters
            # normalize depth map x and y from -10 to 10 meters
            depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint16)
            depth_map_normalized[:,:,0] = (depth_map[:,:,0] / 10 * 65535).astype(np.uint16)
            depth_map_normalized[valid_x,1] = ((depth_map[valid_x,1] +10) / 
                                        (20) * 65535).astype(np.uint16)
            depth_map_normalized[valid_y,2] = ((depth_map[valid_y,2] + 10 ) / 
                                            (20) * 65535).astype(np.uint16)
            
            depth_map_path = f"{output_depth_folder}/depth_map_{img_idx}.png"
            cv2.imwrite(depth_map_path, depth_map_normalized)
            # print(f"Saved depth map for camera {img_idx} at {depth_map_path}")


if __name__ == "__main__":
    main()