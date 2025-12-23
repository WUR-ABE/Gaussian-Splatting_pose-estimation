import json
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from scipy.ndimage.measurements import label
from scipy.ndimage import maximum_filter, minimum_filter, binary_dilation
import scipy.io as sio
import os
import yaml
import sys

def load_apples_pcd(directory):
    # Initialize empty arrays to store the 3D points and colors
    calyx = np.zeros((0, 5))
    center = np.zeros((0, 5))
    extent = np.zeros((0, 5))
    corners = np.zeros((0, 5))
    points_3d_tot = np.zeros((0, 5))
    length = len(os.listdir(directory))
    category_id_list = np.zeros((length,1))

    for files in os.listdir(directory):

    # Load the segmented apple point cloud with RGB colors
        file_path = os.path.join(directory, files)
        
        parts = files.replace('.ply', '').split('_')
        apple_id = int(parts[1])    # Extracts the 1
        calyx_idx = int(parts[2])
        if len(parts) > 3:
            category_id_list[apple_id-1] = 3
 
        point_cloud = o3d.io.read_point_cloud(file_path)
        points_3d = np.asarray(point_cloud.points)  # XYZ coordinates
        #colors = np.asarray(point_cloud.colors)     # RGB colors (0-1 range)
        column_4 = np.ones((points_3d.shape[0], 1))           # Column of 1's
        column_5 = np.full((points_3d.shape[0], 1), apple_id)  # Column of apple_id's

    # Concatenate to form points_3d with the required columns
        points_3d_tot = np.append(points_3d_tot, np.hstack((points_3d, column_4, column_5)), axis=0)
        # Step 1: Compute the initial center as the mean of all points
        aabb = point_cloud.get_oriented_bounding_box() 
        center = np.append(center, np.hstack([aabb.get_center(), 1, apple_id]).reshape(1, -1), axis=0)  # Center of the bounding box
        extent = np.append(extent, np.hstack([aabb.extent, 1, apple_id]).reshape(1, -1), axis=0)  # Extent of the bounding box
        corners = np.append(corners, np.hstack([aabb.get_box_points(), np.ones((8, 1)), np.full((8, 1), apple_id)]), axis=0)  # Corners of the bounding box
        calyx = np.append(calyx, np.hstack([points_3d[calyx_idx], 1, apple_id]).reshape(1, -1), axis=0)  # Calyx point
        # Step 2: Compute the radius that ensures all points are enclosed
        # Iterate through each point to adjust the center and radius if needed
        # radius = 0.04  # Initial radius
        # for point in points_3d:
        #     distance = np.linalg.norm(point - center)
        #     if distance > radius:
        #         # Update the radius and center based on the farthest point
        #         radius = distance * 0.94  # Adjust the radius to be 90% of the farthest point
        #         # Adjust the center halfway between current center and the farthest point
        #         #center = center + (point - center) * 0.

        # Step 3: Create a mesh sphere with the computed radius and center
        # enclosing_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        # enclosing_sphere.translate(center)  # Move the sphere to the calculated center

    points_3d_tot = pd.DataFrame(points_3d_tot, columns=["x", "y", "z", "hom", "apple_id"])
    calyx = pd.DataFrame(calyx, columns=["x", "y", "z", "hom", "apple_id"])
    center = pd.DataFrame(center, columns=["x", "y", "z", "hom", "apple_id"])
    extent = pd.DataFrame(extent, columns=["x", "y", "z", "hom", "apple_id"])
    corners = pd.DataFrame(corners, columns=["x", "y", "z", "hom", "apple_id"])
    return points_3d_tot, calyx, center, category_id_list, corners



    # Iterate through each frame in transforms.json to project points onto images

def load_depth(depth_data_folder, frame, max_dist=4):
    img_path = frame["file_path"]
    img_idx = img_path.split('_')[1].split('.')[0]
    #depth_image = o3d.io.read_image(f"C:/Users/john_/OneDrive/Bureaublad/test/depth_maps/depth_map_{img_idx}.png")
    #load the image and convert integer values fo meters
    depth_path = depth_data_folder+f"/depth_map_{img_idx}.png"
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_image = (depth_image[:,:,0] / 65535 * 10).astype(np.float32)
    # print(min(depth_image.all()))
    depth_image[depth_image >= max_dist] = np.inf
    
    # # depth_image_copy = np.zeros_like(depth_image, dtype=np.float32)
    # # Apply minimum filter for each depth range
    # for min_depth, max_depth, filter_size in depth_ranges: # Depth ranges were removed
    #     # Create a mask for the current depth range
    #     # min_depth = min_depth/4*65535
    #     # max_depth = max_depth/4*65535
    #     mask_depth = ((depth_image > min_depth) & (depth_image < max_depth)).astype(np.uint8) * 255

    #     # Apply morphological operations to the mask
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))

    #     # Apply morphological closing to close small holes
    #     closed_image = cv2.morphologyEx(mask_depth, cv2.MORPH_CLOSE, kernel)

    #     # Apply morphological opening to remove small white spots
    #     opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

    #     depth_map = minimum_filter(depth_image, (filter_size), mode='nearest')

    #     opened_image = cv2.erode(opened_image, kernel)

    #     depth_image_copy = np.where(opened_image != 0, depth_map, depth_image_copy)
    #     depth_image_copy[depth_image_copy == 0] = np.inf
    #     #depth_image_copy = np.where(masked_depth == np.inf, filtered_region, masked_depth)

    #     # Update only the relevant parts of the result image
    return depth_image
    
def convert_3d_points_to_camera_coordinates(points_3d, transform_matrix, fl_x, fl_y, cx, cy, h, w):
    points_3d = points_3d.to_numpy()
    points_cam = (transform_matrix @ (points_3d[:,:4]).T ).T
    points_cam = np.hstack([points_cam, points_3d[:, 4].reshape(-1, 1)])  # Add the apple_id column
    # Project points onto the image plane
    points_cam[:, 0] = (points_cam[:, 0] / -points_cam[:, 2] * fl_x + cx).astype(int)
    points_cam[:, 1] = h - (points_cam[:, 1] / -points_cam[:, 2] * fl_y + cy).astype(int)
    points_cam[:, 2] = -points_cam[:, 2]
    
    points_cam = pd.DataFrame(points_cam, columns=["x", "y", "z", "hom", "apple_id"])
    return points_cam

def filter_points_in_front_of_camera(df, w, h, min_dist, max_dist, pad=0):
    w_min, w_max = pad, w - pad
    h_min, h_max = pad, h - pad
    out_of_range = (
        (df['x'] <= 0) | (df['x'] >= w) |
        (df['y'] <= 0) | (df['y'] >= h) |
        (df['z'] < min_dist) | (df['z'] > max_dist))

    df = df[~out_of_range]
    out_of_range = df[
        (df['x'] < w_min) | (df['x'] > w_max) |
        (df['y'] < h_min) | (df['y'] > h_max)
        ]['apple_id'].unique()
    df = df[~df['apple_id'].isin(out_of_range)]
    

    # Group by columns 0 and 1, keeping the row with the minimum value in column 2
    # Sort by 'col2' to ensure that the minimum value is at the top for each duplicate group
    df = df.sort_values(by="z")
    # Drop duplicates based on columns 0 and 1, keeping the first occurrence (which has the minimum col2 after sorting)
    df = df.drop_duplicates(subset=["x", "y", "apple_id"], keep="first")
    return df

def convert_3d_points_to_camera_frame_and_filter(points_3d, transform_matrix, fl_x, fl_y, cx, cy, h, w, min_dist, max_dist, filter=True):
    points_3d_hom = np.hstack([points_3d[:,:3], np.ones((points_3d.shape[0], 1))])
    points_cam = (transform_matrix @ (points_3d_hom).T ).T
    points_cam = np.hstack([points_cam[:,:3], points_3d[:, 3:]])  # Add the color columns
    # Project points onto the image plane
    points_cam_x = ((points_cam[:, 0] / -points_cam[:, 2] * fl_x).astype(int)) / w
    points_cam_y = (h - (points_cam[:, 1] / -points_cam[:, 2] * fl_y).astype(int)) / w
    points_cam_z = -points_cam[:, 2] * fl_x / w
    points_cam = np.hstack([points_cam_x.reshape(-1, 1), points_cam_y.reshape(-1, 1), points_cam_z.reshape(-1, 1), points_cam])
    
    points_df = pd.DataFrame(points_cam, columns=["cx", "cy", "cz", "x", "y", "z", "r", "g", "b"])

    in_range = (
        (points_df['cx'] >= 0) & 
        (points_df['cx'] <= 1) &
        (points_df['cy'] >= 0) & 
        (points_df['cy'] <= h/w) &
        (points_df['cz'] > min_dist) &
        (points_df['cz'] < max_dist))
    
    if filter:
        filtered_points = points_df[in_range]
        filtered_points = filtered_points.sort_values(by="cz")
        filtered_points = filtered_points.drop_duplicates(subset=["cx", "cy"], keep="first")
    else: 
        filtered_points = points_df
    

    return np.asarray(filtered_points[["cx", "cy", "cz", "r", "g", "b"]])

def convert_apple_pcd_to_depth_map(points_3d, w, h, apple_id):    

    apple_pcd_cam = np.zeros((h, w), dtype=np.float32)
    selected_points = points_3d[points_3d['apple_id'] == apple_id]
    x, y, depth = selected_points['cx'].astype(int), selected_points['cy'].astype(int), selected_points['cz']
        # Update depth map with the closest point at each pixel
   
    apple_pcd_cam[y, x] = depth
    apple_pcd_cam[apple_pcd_cam == 0] = np.inf

    # # Apply minimum filter for each depth range
    # mask = np.zeros_like(apple_pcd_cam, dtype=np.uint8)
    # mask[apple_pcd_cam != 0] = 255
    # apple_pcd_cam_copy = apple_pcd_cam.copy()
    # # Apply minimum filter for each depth range
    # for min_depth, max_depth, filter_size in depth_ranges: # Depth ranges were removed
    #     # Create a mask for the current depth range
    #     # min_depth = min_depth/4*65535
    #     # max_depth = max_depth/4*65535
    #     mask_depth = ((apple_pcd_cam > min_depth) & (apple_pcd_cam < max_depth)).astype(np.uint8) * 255
        
    #     # Apply morphological operations to the mask
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))

    #     # Apply morphological closing to close small holes
    #     closed_image = cv2.morphologyEx(mask_depth, cv2.MORPH_CLOSE, kernel)

    #     # Apply morphological opening to remove small white spots
    #     opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

    #     filtered_depth = minimum_filter(apple_pcd_cam, (filter_size), mode='nearest')

    #     apple_pcd_cam_copy = np.where(opened_image != 0, filtered_depth, apple_pcd_cam_copy)

        #depth_image_copy = np.where(masked_depth == np.inf, filtered_region, masked_depth)

        # Update only the relevant parts of the result image

    
    # apple_pcd_img = cv2.resize(apple_pcd_cam_copy, (1600, 1200))
    # cv2.imshow('apple_pcd', apple_pcd_cam_copy)
    return apple_pcd_cam

def create_apple_mask(apple_pcd_cam, depth_image, cloud_cloud_dist):

    copy_apple_pcd_cam = apple_pcd_cam.copy()
    
    # Create a mask column initialized to 0
    copy_apple_pcd_cam['copy_index'] = copy_apple_pcd_cam.index

    # Merge apple_pcd_cam and depth_image on 'cx' and 'cy' columns
    merged_df = pd.merge(copy_apple_pcd_cam, depth_image, on=['cx', 'cy'], suffixes=('', '_depth'))

    # Update the mask where cz is less than cz_depth + cloud_cloud_dist
    merged_df['mask'] = (merged_df['cz'] < (merged_df['cz_depth'] + cloud_cloud_dist)).astype(int)
    
    new_apple_pcd_cam = merged_df[['cx', 'cy', 'mask', 'copy_index']]
    # Return 'copy_index' as index
    new_apple_pcd_cam = new_apple_pcd_cam.set_index('copy_index')
    
    return new_apple_pcd_cam

def plot_3d_vector_in_2d(point1, point2, apple_id):
    # Calculate the direction vector and normalize it to a unit vector
    direction_vector = np.array(point2) - np.array(point1)
    magnitude = np.linalg.norm(direction_vector)
    unit_vector = direction_vector / magnitude if magnitude != 0 else np.zeros(3)
    start = (point1[:2]).astype(int)
    end = (point2[:2]).astype(int)  # Use x and y components only

    return start, end, unit_vector

def create_bbox(mask):
    # find lowest cx and cy with a positive mask
    min_cx = mask['cx'][mask['mask'] == 1].min()
    min_cy = mask['cy'][mask['mask'] == 1].min()
    # find highest cx and cy with a positive mask
    max_cx = mask['cx'][mask['mask'] == 1].max()
    max_cy = mask['cy'][mask['mask'] == 1].max()
    
    return max_cx, max_cy, min_cx, min_cy

def main():
    params = yaml.safe_load(open("params.yaml"))["convert_randwijk_to_fresh"]
    min_dist = params["min_dist"]
    max_dist = params["max_dist"]
    min_mask_size = params["min_mask_size"]
    min_view_fraction = params["min_view_fraction"]
    border_check_pixel_count = params["border_check_pixel_count"]
    cloud_cloud_dist = params["cloud_cloud_dist"]

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython create_annotions.py image-folder dense-pc-folder apple-pc-folder data-output-folder\n")
        sys.exit(1)

    input_images = sys.argv[1]
    dense_pc = sys.argv[2]
    input_apples = sys.argv[3]
    output_folder = sys.argv[4]

    # Load transforms.json
    with open(os.path.join(input_images, "transforms.json")) as f:
        data = json.load(f)

    # Camera intrinsic parameters from transforms.json
    w, h = int(data["w"]), int(data["h"])
    fl_x, fl_y = data["fl_x"], data["fl_y"]
    cx, cy = int(data["cx"]), int(data["cy"])
    # Intrinsics matrix
    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

    applied_transform = np.eye(4)  # Initialize transformation matrix
    transform_1 = np.asarray(data['applied_transform'])  # Transformation matrix
    applied_transform[:3, :4] = transform_1  # Update transformation matrix

    # Load input apples
    points_3d, calyx, center, category_id_list, extent = load_apples_pcd(input_apples)

    # Load dense point cloud
    dense_pc_path = os.path.join(dense_pc, "dense_pc_full_colour.ply")
    point_cloud = o3d.io.read_point_cloud(dense_pc_path)
    dense_pc_points = np.asarray(point_cloud.points)  # XYZ coordinates
    dense_pc_colors = np.asarray(point_cloud.colors)     # RGB colors (0-1 range)
    # Combine the XYZ coordinates and RGB colors into a single array
    dense_pc_data = np.hstack([dense_pc_points, dense_pc_colors])

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

    used_frames = []

    # Iterate over each frame in the dataset
    for frame_i, frame in enumerate(data["frames"]):
        transform_matrix_camera = np.array(frame["transform_matrix"])
        transform_matrix = np.linalg.inv(applied_transform @ transform_matrix_camera)

        dense_pc_data_in_cam = convert_3d_points_to_camera_frame_and_filter(dense_pc_data, transform_matrix, fl_x, fl_y, cx, cy, h, w, min_dist, max_dist)
        
        if dense_pc_data_in_cam.shape[0] == 0:
            continue
        used_frames.append(frame_i)
        
        dense_pc_file = os.path.join(output_depth, f"{frame_i:06d}.mat")
        sio.savemat(dense_pc_file, {"data": dense_pc_data_in_cam})

        calib_rt = np.eye(3)
        calib = np.vstack([calib_rt.flatten(), K.flatten()])
        calib_file = os.path.join(output_calib, f"{frame_i:06d}.txt")
        np.savetxt(calib_file, calib)

        image = cv2.imread(os.path.join(input_images, frame["file_path"]))
        image_file = os.path.join(output_images, f"{frame_i:06d}.jpg")
        cv2.imwrite(image_file, image)

        label_data = []

        points_3d_w_fake_color = np.hstack([points_3d.to_numpy(), np.ones((points_3d.shape[0], 1))])
        calyx_w_fake_color = np.hstack([calyx.to_numpy(), np.ones((calyx.shape[0], 1))])
        center_w_fake_color = np.hstack([center.to_numpy(), np.ones((center.shape[0], 1))])
        extent_w_fake_color = np.hstack([extent.to_numpy(), np.ones((extent.shape[0], 1))])

        # transform_matrix_extent = transform_matrix.copy()
        # transform_matrix_extent[:3, 3] = 0
        
        array_points_3d_cam = convert_3d_points_to_camera_frame_and_filter(points_3d_w_fake_color, transform_matrix, fl_x, fl_y, cx, cy, h, w, min_dist, max_dist)
        array_points_3d_calyx = convert_3d_points_to_camera_frame_and_filter(calyx_w_fake_color, transform_matrix, fl_x, fl_y, cx, cy, h, w, min_dist, max_dist, filter=False)
        array_points_3d_center = convert_3d_points_to_camera_frame_and_filter(center_w_fake_color, transform_matrix, fl_x, fl_y, cx, cy, h, w, min_dist, max_dist, filter=False)
        array_points_3d_extent = convert_3d_points_to_camera_frame_and_filter(extent_w_fake_color, transform_matrix, fl_x, fl_y, cx, cy, h, w, min_dist, max_dist, filter=False)

        points_3d_cam = pd.DataFrame(array_points_3d_cam, columns=["cx", "cy", "cz", "hom", "apple_id", "int"])
        points_3d_calyx = pd.DataFrame(array_points_3d_calyx, columns=["cx", "cy", "cz", "hom", "apple_id", "int"])
        points_3d_center = pd.DataFrame(array_points_3d_center, columns=["cx", "cy", "cz", "hom", "apple_id", "int"])
        points_3d_extent = pd.DataFrame(array_points_3d_extent, columns=["cx", "cy", "cz", "hom", "apple_id", "int"])

        if not points_3d_cam.empty:
            combined_mask = np.ones((h, w), dtype=np.uint8)
            combined_id_mask = np.zeros((h, w), dtype=np.uint8)

            df_dense_pc_data_in_cam = pd.DataFrame(dense_pc_data_in_cam, columns=["cx", "cy", "cz", "r", "g", "b"])

            for i in np.unique(points_3d_cam['apple_id']):
                apple_depth = points_3d_cam[points_3d_cam['apple_id'] == i]
                mask = create_apple_mask(apple_depth, df_dense_pc_data_in_cam, cloud_cloud_dist)

                if len(apple_depth) * min_view_fraction > np.sum(mask['mask']) or np.sum(mask['mask']) < min_mask_size:
                    continue

                cx1, cy1, cx0, cy0 = create_bbox(mask)
                x1, y1, x0, y0 = int(cx1 * w), int(cy1 * w), int(cx0 * w), int(cy0 * w)

                start, end, unit_vector = plot_3d_vector_in_2d(
                    points_3d_center[points_3d_center['apple_id'] == i][['cx', 'cy', 'cz']].values[0],
                    points_3d_calyx[points_3d_calyx['apple_id'] == i][['cx', 'cy', 'cz']].values[0], i
                )
                center_coord = points_3d_center[points_3d_center['apple_id'] == i][['cx', 'cy', 'cz']].values[0].tolist()
                unit_vector = unit_vector.tolist()
                corner_points = points_3d_extent[points_3d_extent['apple_id'] == i][['cx', 'cy', 'cz']]
                cx_min = corner_points['cx'].min()
                cx_max = corner_points['cx'].max()
                cy_min = corner_points['cy'].min()
                cy_max = corner_points['cy'].max()
                cz_min = corner_points['cz'].min()
                cz_max = corner_points['cz'].max()
                extents = [
                    cx_max - cx_min, 
                    cy_max - cy_min,
                    cz_max - cz_min,
                ]
                

                apple_label = [
                    "apple", x0, y0, x1, y1, center_coord[0], center_coord[1], center_coord[2],
                    extents[0], extents[1], extents[2], 0.0,
                    np.arctan2(unit_vector[2], -unit_vector[0]),
                    np.arctan2(unit_vector[1], -unit_vector[0]), 1.0
                ]

                label_data.append([apple_label])

            if np.count_nonzero(combined_id_mask) == 0:
                continue
        
        label_file = os.path.join(output_labels, f"{frame_i:06d}.txt")
        with open(label_file, "w") as f:
            for label in label_data:
                f.write(" ".join(str(x) for x in label[0]) + "\n")

    # Make a simple train/val/test split
    num_images = len(used_frames)
    num_train = int(num_images * 0.8)
    num_val = int(num_images * 0.1) # Remaining 10% for test

    # Put train ids in a .txt file
    with open(os.path.join(output_folder, "train_data_idx.txt"), "w") as f:
        for i in range(num_train):
            f.write(f"{used_frames[i]}\n")
    
    # Put val ids in a .txt file
    with open(os.path.join(output_folder, "val_data_idx.txt"), "w") as f:
        for i in range(num_train, num_train+num_val):
            f.write(f"{used_frames[i]}\n")
    
    # Put test ids in a .txt file
    with open(os.path.join(output_folder, "test_data_idx.txt"), "w") as f:
        for i in range(num_train+num_val, num_images):
            f.write(f"{used_frames[i]}\n")


if __name__ == "__main__":
    main()