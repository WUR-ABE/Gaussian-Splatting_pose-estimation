import json
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from scipy.ndimage.measurements import label
from scipy.ndimage import maximum_filter, minimum_filter, binary_dilation
import os
import yaml
import sys

def load_apples_pcd(directory):
    # Initialize empty arrays to store the 3D points and colors
    calyx = np.zeros((0, 5))
    center = np.zeros((0, 5))
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
    return points_3d_tot, calyx, center, category_id_list



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


def calculate_transform_matrix(frame):    
    transform_matrix = np.array(frame["transform_matrix"])
    transform_matrix = np.linalg.inv(transform_matrix)  # Invert the transformation matrix
    # Transform 3D points to camera coordinate system
    return transform_matrix
    
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


def convert_apple_pcd_to_depth_map(points_3d, w, h, apple_id):    

    apple_pcd_cam = np.zeros((h, w), dtype=np.float32)
    selected_points = points_3d[points_3d['apple_id'] == apple_id]
    x, y, depth = selected_points['x'].astype(int), selected_points['y'].astype(int), selected_points['z']
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
    
    with np.errstate(invalid='ignore'):
        pixel_distance = np.abs(depth_image - apple_pcd_cam)
    # cv2.imshow('pixel_distance', pixel_distance)
    distance_mask = pixel_distance < cloud_cloud_dist
    distance_mask = distance_mask.astype(np.uint8)
    # # Find contours of the binary mask
    # contours, _ = cv2.findContours(distance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Fill the contours to create a filled mask
    # mask = np.zeros_like(distance_mask)
    # cv2.drawContours(mask, contours, -1, color=1, thickness=cv2.FILLED)

    
    return distance_mask

def plot_3d_vector_in_2d(point1, point2, apple_id):
    # Calculate the direction vector and normalize it to a unit vector
    direction_vector = np.array(point2) - np.array(point1)
    magnitude = np.linalg.norm(direction_vector)
    unit_vector = direction_vector / magnitude if magnitude != 0 else np.zeros(3)
    start = (point1[:2]).astype(int)
    end = (point2[:2]).astype(int)  # Use x and y components only

    return start, end, unit_vector

def depth_mask_image(depth_image, image, min_dist, max_dist):
    # Black out pixels outside the depth range in color image
    depth_mask = (depth_image < min_dist) | (depth_image > max_dist) | (depth_image == np.inf)
    image[depth_mask] = 0
    return image

def create_bbox(mask):
    # find first column with a white pixel
    # Find rows and columns where there are 1's
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Get the indices of the first and last True values
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]
    return col_end, row_end, col_start, row_start

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):  # Convert NumPy integers to Python integers
        return int(obj)
    elif isinstance(obj, np.floating):  # Convert NumPy floats to Python floats
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")

def main():
    params = yaml.safe_load(open("params.yaml"))["project_pcd_to_patches"]
    min_dist = params["min_dist"]
    max_dist = params["max_dist"]
    min_mask_size = params["min_mask_size"]
    min_view_fraction = params["min_view_fraction"]
    border_check_pixel_count = params["border_check_pixel_count"]
    # Path = "C:/Users/Bram/OneDrive - Wageningen University & Research/Data thesis/Foto's + poses + GS + pc/Foto's rij 4 goed (undistorted)+ refined camera poses + goede GS + gesegmenteerde appels in GS/"
    tree = params["tree"]
    cloud_cloud_dist = params["cloud_cloud_dist"]

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython create_annotions.py image-folder depth-image-folder apple-pc-folder annotations-output-folder\n")
        sys.exit(1)

    input_images = sys.argv[1]
    depth_images = sys.argv[2]
    input_apples = sys.argv[3]
    output_annotations = sys.argv[4]

    # Load transforms.json
    with open(os.path.join(input_images, "transforms.json")) as f:
        data = json.load(f)

    # Camera intrinsic parameters from transforms.json
    w, h = int(data["w"]), int(data["h"])
    fl_x, fl_y = data["fl_x"], data["fl_y"]
    cx, cy = int(data["cx"]), int(data["cy"])

    coco_categories = yaml.safe_load(open("params.yaml"))["coco_data_categories"]

    coco_data = {
        "images": [],        # List of image metadata
        "annotations": [],   # List of annotation data
        "categories": coco_categories  # List of categories [id, name, supercategory]
    }

    annotation_id = 1  # Unique ID for each annotation
    image_id = 1       # Unique ID for each image



    points_3d, calyx, center, category_id_list = load_apples_pcd(input_apples)

    for frame_i, frame in enumerate(data["frames"]):
        # print(f"Processing frame {frame_i + 1}/{len(data['frames'])}...")
        # Load image corresponding to the current frame
        # image = cv2.imread(input_images + "/" + frame["file_path"])
        transform_matrix = calculate_transform_matrix(frame)
        points_3d_cam = convert_3d_points_to_camera_coordinates(points_3d, transform_matrix, fl_x, fl_y, cx, cy, h, w)
        points_3d_calyx = convert_3d_points_to_camera_coordinates(calyx, transform_matrix, fl_x, fl_y, cx, cy, h, w)
        points_3d_center = convert_3d_points_to_camera_coordinates(center, transform_matrix, fl_x, fl_y, cx, cy, h, w)
        points_3d_cam = filter_points_in_front_of_camera(points_3d_cam, w, h, min_dist, max_dist)
        # points_3d_calyx = filter_points_in_front_of_camera(points_3d_calyx, w, h, min_dist, max_dist)
        # points_3d_center = filter_points_in_front_of_camera(points_3d_center, w, h, min_dist, max_dist)

        # Add this frame's image to the `images` list
        coco_data["images"].append({
            "id": f"{tree}"+f"{image_id:04d}",
            "file_name": frame["file_path"],
            "width": int(w),
            "height": int(h)
        }) 
        
        # # Step 1: Find common apple_id values across all three DataFrames
        # common_ids = set(points_3d_cam['apple_id']).intersection(points_3d_calyx['apple_id']).intersection(points_3d_center['apple_id'])

        # # Step 2: Filter each DataFrame to keep only rows with common apple_id values
        # points_3d_cam = points_3d_cam[points_3d_cam['apple_id'].isin(common_ids)]
        # points_3d_calyx = points_3d_calyx[points_3d_calyx['apple_id'].isin(common_ids)]
        # points_3d_center = points_3d_center[points_3d_center['apple_id'].isin(common_ids)]

        # print('apple_ids in frame', points_3d_cam['apple_id'].unique())
          
        if not points_3d_cam.empty:
            combined_mask = np.ones((h, w), dtype=np.uint8)
            combined_id_mask = np.zeros((h, w), dtype=np.uint8)
            # print('number of apples in frame', np.unique(points_3d_cam['apple_id']))
            depth_image = load_depth(depth_images, frame, max_dist)
            for i in np.unique(points_3d_cam['apple_id']):
                # tick = time.perf_counter()
                apple_depth = convert_apple_pcd_to_depth_map(points_3d_cam, w, h, i)
                # tock = time.perf_counter()
                # print('time to create depth map', tock-tick)
                mask = create_apple_mask(apple_depth, depth_image, cloud_cloud_dist)
                # tick = time.perf_counter()
                # print('time to create mask', tick-tock)
                # print('apple_id', i, 'points in mask', np.sum(mask), 'total points', np.sum(points_3d_cam['apple_id'] == i))	
                if np.sum(points_3d_cam['apple_id'] == i) * min_view_fraction > np.sum(mask) or np.sum(mask) < min_mask_size: # Check if more than 20% of the original apple pcd is in the view of the camera
                    pass # print('apple mask is too small, or apple is not in frame')
                    
                else:
                    # cv2.line(image, (int(points_3d_calyx[points_3d_calyx['apple_id'] == i]['x']), int(points_3d_calyx[points_3d_calyx['apple_id'] == i]['y'])),
                    #          (int(points_3d_center[points_3d_center['apple_id'] == i]['x']), int(points_3d_center[points_3d_center['apple_id'] == i]['y'])),
                    #          (0, 255, 0), 10)

                        
                    x1, y1, x0, y0 = create_bbox(mask)

                    # Check if the bounding box touches the image borders
                    is_border = (x0 <= border_check_pixel_count or 
                                 y0 <= border_check_pixel_count or 
                                 x1 >= w-border_check_pixel_count or 
                                 y1 >= h-border_check_pixel_count)
                    # print('bbox', x0, y0, x1, y1)
                    # print('border', is_border)

                    # Assign category_id based on border contact
                    if category_id_list[int(i)-1] == 3:
                        category_id = 3
                    else:
                        category_id = 2 if is_border else 1

                    if category_id == 1:
                        start, end, unit_vector  = plot_3d_vector_in_2d((points_3d_center[points_3d_center['apple_id'] == i][['x', 'y', 'z']].values[0]),
                                                                    (points_3d_calyx[points_3d_calyx['apple_id'] == i][['x', 'y', 'z']].values[0]), i)
                        center_coord = center[center['apple_id'] == i][['x', 'y', 'z']].values[0]
                        unit_vector = unit_vector.tolist()
                        center_coord = center_coord.tolist()
                    else:
                        center_coord = [0, 0, 0]
                        unit_vector = [0, 0, 0]
                        start = [0, 0]
                        end = [0, 0]
                    
                    if category_id == 3 or category_id == 2:
                        category_id = 1
                    

                    # Add annotation for this apple to the `annotations` list
                    coco_data["annotations"].append({
                        "id": int(annotation_id),
                        "image_id": f"{tree}"+f"{image_id:04d}",
                        "category_id": int(category_id),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],  # COCO format [x, y, width, height]
                        "area": (x1 - x0) * (y1 - y0),       # Area of the bounding box
                        "iscrowd": 0,                         # 0 = not crowded, typical for object detection
                        "apple_id": int(i),
                        "seq": f"tree_{tree}",
                        "unit_vector": unit_vector,
                        "location" : center_coord
                    })

                    # cv2.arrowedLine(image, start, end, (0, 255, 0), 5)
                    # if category_id == 2:
                    #     cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 5)
                    # elif category_id == 3:
                    #     cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 5)
                    # else:
                    #     cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 5)
                    combined_id_mask[mask == 1] = int(i)
                    combined_mask[mask == 1] = 2

                annotation_id += 1
            if np.count_nonzero(combined_id_mask) == 0:
                check = False   
        
        image_id += 1  # Increment image ID for the next frame
        # if(image_id > 3):
        #     break

    if tree == "full":
        # Ensure output directory exists
        os.makedirs(output_annotations, exist_ok=True)
        with open(output_annotations+"/coco_annotations.json", 'w') as f:
            json.dump(coco_data, f, indent=4, default=convert_to_serializable)

    print("COCO annotations saved to 'coco_annotations.json'")

if __name__ == "__main__":
    main()