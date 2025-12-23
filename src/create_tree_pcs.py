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
    tree_bboxes = yaml.safe_load(open("params.yaml"))["tree_bounding_boxes"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython create_tree_pcs.py complete-pc-folder trees-pc-folder\n")
        sys.exit(1)

    complete_pc_folder = sys.argv[1]
    trees_pc_folder = sys.argv[2]

    if not os.path.exists(trees_pc_folder):
        os.makedirs(trees_pc_folder)

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud(complete_pc_folder + "/dense_pc_full_colour.ply")
    points_3d = np.asarray(point_cloud.points)  # 3D points

    for tree in tree_bboxes:
        bbox = tree_bboxes[tree]
        translate = np.array(bbox["translation"]) 
        rotate = np.radians(bbox["rotation"])
        scale = np.array(bbox["scale"])

        # Bounding box transformation matrix
        bbox_transform = create_transformation_matrix(translate, rotate, scale)
        bbox_transform_inv = np.linalg.inv(bbox_transform)  # Inverse for transforming points

        # Transform points to bounding box local space
        ones = np.ones((points_3d.shape[0], 1))
        points_hom = np.hstack([points_3d, ones])  # Convert to homogeneous coordinates
        points_local = (bbox_transform_inv @ points_hom.T).T[:, :3]  # Transform points

        # Filter points inside the bounding box
        points_in_bbox = points_3d[is_within_unit_cube(points_local)]

        # Store filtered points
        tree_pc = o3d.geometry.PointCloud()
        tree_pc.points = o3d.utility.Vector3dVector(points_in_bbox)
        o3d.io.write_point_cloud(trees_pc_folder + f"/{tree}_pc.ply", tree_pc)
        


if __name__ == "__main__":
    main()