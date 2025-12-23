import open3d as o3d
import os
import numpy as np
import pickle
import sys
import json
import configparser
import cv2
import transforms3d as t3d
import yaml

import mmcv

def write_labelcloud_config(vis_output_folder):
    config_labelcloud = configparser.ConfigParser()
    config_labelcloud['FILE'] = {
        "pointcloud_folder": "point_cloud",
        "label_folder": "labels",
        "class_definitions": "labels/_classes.json"
    }
    config_labelcloud['POINTCLOUD'] = {
        "point_size": "0.01",
        "colorless_color": "0.9, 0.9, 0.9",
        "colorless_colorize": "True",
        "std_translation": "0.1",
        "std_zoom": "0.01",
        "color_with_label": "True",
        "label_color_mix_ratio": "0.3",
    }
    config_labelcloud['LABEL'] = {
        "export_precision": "3",
        "std_boundingbox_length": "0.05",
        "std_boundingbox_width": "0.05",
        "std_boundingbox_height": "0.05",
        "std_translation": "0.01",
        "std_rotation": "0.5",
        "std_scaling": "0.001",
        "min_boundingbox_dimension": "0.01",
        "propagate_labels": "True"
    }
    config_labelcloud['USER_INTERFACE'] = {
        "z_rotation_only": "False",
        "show_floor": "True",
        "show_orientation": "True",
        "background_color": "100, 100, 100",
        "viewing_precision": "2",
        "near_plane": "0.1",
        "far_plane": "300",
        "keep_perspective": "False",
        "show_2d_image": "True",
        "delete_box_after_assign": "True"
    }

    with open(os.path.join(vis_output_folder, "point_cloud_results", "config.ini"), 'w') as f:
        config_labelcloud.write(f)

    classes_dict = {
        "classes": [
            {
                "name": "apple_gt",
                "id": 1,
                "color": "#ff2400"
            },
            {
                "name": "apple_pred",
                "id": 2,
                "color": "#7fc4ff"
            }
        ],
        "default": 1,
        "type": "object_detection",
        "format": "centroid_abs",
        "created_with": {
            "name": "labelCloud",
            "version": "1.1.1"
        }
    }

    with open(os.path.join(vis_output_folder, "point_cloud_results", config_labelcloud["FILE"]["class_definitions"]), 'w') as f:
        json.dump(classes_dict, f, indent=4)


def draw_depth_bbox3d_on_img(points_3d,
                             raw_img,
                             calibs,
                             Rt,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
            3d bbox in depth coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        calibs (dict): Camera calibration information, Rt and K.
        img_metas (dict): Used in coordinates transformation.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    num_bbox = points_3d.shape[0]

    K = calibs['K']

    # Transform corner points to cam frame
    points_3d_cam = points_3d @ Rt

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d_cam, K)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2)

    line_indices = ((0, 1), (0, 2), (0, 3), (1, 6), (1, 7), (2, 5), (2, 7),
                    (3, 5), (3, 6), (4, 5), (4, 6), (4, 7), 
                    (1, 4), (6, 7),) # Final two are diagonal lines for cross on top
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)

def point_cloud_from_obb(points_3d, color=(0, 255, 0), line_interpolation=100):
    """Create point cloud from 3D bounding box.

    Args:
        points_3d (numpy.array): 3D corner points of bounding box.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        line_interpolation (int, optional): Number of interpolation points
            between two corners. Default: 100.
    """
    pcd = o3d.geometry.PointCloud()

    line_indices = ((0, 1), (0, 2), (0, 3), (1, 6), (1, 7), (2, 5), (2, 7),
                    (3, 5), (3, 6), (4, 5), (4, 6), (4, 7), 
                    (1, 4), (6, 7),) # Final two are diagonal lines for cross on top

    num_bbox = points_3d.shape[0]
    for i in range(num_bbox):
        corners = points_3d[i]
        for start, end in line_indices:
            line = np.linspace(corners[start], corners[end], line_interpolation)
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(line)

            pcd += line_pcd

    # Put color in 0 to 1 range
    color = np.array(color) / 255.0

    # Set color
    pcd.paint_uniform_color(color)

    return pcd

def main():
    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython local_visualizer.py predictions-folder data-folder images-folder visualization-output-folder\n")
        sys.exit(1)

    # Set folder with visualizations
    predictions_pickle = sys.argv[1]
    data_folder = sys.argv[2]
    input_images_folder = sys.argv[3]
    vis_output_folder = sys.argv[4]

    # # Alternative for testing
    # predictions_pickle = "results/predictions/freshnet_model/results.pkl"
    # data_folder = "data_disk/dvc_data/papple_split"
    # input_images_folder = "data_disk/dvc_data/papple_trainval"
    # vis_output_folder = "visualizations/freshnet_model"

    # Make output folder if it doesn't exist
    os.makedirs(os.path.join(vis_output_folder, "point_cloud_results", "point_cloud"), exist_ok=True)
    os.makedirs(os.path.join(vis_output_folder, "point_cloud_results", "labels"), exist_ok=True)
    os.makedirs(os.path.join(vis_output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(vis_output_folder, "point_cloud_annotated"), exist_ok=True)


    Rt = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])



    # Write labelcloud config
    write_labelcloud_config(vis_output_folder)

    # Params
    params = yaml.safe_load(open("params.yaml"))["local_visualizer"]
    score_threshold = params["score_threshold"]

    # Load model results
    with open(predictions_pickle, 'rb') as f:
        model_results = pickle.load(f)

    # Load validation data GT
    with open(os.path.join(data_folder, "papple_infos_val.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    # Iterate each test data
    for i in range(len(model_results)):
        # Test file name
        test_file_name = test_data[i]['image']['image_idx']
        # Load dictionary containing point cloud
        pcd_dict = np.fromfile(os.path.join(os.path.split(data_folder)[0], test_data[i]['pts_path']))
        pcd_data = pcd_dict.reshape(-1, 6)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pcd_data[:, 3:6])

        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(vis_output_folder, "point_cloud_results", "point_cloud", f"{test_file_name}.ply"), pcd)

        # Get GT_boxes
        gt_3d_boxes = []
        gt_3d_boxes_corners = []
        if test_data[i]['annos']['gt_num'] > 0:
            for j in range(len(test_data[i]['annos']['gt_boxes_upright_depth'])):
                gt_box = test_data[i]['annos']['gt_boxes_upright_depth'][j]

                # Divide by 10 to get real values
                gt_box /= 10.0
                

                gt_box_dict = {
                    "name": "apple_gt",
                    "centroid": {
                        "x": float(gt_box[0]),
                        "y": float(gt_box[1]),
                        "z": float(gt_box[2])
                    },
                    "dimensions": {
                        "length": float(gt_box[3]),
                        "width": float(gt_box[4]),
                        "height": float(gt_box[5])
                    },
                    "rotations": {
                        "x": float(np.rad2deg(gt_box[6])),
                        "y": float(np.rad2deg(gt_box[7])),
                        "z": float(np.rad2deg(gt_box[8]))
                    }
                }
                gt_3d_boxes.append(gt_box_dict)

                # Make rotation matrix from euler angles
                R = t3d.euler.euler2mat(gt_box[6], gt_box[7], gt_box[8], 'sxyz')

                # Make o3d OBB
                obb = o3d.geometry.OrientedBoundingBox(center=gt_box[:3], R=R, extent=gt_box[3:6])

                # Get corners
                corners = np.array(obb.get_box_points())
                gt_3d_boxes_corners.append(corners)

        # Get predicted boxes
        pred_3d_boxes = []
        pred_3d_boxes_corners = []
        for j in range(len(model_results[i]['boxes_3d'])):
            if model_results[i]['scores_3d'][j] < score_threshold:
                continue
            pred_box = model_results[i]['boxes_3d'][j].tensor.flatten().cpu().numpy()

            # Divide by 10 to get real values
            pred_box[:6] /= 10.0

            pred_box_dict = {
                "name": "apple_pred",
                "centroid": {
                    "x": float(pred_box[0]),
                    "y": float(pred_box[1]),
                    "z": float(pred_box[2])
                },
                "dimensions": {
                    "length": float(pred_box[3]),
                    "width": float(pred_box[4]),
                    "height": float(pred_box[5])
                },
                "rotations": {
                    "x": float(np.rad2deg(pred_box[6])),
                    "y": float(np.rad2deg(pred_box[7])),
                    "z": float(np.rad2deg(pred_box[8]))
                }
            }
            pred_3d_boxes.append(pred_box_dict)

            # Make rotation matrix from euler angles
            R = t3d.euler.euler2mat(pred_box[6], pred_box[7], pred_box[8], 'sxyz')

            # Make o3d OBB
            obb = o3d.geometry.OrientedBoundingBox(center=pred_box[:3], R=R, extent=pred_box[3:6])

            # Get corners
            corners = np.array(obb.get_box_points())
            pred_3d_boxes_corners.append(corners)

        # Make array
        gt_3d_boxes_corners = np.array(gt_3d_boxes_corners)
        pred_3d_boxes_corners = np.array(pred_3d_boxes_corners)

        # Save GT and predicted boxes
        label_dict = {
            "folder": "point_cloud",
            "filename": f"{test_file_name}.ply",
            "path": str(os.path.join(vis_output_folder, "point_cloud_results", "point_cloud", f"{test_file_name}.ply")),
            "objects": gt_3d_boxes + pred_3d_boxes
        }

        with open(os.path.join(vis_output_folder, "point_cloud_results", "labels", f"{test_file_name}.json"), 'w') as f:
            json.dump(label_dict, f, indent=4)
        

        # Load image
        img = mmcv.imread(os.path.join(input_images_folder, test_data[i]['image']['image_path']), channel_order='rgb')

        # color, hex to RGB
        gt_color_hex = "#ff2400"
        pred_color_hex = "#7fc4ff"
        gt_color = tuple(int(gt_color_hex[i:i+2], 16) for i in (1, 3, 5))
        pred_color = tuple(int(pred_color_hex[i:i+2], 16) for i in (1, 3, 5))

        # Draw 3D boxes of GT and predicted
        if len(gt_3d_boxes_corners) > 0:
            drawn_img_gt = draw_depth_bbox3d_on_img(
                points_3d=gt_3d_boxes_corners, 
                raw_img=img,
                calibs=test_data[i]['calib'],
                Rt=Rt,
                color=gt_color,
                thickness=2
            )
            gt_pcd = point_cloud_from_obb(gt_3d_boxes_corners, color=gt_color)
            pcd += gt_pcd
        else:
            drawn_img_gt = img
        if len(pred_3d_boxes_corners) > 0:
            drawn_img = draw_depth_bbox3d_on_img(
                points_3d=pred_3d_boxes_corners, 
                raw_img=drawn_img_gt,
                calibs=test_data[i]['calib'],
                Rt=Rt,
                color=pred_color,
                thickness=2
            )
            pred_pcd = point_cloud_from_obb(pred_3d_boxes_corners, color=pred_color)
            pcd += pred_pcd
        else:
            drawn_img = drawn_img_gt

        # Convert RGB to BGR
        drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)

        # Save image
        mmcv.imwrite(drawn_img, os.path.join(vis_output_folder, "images", f"{test_file_name}.png"))

        # Save point cloud with bounding boxes
        o3d.io.write_point_cloud(os.path.join(vis_output_folder, "point_cloud_annotated", f"{test_file_name}_boxes.ply"), pcd)
    
    
if __name__ == '__main__':
    main()
