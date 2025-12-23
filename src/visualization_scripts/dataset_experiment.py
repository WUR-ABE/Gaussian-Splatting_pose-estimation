#%% Imports
import open3d as o3d
import numpy as np
import scipy.io as sio
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from multiprocessing.pool import ThreadPool
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

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

darkest_flare_color = sns.color_palette("flare", 6).as_hex()[-1]


#%% Load results of dataset size experiment

experiment_path = "data_disk/testing/testing_dataset/"

dict_results = {}
for experiment_name in os.listdir(experiment_path):
    if "." in experiment_name:
        continue
    if "old" in experiment_name:
        continue
    for file_name in os.listdir(os.path.join(experiment_path, experiment_name)):
        if file_name.endswith(".txt"):
            with open(os.path.join(experiment_path, experiment_name, file_name), "r") as f:
                s = f.read()
                lines = s.split("\n")

            results = {}
            for line in lines[:-1]:
                vis_dict = json.loads(line.replace("\'null\'", "null").replace("\'", "\""))
                # Convert each entry except "visibility_lower_bound" and all AP's to arrays
                for key in vis_dict.keys():
                    if key != "visibility_lower_bound" and key != "gt_counts" and "AP_" not in key:
                        vis_dict[key] = np.asarray(vis_dict[key], dtype=float)
                
                # Add splat and real fraction to vis_dict
                vis_dict["real_fraction"] = float(experiment_name.split("_")[3])/1000
                vis_dict["splat_fraction"] = float(experiment_name.split("_")[5])/1000
                
                results[vis_dict["visibility_lower_bound"]] = vis_dict

            dict_results[experiment_name] = results

#%% Evaluation functions
import torch

class plane():
    def __init__(self,p0,n):
        self.p0 = p0
        self.n = n/np.linalg.norm(n)
        
    def __init__(self,p0,p1,p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.v = np.stack([p1-p0,p2-p0]).T
        n = np.cross(self.v[:,0],self.v[:,1])
        self.n = n/np.linalg.norm(n)
    
    def intersect_lines(self, lines, eps=1e-8):
        """Finds intersections with lines given by tuples"""
        m = lines[:,1]-lines[:,0] # gradient
        dot = self.n @ m.T # angles between plane and lines

        valid = abs(dot) > eps # only non parallel lines have valid solutions

        t = (self.n @ (self.p0 - lines[valid,0]).T) / dot[valid] # line parameter
        intersections = lines[valid,0] + (m[valid]*t[..., np.newaxis])
        
        return intersections, t
    
    def project_points(self, points, check=True, eps = 1e-8):
        v = (points-self.p0).T
        dist = self.n @ v
        prj_points = points - self.n * dist[..., np.newaxis]
        
        if check:
            t = np.linalg.inv(self.v.T @ self.v) @ self.v.T @ v
            valid = (0-eps<=t[0]) & (t[0]<=1+eps) & (0-eps<=t[1]) & (t[1]<=1+eps)
            points = points[valid]
            prj_points = prj_points[valid]
            dist = dist[valid]
        return list(zip(abs(dist),points,prj_points))

class OBB():
    def __init__(self, T, dimensions):
        self.edges = [[0, 1], [1, 7], [7, 2], [2, 0], [3, 6], [6, 4],
                      [4, 5], [5, 3], [0, 3], [1, 6], [7, 4], [2, 5]]
        self.faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [4, 5, 6], [4, 5, 7], [4, 6, 7]]
        self.T = T
        self.dimensions = dimensions
        
        w,h,l = self.dimensions/2.0
        
        self.corners = np.array([[-0.5, -0.5, -0.5, 1],
                            [0.5, -0.5, -0.5, 1],
                            [-0.5, 0.5, -0.5, 1],
                            [-0.5, -0.5, 0.5, 1],
                            [0.5, 0.5, 0.5, 1],
                            [-0.5, 0.5, 0.5, 1],
                            [0.5, -0.5, 0.5, 1],
                            [0.5, 0.5, -0.5, 1]]).T

    def get_box_points(self):
        return (self.T @ self.corners)[:3].T
    
    def get_box_faces(self):
        cor = self.get_box_points()
        face_array = []
        for p0, p1, p2 in self.faces:
            f = plane(cor[p0], cor[p1], cor[p2])
            face_array.append(f)
        return face_array
    
    def get_box_edges(self):
        cor = self.get_box_points()
        edges = []
        for [e0, e1] in self.edges:
            edges.append([cor[e0], cor[e1]])
        return np.array(edges)
    
    def get_point_indices_within_bounding_box(self, points, eps=1e-10):
        temp = np.linalg.inv(self.T) @ np.vstack((points.T, np.ones(len(points))))
        return np.all(temp[:3] <= 0.5 + eps, 0) & np.all(temp[:3] >= -0.5 - eps, 0)

    def intersect_lines(self, lines, check=True, eps=1e-10):
        poi = np.empty([0, 3])
        for face in self.get_box_faces():
            inters, t = face.intersect_lines(lines, eps)
            poi = np.concatenate([poi, inters])

        if check:
            valid = self.get_point_indices_within_bounding_box(poi)
            poi = poi[valid]
        return poi

    def IoU_v(self, box2, eps=1e-10):
        from scipy.spatial import ConvexHull
        
        poi = self.get_box_points()
        poi = np.vstack((poi, box2.get_box_points()))
        
        edges = box2.get_box_edges()
        poi = np.vstack((poi, self.intersect_lines(edges, False)))
        
        edges = self.get_box_edges()
        poi = np.vstack((poi, box2.intersect_lines(edges, False)))

        valid = (self.get_point_indices_within_bounding_box(poi, eps=eps) &
                 box2.get_point_indices_within_bounding_box(poi, eps=eps))

        try:
            h = ConvexHull(poi[valid])
        except:
            return 0
        

        intersection = h.volume
        union = self.volume() + box2.volume() - intersection
        IoU = intersection / union
        return IoU

    def volume(self):
        p, r, d = self.get_prd()
        return np.prod(d)

    def get_prd(self):
        p = self.T[:3, 3]
        r = self.T[:3, :3]
        d = np.linalg.norm(r, axis=0)
        r = r / d
        return p, r, d
    
    # def pd(self, box2):
    #     """ returns the distance the centers of the boxes """
    #     p, r, d = self.get_prd()
    #     p2, r2, d2 = box2.get_prd()
    #     return positionDifference(p, p2)
    
def create_transformation_matrix(box_params):
    """
    Create a transformation matrix from the bounding box parameters.

    Args:
        box_params: A list or array containing the bounding box parameters.
                     Expected format: [x, y, z, w, h, l, roll, pitch, yaw].

    Returns:
        A 4x4 transformation matrix.
    """
    x, y, z, w, h, l, roll, pitch, yaw = box_params

    # Create a rotation matrix from roll, pitch, yaw
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Create the transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])
    
    return T

def vect_angle_calc(T1, T2):
    """Calculate the angle between two vectors rotated by rotation of T1 and T2."""
    vx1 = np.array([1.0, 0, 0])
    vx2 = np.array([1.0, 0, 0])
    vx1 = T1[:3, :3] @ vx1
    vx2 = T2[:3, :3] @ vx2
    return np.arccos(np.dot(vx1, vx2) / (np.linalg.norm(vx1) * np.linalg.norm(vx2)))

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return np.asarray([qw, qx, qy, qz])

def quat_angle_calc(rpy1, rpy2):
    """Calculate the quaternion angle between two rotation matrices."""
    q1 = euler_to_quaternion(rpy1[0], rpy1[1], rpy1[2])
    q2 = euler_to_quaternion(rpy2[0], rpy2[1], rpy2[2])
    return np.arccos(np.abs((q1 * q2).sum(-1)))

def euclid_dist_calc(c1, c2):
    return np.sqrt(sum((c1-c2)**2))


#%% Determine the ground truth visibility for each test image

# Load the txt file with the validation image ids
val_image_ids = np.loadtxt("data_disk/dvc_data/combined_randwijk_papple/val_data_idx.txt", dtype=str)

# Iterate the val image ids and load the corresponding label files to get the visibility
val_image_gts = {}
for val_image_id in val_image_ids:
    label_file = f"data_disk/dvc_data/real_randwijk_papple_test_labels/label/{val_image_id}.txt"
    with open(label_file, "r") as f:
        lines = f.readlines()
    
    image_visibilities = {}
    for line in lines:
        elements = line[:-1].split(" ")
        x = float(elements[5])
        y = float(elements[6])
        z = float(elements[7])
        d_x = float(elements[8])
        d_y = float(elements[9])
        d_z = float(elements[10])
        roll = float(elements[11])
        pitch = float(elements[12])
        yaw = float(elements[13])
        visibility = float(elements[15])
        apple_id = int(elements[16])
        image_visibilities[apple_id] = [
            [
                x, y, z, 
                d_x, d_y, d_z, 
                roll, pitch, yaw, 
            ],  # Pose
            visibility
        ]
    
    val_image_gts[val_image_id] = image_visibilities

# Determine the count of GT apples per visibility bins
visibility_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
visibility_bin_counts = {vb: 0 for vb in visibility_bins}
for val_image_id in val_image_ids:
    image_gts = val_image_gts[val_image_id]
    for apple_id in image_gts.keys():
        visibility = image_gts[apple_id][1]
        for vb in visibility_bins:
            if visibility >= vb:
                visibility_bin_counts[vb] += 1

# Remove the number of GT apples in higher visibility bins from lower visibility bins
for i in range(len(visibility_bins)-1):
    visibility_bin_counts[visibility_bins[i]] -= visibility_bin_counts[visibility_bins[i+1]]


# Iterate the experiments
total_results = {}
iou_thr = [0.5]
for experiment_name in dict_results.keys():
    predictions_df = pd.DataFrame()
    for result_column in ['apple_confidences_25', 'apple_recalls_25', 'apple_precisions_25', 'apple_image_ids_25', 'apple_t_positives_25', 'apple_f_positives_25', 'apple_VA_25', 'apple_QD_25', 'apple_RE_25', 'apple_PE_25', 'apple_YE_25', 'apple_ED_25', 'apple_AP_50', 'mAP_50', 'apple_rec_50', 'mRe_50', 'apple_prec_50', 'mPr_50', 'apple_confidences_50', 'apple_recalls_50', 'apple_precisions_50', 'apple_image_ids_50', 'apple_t_positives_50', 'apple_f_positives_50', 'apple_VA_50', 'apple_QD_50', 'apple_RE_50', 'apple_PE_50', 'apple_YE_50', 'apple_ED_50',]:
        predictions_df[result_column] = dict_results[experiment_name][0.0][result_column]

    predictions_pkl = f"data_disk/testing/testing_dataset/{experiment_name}/predictions.pkl"
    all_images_prediction_bbs = pickle.load(open(predictions_pkl, "rb"))

    predictions_df["apple_gt_id"] = np.NaN
    predictions_df["apple_gt_visibility"] = np.NaN

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    vect_angles = []
    quat_dists = []
    roll_errs = []
    pitch_errs = []
    yaw_errs = []
    euclid_errs = []


    # Iterate the images
    for pred_image_id, gt_image_id in enumerate(val_image_ids):
        id_image_gt_bbs = val_image_gts[gt_image_id]
        id_image_pred_bbs = all_images_prediction_bbs[pred_image_id]

        image_predictions_df = predictions_df[predictions_df.apple_image_ids_50 == float(pred_image_id)]
        indeces_image_predictions = image_predictions_df.index
        cur_num = len(indeces_image_predictions)

        # If there are predictions
        if cur_num != 0:
            # If there are GTs
            if len(id_image_gt_bbs) != 0:
                # Scale down prediction by dividing by 10
                pred_cur2 = copy.deepcopy(id_image_pred_bbs)
                pred_cur2[:,:6] *= 0.1

                gt_keys = np.array([j for j in id_image_gt_bbs.keys()])
                gt_cur2 = np.asarray([id_image_gt_bbs[j][0] for j in gt_keys])
                gt_visibilities = np.asarray([id_image_gt_bbs[j][1] for j in gt_keys])
                gt_detected = np.array([False for j in gt_keys])
                

                N = len(id_image_pred_bbs)
                M = len(id_image_gt_bbs)
                pred_boxes = [OBB(create_transformation_matrix(pred_cur2[i]), 
                        np.array([pred_cur2[i,3], 
                                pred_cur2[i,4], 
                                pred_cur2[i,5]])) 
                        for i in range(N)]
                
                target_boxes = [OBB(create_transformation_matrix(gt_cur2[j]), 
                            np.array([gt_cur2[j][3], 
                            gt_cur2[j][4], 
                            gt_cur2[j][5]])) 
                            for j in range(M)]
                
                iou_volume = torch.zeros((N,M), dtype=torch.float32)
                vect_angle_grid = torch.zeros((N,M), dtype=torch.float32)
                quat_dist_grid = torch.zeros((N,M), dtype=torch.float32)
                roll_err_grid = torch.zeros((N,M), dtype=torch.float32)
                pitch_err_grid = torch.zeros((N,M), dtype=torch.float32)
                yaw_err_grid = torch.zeros((N,M), dtype=torch.float32)
                euclid_err_grid = torch.zeros((N,M), dtype=torch.float32)
                
                for i in range(N):
                    for j in range(M):
                        iou_volume[i,j] = pred_boxes[i].IoU_v(target_boxes[j])
                        vect_angle_grid[i,j] = vect_angle_calc(pred_boxes[i].T, target_boxes[j].T)
                        quat_dist_grid[i,j] = quat_angle_calc(pred_cur2[i][6:], gt_cur2[j][6:])
                        roll_err_grid[i,j] = min(
                            abs(pred_cur2[i][6] - gt_cur2[j][6]),
                            abs(pred_cur2[i][6] - gt_cur2[j][6] - (2 * np.pi)),
                            abs(pred_cur2[i][6] - gt_cur2[j][6] + (2 * np.pi))
                        )
                        pitch_err_grid[i,j] = min(
                            abs(pred_cur2[i][7] - gt_cur2[j][7]),
                            abs(pred_cur2[i][7] - gt_cur2[j][7] - (2 * np.pi)),
                            abs(pred_cur2[i][7] - gt_cur2[j][7] + (2 * np.pi))
                        )
                        yaw_err_grid[i,j] = min(
                            abs(pred_cur2[i][8] - gt_cur2[j][8]),
                            abs(pred_cur2[i][8] - gt_cur2[j][8] - (2 * np.pi)),
                            abs(pred_cur2[i][8] - gt_cur2[j][8] + (2 * np.pi))
                        )
                        euclid_err_grid[i,j] = euclid_dist_calc(pred_cur2[i][:3], gt_cur2[j][:3])


                    iou_max = -np.inf
                    for j in range(M):
                        iou = iou_volume[i,j]
                        if iou > iou_max:
                            iou_max = iou
                            jmax = j

                    thresh = iou_thr[0]
                    if iou_max > thresh:
                        if not gt_detected[jmax]:
                            # Store the data in the predictions DF
                            predictions_df.at[indeces_image_predictions[i], "apple_gt_visibility"] = gt_visibilities[jmax]
                            predictions_df.at[indeces_image_predictions[i], "apple_gt_id"] = gt_keys[jmax]
                            predictions_df.at[indeces_image_predictions[i], "apple_gt_img_id"] = gt_image_id
                            predictions_df.at[indeces_image_predictions[i], "new_apple_roll_error"] = float(roll_err_grid[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_pitch_error"] = float(pitch_err_grid[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_yaw_error"] = float(yaw_err_grid[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_vect_angle"] = float(vect_angle_grid[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_quat_dist"] = float(quat_dist_grid[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_euclid_dist"] = float(euclid_err_grid[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_iou"] = float(iou_volume[i, jmax])
                            predictions_df.at[indeces_image_predictions[i], "new_apple_tp"] = 1.0
                            predictions_df.at[indeces_image_predictions[i], "new_apple_fp"] = 0.0
                            gt_detected[jmax] = True
                            predictions_df.at[indeces_image_predictions[i], "apple_pred_roll"] = pred_cur2[i, 6]
                            predictions_df.at[indeces_image_predictions[i], "apple_pred_pitch"] = pred_cur2[i, 7]
                            predictions_df.at[indeces_image_predictions[i], "apple_pred_yaw"] = pred_cur2[i, 8]
                    else: 
                        # Store the data in the predictions DF
                        predictions_df.at[indeces_image_predictions[i], "apple_gt_visibility"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "apple_gt_id"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "apple_gt_img_id"] = gt_image_id
                        predictions_df.at[indeces_image_predictions[i], "new_apple_roll_error"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_pitch_error"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_yaw_error"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_vect_angle"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_quat_dist"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_euclid_dist"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_iou"] = np.nan
                        predictions_df.at[indeces_image_predictions[i], "new_apple_tp"] = 0.0
                        predictions_df.at[indeces_image_predictions[i], "new_apple_fp"] = 1.0
                        predictions_df.at[indeces_image_predictions[i], "apple_pred_roll"] = pred_cur2[i, 6]
                        predictions_df.at[indeces_image_predictions[i], "apple_pred_pitch"] = pred_cur2[i, 7]
                        predictions_df.at[indeces_image_predictions[i], "apple_pred_yaw"] = pred_cur2[i, 8]

    total_results[experiment_name] = copy.deepcopy(predictions_df)

# Store the predictions df for later
total_predictions_df = pd.DataFrame()
for experiment_name in total_results.keys():
    predictions_df = total_results[experiment_name]
    predictions_df["real_fraction"] = float(experiment_name.split("_")[3])/1000
    predictions_df["splat_fraction"] = float(experiment_name.split("_")[5])/1000
    predictions_df["visibility_lower_bound"] = float(experiment_name.split("_")[1])/100

    if float(experiment_name.split("_")[3])/1000 == 0.0:
        predictions_df["result_type"] = "Rendered"
    elif float(experiment_name.split("_")[5])/1000 == 0.0:
        predictions_df["result_type"] = "Original"
    else:
        predictions_df["result_type"] = "Mixed"

    total_predictions_df = pd.concat([total_predictions_df, predictions_df], ignore_index=True)

# Save to csv
total_predictions_df.to_csv("data_disk/testing/testing_dataset/all_predictions.csv", index=False)

#%% Load from csv

total_predictions_df = pd.read_csv("data_disk/testing/testing_dataset/all_predictions.csv")

# Replace "Generated" with "Rendered" in result_type
total_predictions_df["result_type"] = total_predictions_df["result_type"].replace("Generated", "Rendered")

# Convert euclid dist back to mm
total_predictions_df["apple_ED_50"] = total_predictions_df["apple_ED_50"] * 100
total_predictions_df["apple_ED_25"] = total_predictions_df["apple_ED_25"] * 100

# Put visibility into bins of 0.1
total_predictions_df["visibility_bin"] = (total_predictions_df["apple_gt_visibility"] // 0.1) * 0.1

# Set values greater than 0.7 to 0.7
total_predictions_df.loc[total_predictions_df["visibility_bin"] > 0.7, "visibility_bin"] = 0.7

total_predictions_df["normalized_label_count"] = total_predictions_df["real_fraction"] + total_predictions_df["splat_fraction"]

# Get log of normalized label count
total_predictions_df["log_label_count"] = np.log(total_predictions_df["normalized_label_count"])

# Get degrees vector angle
total_predictions_df["angle_error"] = np.rad2deg(np.arccos(1.0-total_predictions_df["apple_VA_50"]))

total_predictions_df["angle_roll"] = np.rad2deg(total_predictions_df["apple_RE_50"])
total_predictions_df["angle_pitch"] = np.rad2deg(total_predictions_df["apple_PE_50"])
total_predictions_df["angle_yaw"] = np.rad2deg(total_predictions_df["apple_YE_50"])

total_predictions_df["roll_deg"] = total_predictions_df["apple_pred_roll"] * 180 / np.pi
total_predictions_df["pitch_deg"] = total_predictions_df["apple_pred_pitch"] * 180 / np.pi
total_predictions_df["yaw_deg"] = total_predictions_df["apple_pred_yaw"] * 180 / np.pi

# Rotate yaw by -90 degrees to align with camera coordinate system
total_predictions_df["yaw_deg"] -= 90.0

# Remap below -180
too_lows_roll = total_predictions_df["roll_deg"] < -180.0
total_predictions_df.loc[too_lows_roll, "roll_deg"] += 360.0
too_lows_pitch = total_predictions_df["pitch_deg"] < -180.0
total_predictions_df.loc[too_lows_pitch, "pitch_deg"] += 360.0
too_lows_yaw = total_predictions_df["yaw_deg"] < -180.0
total_predictions_df.loc[too_lows_yaw, "yaw_deg"] += 360.0


#%% Determine recall per visibility bin

# Sum the tp per visibility bin and normalized label count
recall_results = total_predictions_df[total_predictions_df.apple_t_positives_50 == 1].groupby(["visibility_bin", "normalized_label_count", "log_label_count", "result_type"]).size().reset_index(name='tp_count')

# Calculate recall by dividing tp_count by the total number of gt apples in that visibility bin
recall_results["recall"] = 0.0
for index, row in recall_results.iterrows():
    vb = row["visibility_bin"]
    nl = row["normalized_label_count"]
    rt = row["result_type"]
    tp_count = row["tp_count"]
    total_gt_count = visibility_bin_counts[round(vb, 3)]
    recall = tp_count / total_gt_count if total_gt_count > 0 else 0.0
    recall_results.at[index, "recall"] = recall
    
#%% Plot distribution of predictions

tp_predictions_df = total_predictions_df[total_predictions_df.new_apple_tp == 1]

for result_dataset_type in tp_predictions_df.result_type.unique():
    dataset_total_predictions_df = tp_predictions_df[tp_predictions_df.result_type == result_dataset_type]
    for result_label_count in dataset_total_predictions_df.normalized_label_count.unique():
        print(f"Dataset: {result_dataset_type}, label count: {result_label_count}")

        # Alternative plot
        g1 = sns.JointGrid(data=dataset_total_predictions_df[dataset_total_predictions_df["normalized_label_count"] == result_label_count], x="pitch_deg", y="yaw_deg", xlim=(-90,90), ylim=(-180,180), height=4)
        g1.plot_joint(sns.histplot, element="step")
        g1.plot_marginals(sns.histplot, element="step")
        g1.set_axis_labels("Pred Pitch (deg)", "Pred Yaw (deg)")
        g1.savefig(f"data_disk/testing/testing_dataset/{result_dataset_type.lower()}_{result_label_count}_predictions_pitch_yaw.pdf", bbox_inches='tight')
        plt.show()

#%% Plot the vector angle error vs visibility
tp_predictions_df = total_predictions_df[(total_predictions_df.apple_t_positives_50 == 1) & (total_predictions_df.normalized_label_count == 1)]

result_type_set = ["Original", "Rendered"]

norm = plt.Normalize(tp_predictions_df.log_label_count.min(), tp_predictions_df.log_label_count.max())

for col in ["angle_error", "apple_ED_50"]:
    
    # fig, axs = plt.subplots(1, 2, 
    #     figsize=(7, 3.5), 
    #     constrained_layout=True,
    #     sharey=True,
    # )

    for dataset_iter, dataset_result_type in enumerate(result_type_set):

        fig = plt.figure(
            figsize=(3, 3),
            constrained_layout=True,
        )
        ax = fig.add_subplot(1, 1, 1)
        
        sns.lineplot(
            data=tp_predictions_df[tp_predictions_df["result_type"] == dataset_result_type], 
            x="visibility_bin", 
            y=col, 
            ax=ax,
            color=darkest_flare_color,
            estimator="mean",
            err_style="bars",
        )

        # ax.get_legend().remove()
        ax.set_xlabel("Occlusion [\%]")
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        # Replace the text at the xticks with 0:10, etc
        ax.set_xticklabels(['90:100', '80:90', '70:80', '60:70', '50:60', '40:50', '30:40', '0:30'], rotation=45, ha='center')

        # Invert direction of x-axis
        ax.invert_xaxis()

        # Add gridlines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if col == "angle_error":
            y_label_text = "Vector Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "apple_ED_50":
            y_label_text = "Euclidean Distance Error(mm)"
            ax.set_ylabel(y_label_text)
        elif col == "angle_roll":
            y_label_text = "Roll Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "angle_pitch":
            y_label_text = "Pitch Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "angle_yaw":
            y_label_text = "Yaw Angle Error (°)"
            ax.set_ylabel(y_label_text)
        
        ax.set_ylim(0.0)

        # Save plots
        fig.savefig(f"data_disk/testing/testing_dataset/{dataset_result_type.lower()}_{col}_vs_occlusion_bins.pdf")

        plt.show()

#%% Plot the recall vs visibility

result_type_set = ["Original", "Rendered"]

for col in ["recall"]:
    for dataset_iter, dataset_result_type in enumerate(result_type_set):

        fig = plt.figure(
            figsize=(3, 3),
            constrained_layout=True,
        )
        ax = fig.add_subplot(1, 1, 1)
        
        sns.lineplot(
            data=recall_results[(recall_results["result_type"] == dataset_result_type) & (recall_results.normalized_label_count == 1)], 
            x="visibility_bin", 
            y=col, 
            ax=ax, 
            color=darkest_flare_color,
        )
    
        ax.set_xlabel("Occlusion [\%]")
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        # Replace the text at the xticks with 0:10, etc
        ax.set_xticklabels(['90:100', '80:90', '70:80', '60:70', '50:60', '40:50', '30:40', '0:30'], rotation=45, ha='center')

        # Invert direction of x-axis
        ax.invert_xaxis()

        # Add gridlines
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if col == "angle_error":
            y_label_text = "Vector Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "apple_ED_50":
            y_label_text = "Euclidean Distance Error (mm)"
            ax.set_ylabel(y_label_text)
        elif col == "angle_roll":
            y_label_text = "Roll Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "angle_pitch":
            y_label_text = "Pitch Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "angle_yaw":
            y_label_text = "Yaw Angle Error (°)"
            ax.set_ylabel(y_label_text)
        elif col == "recall":
            y_label_text = "Recall"
            ax.set_ylabel(y_label_text)
            ax.set_ylim(0.0, 1.05)


        # Save plots
        fig.savefig(f"data_disk/testing/testing_dataset/{dataset_result_type.lower()}_{col}_vs_occlusion_bins.pdf")

        plt.show()

#%% Use bootstrapping to calculate confidence intervals for precision, recall and F1

def bootstrapped_performance_calculator(outputs):
    performance_indicators = {}

    for key in outputs.keys():
        performance_indicators[key] = {}
        sum_gts = sum(outputs[key]['gt_counts'].values())
        performance_indicators[key]['gts_sum'] = sum_gts
        for iou_threshold in ["_25", "_50"]:
            confidence_order = np.argsort(-outputs[key]["apple_confidences"+iou_threshold])

            # Sort all model predictions
            confidences = outputs[key]["apple_confidences"+iou_threshold][confidence_order]
            true_positives = outputs[key]["apple_t_positives"+iou_threshold][confidence_order]
            false_positives = outputs[key]["apple_f_positives"+iou_threshold][confidence_order]
            neutralized_false_positives = outputs[0.0]["apple_f_positives"+iou_threshold][confidence_order]
            neutralized_true_positives = outputs[0.0]["apple_t_positives"+iou_threshold][confidence_order]
            vect_angle_errors = outputs[key]["apple_VA"+iou_threshold][confidence_order]
            quaternion_dists = outputs[key]["apple_QD"+iou_threshold][confidence_order]
            roll_errors = outputs[key]["apple_RE"+iou_threshold][confidence_order]
            pitch_errors = outputs[key]["apple_PE"+iou_threshold][confidence_order]
            yaw_errors = outputs[key]["apple_YE"+iou_threshold][confidence_order]
            euclidean_dists = outputs[key]["apple_ED"+iou_threshold][confidence_order]

            # Calculate precision and recall from highest to lowest confidence
            tp_sum = np.cumsum(true_positives)
            fp_sum = np.cumsum(false_positives)
            recalls = tp_sum / sum_gts
            precisions = tp_sum / np.maximum(tp_sum + fp_sum, np.finfo(np.float64).eps)

            # Calculate f1_scores
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + np.finfo(np.float64).eps)

            max_f1_confidence = confidences[np.argmax(f1_scores)]
            f1_score = np.max(f1_scores)

            # Calculate precision and recall at each confidence level with neutral positives
            ntp_sum = np.cumsum(neutralized_true_positives)
            nfp_sum = np.cumsum(neutralized_false_positives)
            neutral_precisions = tp_sum / np.maximum(tp_sum + nfp_sum, np.finfo(np.float64).eps)
            double_neutral_precisions = ntp_sum / np.maximum(ntp_sum + nfp_sum, np.finfo(np.float64).eps)

            # Calculate neutral f1 scores
            neutral_f1_scores = 2 * (neutral_precisions * recalls) / (neutral_precisions + recalls + np.finfo(np.float64).eps)
            double_neutral_f1_scores = 2 * (double_neutral_precisions * recalls) / (double_neutral_precisions + recalls + np.finfo(np.float64).eps)

            neutral_max_f1_confidence = confidences[np.argmax(neutral_f1_scores)]
            double_neutral_max_f1_confidence = confidences[np.argmax(double_neutral_f1_scores)]
            neutral_f1_score = np.max(neutral_f1_scores)
            double_neutral_f1_score = np.max(double_neutral_f1_scores)
            
            desired_predictions = confidences > max_f1_confidence

            mean_vect_angle_error = np.nanmean(vect_angle_errors[desired_predictions])
            mean_quaternion_dist = np.nanmean(quaternion_dists[desired_predictions])
            mean_roll_error = np.nanmean(roll_errors[desired_predictions])
            mean_pitch_error = np.nanmean(pitch_errors[desired_predictions])
            mean_yaw_error = np.nanmean(yaw_errors[desired_predictions])
            mean_euclidean_dist = np.nanmean(euclidean_dists[desired_predictions])

            # Store the performance indicators
            performance_indicators[key].update({
                "max_f1_confidence"+iou_threshold: max_f1_confidence,
                "f1_score"+iou_threshold: f1_score,
                "neutral_max_f1_confidence"+iou_threshold: neutral_max_f1_confidence,
                "neutral_f1_score"+iou_threshold: neutral_f1_score,
                "double_neutral_max_f1_confidence"+iou_threshold: double_neutral_max_f1_confidence,
                "double_neutral_f1_score"+iou_threshold: double_neutral_f1_score,
                "mean_vect_angle_error"+iou_threshold: mean_vect_angle_error,
                "mean_quaternion_dist"+iou_threshold: mean_quaternion_dist,
                "mean_roll_error"+iou_threshold: mean_roll_error,
                "mean_pitch_error"+iou_threshold: mean_pitch_error,
                "mean_yaw_error"+iou_threshold: mean_yaw_error,
                "mean_euclidean_dist"+iou_threshold: mean_euclidean_dist,
                "sum_tp"+iou_threshold: tp_sum,
                "sum_fp"+iou_threshold: fp_sum,
                "sum_ntp"+iou_threshold: ntp_sum,
                "sum_nfp"+iou_threshold: nfp_sum,
            })

    return performance_indicators

def build_subset(results, sample_ids, iter_columns):
        outputs = {}
        sample_ids = np.array(sample_ids)
        sample_ids_str = set(map(str, sample_ids))
        for key in results.keys():
            outputs[key] = {}
            # Precompute image ids arrays for both thresholds
            image_ids_25 = np.asarray(results[key]["apple_image_ids_25"])
            image_ids_50 = np.asarray(results[key]["apple_image_ids_50"])
            for iou_threshold, image_ids in [("_25", image_ids_25), ("_50", image_ids_50)]:
                column = 'apple_image_ids'
                # Assign output_sample_id for each sample_id occurrence
                temp_outputs_list = np.empty(0, dtype=int)
                target_ids = np.empty(0, dtype=int)
                for output_sample_id, sample_id in enumerate(sample_ids):
                    mask = image_ids == sample_id
                    if np.any(mask):
                        temp_outputs_list = np.concatenate([temp_outputs_list, np.full(np.sum(mask), output_sample_id, dtype=int)])
                        target_ids = np.concatenate([target_ids, np.argwhere(mask).flatten()])
                outputs[key][column + iou_threshold] = temp_outputs_list
                for column in iter_columns:
                    arr = np.asarray(results[key][column + iou_threshold])
                    # Use boolean mask for fast selection
                    outputs[key][column + iou_threshold] = arr[target_ids]
            # Update the GT dict
            gt_dict = {output_sample_id: results[key]["gt_counts"][str(sample_id)] for output_sample_id, sample_id in enumerate(sample_ids) if str(sample_id) in results[key]["gt_counts"]}
            outputs[key]["gt_counts"] = gt_dict
        return outputs

def build_and_process_subset(results, sample_ids, iter_columns):
    """
    Build a subset of the results based on the sample ids and process it.
    """
    outputs = build_subset(results, sample_ids, iter_columns)
    return bootstrapped_performance_calculator(outputs)


iter_columns = [
    'apple_confidences', 
    'apple_t_positives', 
    'apple_f_positives', 
    'apple_VA', 
    'apple_QD', 
    'apple_RE', 
    'apple_PE', 
    'apple_YE', 
    'apple_ED',
]

# Testing sample ids
sample_ids = [2, 0, 2]


# Set seed for np random
np.random.seed(42)

bootstrapped_results = {}

for dataset_setting in dict_results.keys():
    print(f"Processing dataset: {dataset_setting}")
    curr_results = dict_results[dataset_setting]

    unsampled_ids = list(curr_results[list(curr_results.keys())[0]]["gt_counts"].keys())
    # Convert to integers
    unsampled_ids = [int(unsampled_id) for unsampled_id in unsampled_ids]

    # Perform sampling
    bootstrap_sample_id_sets = [
        np.random.choice(
            unsampled_ids, 
            size=len(unsampled_ids), 
            replace=True,
        ) for _ in range(100)
    ]

    n_cores = int(os.cpu_count())
    with ThreadPool(n_cores) as pool:
        output_results = list(pool.starmap(
            build_and_process_subset, 
            [
                (
                    curr_results, sample_ids, iter_columns
                ) for sample_ids in bootstrap_sample_id_sets
            ]
        ))

    # Convert the list of dictionaries to a single df with columns for each metric and visibility
    performance_indicators = {}
    for i, output in enumerate(output_results):
        for key in output.keys():
            for metric in output[key].keys():
                combined_key = f"{key}_{metric}"
                if combined_key not in performance_indicators:
                    performance_indicators[combined_key] = []
                performance_indicators[combined_key].append(output[key][metric])

    df_performance_indicators = pd.DataFrame(performance_indicators)
    # Place in dict
    bootstrapped_results[dataset_setting] = df_performance_indicators

# %% Save the bootstrapped results to a file

test_visibilities = [
    0.0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
]

# Convert dictionary to a DataFrame
results_df = pd.DataFrame()
for result_type in bootstrapped_results.keys():
    df_curr_results = bootstrapped_results[result_type]
    for test_visibility in test_visibilities:
        df_curr_results_filtered = copy.deepcopy(df_curr_results.filter(like=f"{test_visibility}_"))
        df_curr_results_filtered.columns = ["_".join(col.split("_")[1:]) for col in df_curr_results_filtered.columns]
        df_curr_results_filtered["test_visibility"] = test_visibility
        df_curr_results_filtered["visibility_lower_bound"] = float(result_type.split("_")[1])/100.
        # Add splat and real fraction to vis_dict
        df_curr_results_filtered["real_fraction"] = float(result_type.split("_")[3])/1000
        df_curr_results_filtered["splat_fraction"] = float(result_type.split("_")[5])/1000
        if float(result_type.split("_")[3])/1000 == 0.0:
            df_curr_results_filtered["result_type"] = "only_splat"
        elif float(result_type.split("_")[5])/1000 == 0.0:
            df_curr_results_filtered["result_type"] = "only_real"
        else:
            df_curr_results_filtered["result_type"] = "mixed"
                
        results_df = pd.concat([results_df, df_curr_results_filtered], ignore_index=True)

# Save the results to a CSV file
output_path = "data_disk/testing/testing_dataset/bootstrapped_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

#%% Import the results from the CSV file
results_df = pd.read_csv("data_disk/testing/testing_dataset/bootstrapped_results.csv")
results_df["test_occlusion"] = 1.0 - results_df["test_visibility"]
results_df["occlusion_upper_bound"] = 1.0 - results_df["visibility_lower_bound"]
results_df["normalized_label_count"] = results_df["real_fraction"] + results_df["splat_fraction"]

# Convert euclidean distance from dm to mm
results_df["mean_euclidean_dist_25"] = results_df["mean_euclidean_dist_25"] * 100
results_df["mean_euclidean_dist_50"] = results_df["mean_euclidean_dist_50"] * 100

# Filter out rows with real_fraction==1.0 and splat_fraction==0.5
results_df = results_df[~((results_df["real_fraction"] == 1.0) & (results_df["splat_fraction"] == 0.5))]

# Rename result types
results_df["result_type"] = results_df["result_type"].replace({
    "only_splat": "Rendered",
    "only_real": "Original",
    "mixed": "Mixed",
})

# Get degrees vector angle
results_df["angle_error"] = np.rad2deg(np.arccos(1.0-results_df["mean_vect_angle_error_50"]))

results_df["angle_roll"] = np.rad2deg(results_df["mean_roll_error_50"])
results_df["angle_pitch"] = np.rad2deg(results_df["mean_pitch_error_50"])
results_df["angle_yaw"] = np.rad2deg(results_df["mean_yaw_error_50"])

#%% Make plots for the bootstrapped results, F1 score and neutral F1 score vs normalized label count
# Define the figure with two subplots
fig1 = plt.figure(
    figsize=(3, 3),
    constrained_layout=True,
    )
axs1 = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure(
    figsize=(3, 3),
    constrained_layout=True,
)
axs2 = fig2.add_subplot(1, 1, 1)

test_occlusion_setting = 0.85
df_curr_results = results_df[(results_df.test_occlusion == test_occlusion_setting)]

# Plot the results, with rows being repetitions, merged to show mean and confidence interval
sns.lineplot(
    data=df_curr_results, 
    x="normalized_label_count", 
    y="f1_score_50", 
    ax=axs1, 
    hue="result_type",
    palette=color_palette[1::2],
    errorbar="pi",
    estimator="mean",
    err_style="bars",
)

sns.lineplot(
    data=df_curr_results, 
    x="normalized_label_count", 
    y="double_neutral_f1_score_50", 
    ax=axs2, 
    hue="result_type",
    palette=color_palette[1::2],
    errorbar="pi",
    estimator="mean",
    err_style="bars",
)


# Set axis ranges and labels
# axs1.set_ylim(0, 1.05)
axs1.set_xlabel("Normalized label count")
axs1.set_ylabel("F1 Score")
# axs2.set_ylim(0, 1.05)
axs2.set_xlabel("Normalized label count")
axs2.set_ylabel("Neutral F1 Score")

# Add legend
axs1.legend(title="Dataset")
axs2.legend(title="Dataset")

# Add gridlines
axs1.grid(True, which='both', linestyle='--', linewidth=0.5)
axs2.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Save the plots as PDF files
fig1.savefig(f"data_disk/testing/testing_dataset/f1_score_50_vs_label_count.pdf")
fig2.savefig(f"data_disk/testing/testing_dataset/neutral_f1_score_50_vs_label_count.pdf")

#%% Make plots for the bootstrapped results, vector angle and euclid dist vs normalized label count
# Define the figure with two subplots
fig1 = plt.figure(
    figsize=(3, 3),
    constrained_layout=True,
    )
axs1 = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure(
    figsize=(3, 3),
    constrained_layout=True,
)
axs2 = fig2.add_subplot(1, 1, 1)

test_occlusion_setting = 0.85
df_curr_results = results_df[(results_df.test_occlusion == test_occlusion_setting)]

# Plot the results, with rows being repetitions, merged to show mean and confidence interval
sns.lineplot(
    data=df_curr_results, 
    x="normalized_label_count", 
    y="angle_error", 
    ax=axs1, 
    hue="result_type",
    palette=color_palette[1::2],
    errorbar="pi",
    estimator="mean",
    err_style="bars",
)

sns.lineplot(
    data=df_curr_results, 
    x="normalized_label_count", 
    y="mean_euclidean_dist_50", 
    ax=axs2, 
    hue="result_type",
    palette=color_palette[1::2],
    errorbar="pi",
    estimator="mean",
    err_style="bars",
)


# Set axis ranges and labels
# axs1.set_ylim(0, 1.05)
axs1.set_xlabel("Normalized label count")
axs1.set_ylabel("Vector Angle Error (degrees)")
# axs2.set_ylim(0, 1.05)
axs2.set_xlabel("Normalized label count")
axs2.set_ylabel("Euclidean Distance Error (mm)")

# Add legend
axs1.legend(title="Dataset")
axs2.legend(title="Dataset")

# Add gridlines
axs1.grid(True, which='both', linestyle='--', linewidth=0.5)
axs2.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Save the plots as PDF files
fig1.savefig(f"data_disk/testing/testing_dataset/bootstrapped_vector_angle_50_vs_label_count.pdf")
fig2.savefig(f"data_disk/testing/testing_dataset/bootstrapped_euclid_dist_50_vs_label_count.pdf")
# %%
