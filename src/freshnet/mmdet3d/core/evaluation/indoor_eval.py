# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable
import random
import os
import json
from tqdm import tqdm
import copy

from scipy.spatial import ConvexHull

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
    
    def pd(self, box2):
        """ returns the distance the centers of the boxes """
        p, r, d = self.get_prd()
        p2, r2, d2 = box2.get_prd()
        return positionDifference(p, p2)
    
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
    return 1.0 - np.clip(np.dot(vx1, vx2) / (np.linalg.norm(vx1) * np.linalg.norm(vx2)), -1.0, 1.0)

def euler_to_quaternion(roll, pitch, yaw):
    qx = torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) - torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    qy = torch.cos(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2)
    qz = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.sin(yaw / 2) - torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.cos(yaw / 2)
    qw = torch.cos(roll / 2) * torch.cos(pitch / 2) * torch.cos(yaw / 2) + torch.sin(roll / 2) * torch.sin(pitch / 2) * torch.sin(yaw / 2)
    return torch.stack([qw, qx, qy, qz], dim=-1)

def quat_angle_calc(rpy1, rpy2):
    """Calculate the quaternion angle between two rotation matrices."""
    q1 = euler_to_quaternion(rpy1[0], rpy1[1], rpy1[2])
    q2 = euler_to_quaternion(rpy2[0], rpy2[1], rpy2[2])
    return torch.acos(torch.abs((q1 * q2).sum(-1)))

def euclid_dist_calc(c1, c2):
    return np.sqrt(sum((c1-c2)**2))

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_det_cls(pred, gt, iou_thr=None, ckpt_pth=None):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    for key, value in pred.items():
        for item in value:
            tensor_obj = value[0][0].tensor
            break
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, tensor_obj.shape[1]], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

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

    for img_id in tqdm(pred.keys()):
        confidence_per_img=[]
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        if pred[img_id][0][0].tensor.shape[1] == 9:
            pred_cur = torch.zeros((cur_num, 9), dtype=torch.float32)
        else:
            pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            confidence_per_img.append(score)
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            ########### stem-axis rotation evaluation #####################
            pred_cur2 = pred_cur.tensor
            gt_cur2 = gt_cur.tensor
            
            N = pred_cur2.shape[0]
            M = gt_cur2.shape[0]
            # import pdb;pdb.set_trace()
            pred_boxes = [OBB(create_transformation_matrix(pred_cur2[i].detach().cpu().numpy()), 
                        np.array([pred_cur2[i][3].detach().cpu().numpy(), 
                                pred_cur2[i][4].detach().cpu().numpy(), 
                                pred_cur2[i][5].detach().cpu().numpy()])) 
                        for i in range(N)]
            
            target_boxes = [OBB(create_transformation_matrix(gt_cur2[j].detach().cpu().numpy()), 
                            np.array([gt_cur2[j][3].detach().cpu().numpy(), 
                            gt_cur2[j][4].detach().cpu().numpy(), 
                            gt_cur2[j][5].detach().cpu().numpy()])) 
                            for j in range(M)]
            
            iou_volume = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            vect_angle_grid = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            quat_dist_grid = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            roll_err_grid = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            pitch_err_grid = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            yaw_err_grid = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            euclid_err_grid = torch.zeros((N,M), dtype=torch.float32, device=pred_cur2.device)
            
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

                    # roll_err_grid[i,j] = min(
                    #     abs(np.mod(pred_cur2[i][6] - gt_cur2[j][6], 2*np.pi)),
                    #     abs(np.mod(gt_cur2[j][6] - pred_cur2[i][6], 2*np.pi))
                    # )
                    # pitch_err_grid[i,j] = min(
                    #     abs(np.mod(pred_cur2[i][7] - gt_cur2[j][7], 2*np.pi)),
                    #     abs(np.mod(gt_cur2[j][7] - pred_cur2[i][7], 2*np.pi))
                    # )
                    # yaw_err_grid[i,j] = min(
                    #     abs(np.mod(pred_cur2[i][8] - gt_cur2[j][8], 2*np.pi)),
                    #     abs(np.mod(gt_cur2[j][8] - pred_cur2[i][8], 2*np.pi))
                    # )

                    
            iou_cur = iou_volume
            
            #############################################################
            # iou_cur = pred_cur.overlaps(pred_cur, gt_cur) ## original evaluation
            for i in range(cur_num):
                ious.append(iou_cur[i])
                vect_angles.append(vect_angle_grid[i])
                quat_dists.append(quat_dist_grid[i])
                roll_errs.append(roll_err_grid[i])
                pitch_errs.append(pitch_err_grid[i])
                yaw_errs.append(yaw_err_grid[i])
                euclid_errs.append(euclid_err_grid[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))
                vect_angles.append(np.full(1, np.pi))
                quat_dists.append(np.full(1, np.pi))
                roll_errs.append(np.full(1, np.pi))
                pitch_errs.append(np.full(1, np.pi))
                yaw_errs.append(np.full(1, np.pi))
                euclid_errs.append(np.full(1, np.nan))
        

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    confidences = [confidence[x] for x in sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]
    vect_angles = [vect_angles[x] for x in sorted_ind]
    quat_dists = [quat_dists[x] for x in sorted_ind]
    roll_errs = [roll_errs[x] for x in sorted_ind]
    pitch_errs = [pitch_errs[x] for x in sorted_ind]
    yaw_errs = [yaw_errs[x] for x in sorted_ind]
    euclid_errs = [euclid_errs[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    vect_angle_thr = [np.full(nd, np.nan) for i in iou_thr]
    quat_dist_thr = [np.full(nd, np.nan) for i in iou_thr]
    roll_err_thr = [np.full(nd, np.nan) for i in iou_thr]
    pitch_err_thr = [np.full(nd, np.nan) for i in iou_thr]
    yaw_err_thr = [np.full(nd, np.nan) for i in iou_thr]
    euclid_err_thr = [np.full(nd, np.nan) for i in iou_thr]
    
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]
        cur_vect_angle = vect_angles[d]
        cur_quat_dist = quat_dists[d]
        cur_roll_err = roll_errs[d]
        cur_pitch_err = pitch_errs[d]
        cur_yaw_err = yaw_errs[d]
        cur_euclid_err = euclid_errs[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                    vect_angle_thr[iou_idx][d] = cur_vect_angle[jmax]
                    quat_dist_thr[iou_idx][d] = cur_quat_dist[jmax]
                    roll_err_thr[iou_idx][d] = cur_roll_err[jmax]
                    pitch_err_thr[iou_idx][d] = cur_pitch_err[jmax]
                    yaw_err_thr[iou_idx][d] = cur_yaw_err[jmax]
                    euclid_err_thr[iou_idx][d] = cur_euclid_err[jmax]
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.
        
    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)

        # Calculate the average vector angle, quaternion distance, roll, pitch, yaw error
        vect_angle = np.where(np.isnan(vect_angle_thr[iou_idx]), 
                                     'null', vect_angle_thr[iou_idx])
        quat_dist = np.where(np.isnan(quat_dist_thr[iou_idx]), 
                                     'null', quat_dist_thr[iou_idx])
        roll_err = np.where(np.isnan(roll_err_thr[iou_idx]), 
                                     'null', roll_err_thr[iou_idx])
        pitch_err = np.where(np.isnan(pitch_err_thr[iou_idx]), 
                                     'null', pitch_err_thr[iou_idx])
        yaw_err = np.where(np.isnan(yaw_err_thr[iou_idx]), 
                                     'null', yaw_err_thr[iou_idx])
        euclid_err = np.where(np.isnan(euclid_err_thr[iou_idx]), 
                                     'null', euclid_err_thr[iou_idx])


        ret.append((
            image_ids,
            tp_thr[iou_idx],
            fp_thr[iou_idx],
            confidences,
            recall, 
            precision, 
            ap, 
            vect_angle, 
            quat_dist, 
            roll_err, 
            pitch_err, 
            yaw_err, 
            euclid_err,
        ))
        
    return ret

def eval_map_recall(pred, gt, ovthresh=None,ckpt_pth=None):
    """Evaluate mAP and recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred (dict): Information of detection results,
            which maps class_id and predictions.
        gt (dict): Information of ground truths, which maps class_id and
            ground truths.
        ovthresh (list[float], optional): iou threshold. Default: None.

    Return:
        tuple[dict]: dict results of recall, AP, and precision for all classes.
    """

    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh, ckpt_pth=ckpt_pth)
    image_ids = [{} for i in ovthresh]
    tps = [{} for i in ovthresh]
    fps = [{} for i in ovthresh]
    scores = [{} for i in ovthresh]
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]
    vect_angle = [{} for i in ovthresh]
    quat_dist = [{} for i in ovthresh]
    roll_err = [{} for i in ovthresh]
    pitch_err = [{} for i in ovthresh]
    yaw_err = [{} for i in ovthresh]
    euclid_err = [{} for i in ovthresh]

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                image_ids[iou_idx][label], tps[iou_idx][label], fps[iou_idx][label], scores[iou_idx][label], recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][label], vect_angle[iou_idx][label], quat_dist[iou_idx][label], roll_err[iou_idx][label], pitch_err[iou_idx][label], yaw_err[iou_idx][label], euclid_err[iou_idx][label] = ret_values[label][iou_idx]
            else:
                scores[iou_idx][label] = np.zeros(1)
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)
                vect_angle[iou_idx][label] = np.nan
                quat_dist[iou_idx][label] = np.nan
                roll_err[iou_idx][label] = np.nan
                pitch_err[iou_idx][label] = np.nan
                yaw_err[iou_idx][label] = np.nan
                euclid_err[iou_idx][label] = np.nan

    return image_ids, tps, fps, scores, recall, precision, ap, vect_angle, quat_dist, roll_err, pitch_err, yaw_err, euclid_err

def indoor_eval(gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None,
                box_type_3d=None,
                box_mode_3d=None,
                ckpt_pth=None):
    """Indoor Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection annotations. the dict
            includes the following keys

            - labels_3d (torch.Tensor): Labels of boxes.
            - boxes_3d (:obj:`BaseInstance3DBoxes`):
                3D bounding boxes in Depth coordinate.
            - scores_3d (torch.Tensor): Scores of boxes.
        metric (list[float]): IoU thresholds for computing average precisions.
        label2cat (dict): Map from label to category.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Return:
        dict[str, float]: Dict of results.
    """
    assert len(dt_annos) == len(gt_annos)
    
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    label_gt_count = {}

    for img_id in range(len(dt_annos)):
        # parse detected annotations
        det_anno = dt_annos[img_id]
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))


        # parse gt annotations
        gt_anno = gt_annos[img_id]
        # Store gt_num
        label_gt_count[str(img_id)] = gt_anno['gt_num']
        if gt_anno['gt_num'] != 0:
            boxes_gt_upright = copy.deepcopy(gt_anno['gt_boxes_upright_depth'])
            boxes_gt_upright[:, 6:] *= 0.1  # If data loading changes, comment this line (KITTI)
            gt_boxes = box_type_3d(
                boxes_gt_upright,
                box_dim=boxes_gt_upright.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            labels_3d = gt_anno['class']
        else:
            gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            labels_3d = np.array([], dtype=np.int64)

        for i in range(len(labels_3d)):
            label = labels_3d[i]
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    image_ids, tps, fps, scores, rec, prec, ap, vect_angle, quat_dist, roll_err, pitch_err, yaw_err, euclid_err = eval_map_recall(pred, gt, metric,ckpt_pth=ckpt_pth)
    ret_dict = dict()
    header = ['classes']
    table_columns = [[label2cat[label]
                      for label in ap[0].keys()] + ['Overall']]

    for i, iou_thresh in enumerate(metric):
        header.append(f'AP_{int(100*iou_thresh)}')
        header.append(f'Re_{int(100*iou_thresh)}')
        header.append(f'Pr_{int(100*iou_thresh)}')
        rec_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{int(100*iou_thresh)}'] = float(
                ap[i][label][0])
        ret_dict[f'mAP_{int(100*iou_thresh)}'] = float(
            np.mean(list(ap[i].values())))

        table_columns.append(list(map(float, list(ap[i].values()))))
        table_columns[-1] += [ret_dict[f'mAP_{int(100*iou_thresh)}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{int(100*iou_thresh)}'] = float(
                rec[i][label][-1])
            rec_list.append(rec[i][label][-1])
        ret_dict[f'mRe_{int(100*iou_thresh)}'] = float(np.mean(rec_list))

        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f'mRe_{int(100*iou_thresh)}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        prec_list=[]
        for label in prec[i].keys():
            ret_dict[f'{label2cat[label]}_prec_{int(100*iou_thresh)}'] = float(
                prec[i][label][0])
            prec_list.append(prec[i][label][0])
        ret_dict[f'mPr_{int(100*iou_thresh)}'] = float(np.mean(prec_list))

        table_columns.append(list(map(float, prec_list)))
        table_columns[-1] += [ret_dict[f'mPr_{int(100*iou_thresh)}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        # Additionally, add all 
        # confidences, recalls, precisions, image_ids, t_positives, f_positives
        # vector angle errors, quat dists, 
        # roll errors, pitch errors, yaw errrors, and euclid distances
        # to the return dict
        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_confidences_{int(100*iou_thresh)}'] = list(
                scores[i][label])
            ret_dict[f'{label2cat[label]}_recalls_{int(100*iou_thresh)}'] = list(
                rec[i][label])
            ret_dict[f'{label2cat[label]}_precisions_{int(100*iou_thresh)}'] = list(
                prec[i][label])
            ret_dict[f'{label2cat[label]}_image_ids_{int(100*iou_thresh)}'] = list(
                image_ids[i][label])
            ret_dict[f'{label2cat[label]}_t_positives_{int(100*iou_thresh)}'] = list(
                tps[i][label])
            ret_dict[f'{label2cat[label]}_f_positives_{int(100*iou_thresh)}'] = list(
                fps[i][label])
            ret_dict[f'{label2cat[label]}_VA_{int(100*iou_thresh)}'] = list(
                vect_angle[i][label])
            ret_dict[f'{label2cat[label]}_QD_{int(100*iou_thresh)}'] = list(
                quat_dist[i][label])
            ret_dict[f'{label2cat[label]}_RE_{int(100*iou_thresh)}'] = list(
                roll_err[i][label])
            ret_dict[f'{label2cat[label]}_PE_{int(100*iou_thresh)}'] = list(
                pitch_err[i][label])
            ret_dict[f'{label2cat[label]}_YE_{int(100*iou_thresh)}'] = list(
                yaw_err[i][label])
            ret_dict[f'{label2cat[label]}_ED_{int(100*iou_thresh)}'] = list(
                euclid_err[i][label])

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    # Add GT count
    ret_dict['gt_counts'] = label_gt_count

    return ret_dict
