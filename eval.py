import numpy as np
import cv2
import argparse as ap
import _pickle as cPickle
import os
import matplotlib.pyplot as plt
from wild6d_mesh import virtual_correspondence
from utils import *
from PIL import Image

colors = [
    np.array([255, 0, 0]),   # Red
    np.array([0, 255, 0]),   # Green
    np.array([0, 0, 255]),   # Blue
    np.array([255, 255, 0]), # Yellow
    np.array([255, 0, 255]), # Magenta
    np.array([0, 255, 255]), # Cyan
    np.array([128, 0, 0]),  # Maroon
    np.array([0, 128, 0]),  # Green (darker shade)
    np.array([0, 0, 128]),  # Navy
    np.array([128, 128, 0]) # Olive
]


def match_scales(data1, data2, box_type='gt', scales=[0.1, 0.1, 0.1]):
    if box_type=='gt':
        gtrts1 = data1['gt_RTs']
        gtrts2 = data2['gt_RTs']
    elif box_type=='pred':
        gtrts1 = data1['pred_RTs']
        gtrts2 = data2['pred_RTs']

    box1 = get_3d_bbox(scales)
    box3d1 = transform_coordinates_3d(box1, gtrts1[0])
    proj_box1 = calculate_2d_projections(box3d1, data1['K'])

    box2 = get_3d_bbox(scales)
    box3d2 = transform_coordinates_3d(box2, gtrts2[0])
    proj_box2 = calculate_2d_projections(box3d2, data2['K'])
    return proj_box1, proj_box2


def recover_pose(kps1, kps2, K, method=cv2.RANSAC):
    E, mask = cv2.findEssentialMat(kps1, kps2, cameraMatrix=K, method=method)
    retval, R, t, mask = cv2.recoverPose(E, kps2, kps1, cameraMatrix=K)
    return E, R, t


def relative_pose(data1, data2):
    K = data1['K']
    kps1_gt, kps2_gt = match_scales(data1, data2, box_type='gt')
    kps1_pred, kps2_pred = match_scales(data1, data2, box_type='pred')
    E_gt, R_gt, t_gt = recover_pose(kps1_gt, kps2_gt, K)
    E_pred, R_pred, t_pred = recover_pose(kps1_pred, kps2_pred, K)
    return R_pred, R_gt


def relative_rotation(R1, R2):
    # this is what works do dont question it
    return R2 @ np.linalg.inv(R1)


def compute_rotation_error(RT_1, RT_2):
    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    # # symmetric when rotating around y-axis
    # if synset_names[class_id] in ['bottle', 'can', 'bowl'] or \
    #     (synset_names[class_id] == 'mug' and handle_visibility == 0):
    #     y = np.array([0, 1, 0])
    #     y1 = R1 @ y
    #     y2 = R2 @ y
    #     cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    # else:
    R = R1 @ R2.transpose()
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    return theta


def sim_normalized_points(pts):
    x_centroid, y_centroid = np.mean(pts[:, :2], axis=0)
    d_avg = np.mean(np.linalg.norm(pts[:, :2] - np.array([x_centroid, y_centroid]), axis=1))
    s = np.sqrt(2) / d_avg

    T_mat = np.array([
        [s, 0, - s * x_centroid],
        [0, s, - s * y_centroid],
        [0, 0, 1]
    ])

    normalized_pts = (T_mat @ pts.T).T
    return T_mat, normalized_pts


def eval_virtual_correspondence(data, id1, id2, pred_type='gt', viz=False):
    K = data['pkl_data'][id1]['K']
    base_verts, base_faces = data['mesh_data']
    fp1, fp2 = virtual_correspondence(data, [id1, id2], base_verts, base_faces, pred_type=pred_type, viz=False)

    faces1 = set(list(fp1.keys()))
    faces2 = set(list(fp2.keys()))
    common_faces = faces1.intersection(faces2)

    view1 = []
    view2 = []

    for fid in common_faces:
        px1 = fp1[fid]
        px2 = fp2[fid]
        view1.append(px1.mean(axis=0))
        view2.append(px2.mean(axis=0))
    view1 = np.array(view1)
    view2 = np.array(view2)
    T_mat1, sim_points1 = sim_normalized_points(view1)
    T_mat2, sim_points2 = sim_normalized_points(view2)


    F, mask = cv2.findFundamentalMat(sim_points1, sim_points2, method=cv2.RANSAC, ransacReprojThreshold=1e-3)
    F = T_mat2.T @ F @ T_mat1
    inlier_points1 = view1[mask.ravel() == 1]
    inlier_points2 = view2[mask.ravel() == 1]

    E = K.T @ F @ K
    retval, R_vc, t_vc, mask = cv2.recoverPose(E, inlier_points2[:,:2].astype(np.float32), inlier_points1[:,:2].astype(np.float32), cameraMatrix=K)

    if viz:
        img1 = data['image_list'][id1]
        img2 = data['image_list'][id2]
        im1 = cv2.imread(img1)
        im2 = cv2.imread(img2)

        for i, (p1, p2) in enumerate(zip(inlier_points1, inlier_points2)):
            im1 = cv2.circle(im1, p1[:2].astype(int), 1, colors[i%len(colors)].tolist(), 5)
            im2 = cv2.circle(im2, p2[:2].astype(int), 1, colors[i%len(colors)].tolist(), 5)
            if i == 9:
                break
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(im1)
        ax2.imshow(im2)
        plt.show()

    return R_vc, t_vc


def plot_errors(errs):
    errs = np.array(errs)
    print(np.mean(errs[:,3]))
    errs_b1 = errs[:,3]
    errs_b1 = errs_b1[errs_b1 < 80]
    print(np.mean(errs_b1))
    plt.hist(errs[:,3])
    plt.show()
    pass

def main(args):

    data = load_data(args.data_dir, args.object)

    rand_pairs = generate_random_pairs2(100, max_val=len(data['pkl_data']))
    rand_pairs = np.array([[10, 230]]) # test for camera dataset

    errs = {
        'baseline1' : [], # direct R.T @ R between two frames
        'baseline2' : [], # 8 predicted bounding box points from pred_scales/ gt_scales
        'gt_val_errors' : [], # for sanity checks, should be close to zero
        'rot_err_vc' : [], # rotation error between virtual_corr and gt
    }

    for p in rand_pairs[:,:]:

        R_pred, R_gt_proj = relative_pose(data['pkl_data'][p[0]], data['pkl_data'][p[1]])

        R_vc, t_vc = eval_virtual_correspondence(data, p[0], p[1], pred_type=args.pred_type, viz=False)
        
        # raw data
        gt_RTs_0 = data['pkl_data'][p[0]]['gt_RTs']
        gt_RTs_1 = data['pkl_data'][p[1]]['gt_RTs']
        pred_RTs_0 = data['pkl_data'][p[0]]['pred_RTs']
        pred_RTs_1 = data['pkl_data'][p[1]]['pred_RTs']

        R_gt = relative_rotation(gt_RTs_0[0][:3, :3], gt_RTs_1[0][:3, :3]) # truth value for our case

        R_preds = relative_rotation(pred_RTs_0[0][:3, :3], pred_RTs_1[0][:3, :3]) # use for baseline 1

        # baseline 1
        err1 = compute_rotation_error(R_preds, R_gt)

        # baseline 2
        err2 = compute_rotation_error(R_pred, R_gt)         # b/w green box and gt

        # sanity check to see if relative pose is working, should be close to zero
        err3 = compute_rotation_error(R_gt_proj, R_gt)      # b/w red box and gt

        # virtual correspondence error
        err4 = compute_rotation_error(R_vc, R_gt)

        errs['baseline1'].append(err1)
        errs['baseline2'].append(err2)
        errs['gt_val_errors'].append(err3)
        errs['rot_err_vc'].append(err4)

        # print("Error for pair {} :: {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}".format(p, err1, err2, err3, err4))

        print("Rotation errors::  direct pred RTR -> {:.3f} \t \
               from Emat -> {:.3f} \t \
               b/w GT -> {:.3f} \t \
                Virtual Corr -> {:.3f}".format(err1, err2, err3, err4))
    
    # plot_errors(errs)


if __name__ == '__main__':
    ap = ap.ArgumentParser()
    ap.add_argument('-d', '--data_dir', default='./data/', help='Path to data directory')
    ap.add_argument('-o', '--object', choices=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'],default='camera', help='Object set to use')
    ap.add_argument('-p', '--pred_type', choices=['gt', 'pred'], default='gt', help='Transforms to use')
    ap.add_argument('-n', '--num_samples', default=100, help='Transforms to use')
    args = ap.parse_args()
    main(args)