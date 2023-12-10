import numpy as np
import cv2
import argparse as ap
import _pickle as cPickle
import os
import matplotlib.pyplot as plt
from wild6d_mesh import virtual_correspondence
from utils import *
from PIL import Image
from scipy import stats

from eval import compute_rotation_error, relative_rotation
from utils import *

CAMERA_K = np.array(
    [[594.949646,     0.,         242.3452301 ],
 [  0. ,        595.97717285, 320.0859375 ],
 [  0. ,          0.     ,      1.        ]]
)

MUG_K = np.array(
    [[435.30108643,   0.,         242.00372314],
 [  0.,         435.30108643, 319.21679688],
 [  0. ,          0. ,          1.        ]]
)

def main(args):

    if True:
        rand_pairs = np.load('./rand_list.npy')
    else:
        rand_pairs = generate_random_pairs2(args.num_samples, max_val=len(data['pkl_data']))
        np.save('seq_list_new.npy', rand_pairs)

    camera_fmats = np.load('camera_fmats_final.npy', allow_pickle=True).item()
    mug_fmats = np.load('mug_fmats_final.npy', allow_pickle=True).item()
    print(camera_fmats['inlier_pts1'].shape, camera_fmats['inlier_pts2'].shape)
    # camFmats = camera_fmats['Fmats']
    # mugFmats = mug_fmats['Fmats']

    data = load_data(args.data_dir, 'camera')
    cam_list = [i for i in range(100) if i not in camera_fmats['error_idxs']]
    cam_errs = []
    for i in cam_list:
        p = rand_pairs[i]
        F = camera_fmats['Fmats'][i]
        K = CAMERA_K
        inlier_points1 = camera_fmats['inlier_pts1'][i]
        inlier_points2 = camera_fmats['inlier_pts2'][i]
        E = K.T @ F @ K
        retval, R_vc, t_vc, mask = cv2.recoverPose(E, inlier_points2[:,:2].astype(np.float32), inlier_points1[:,:2].astype(np.float32), cameraMatrix=K)

        gt_RTs_0 = data['pkl_data'][p[0]]['gt_RTs']
        gt_RTs_1 = data['pkl_data'][p[1]]['gt_RTs']
        R_gt = relative_rotation(gt_RTs_0[0][:3, :3], gt_RTs_1[0][:3, :3]) # truth value for our case

        err = compute_rotation_error(R_vc, R_gt)
        print("Camera: ", i, err)
        cam_errs.append(err)

    data = load_data(args.data_dir, 'mug')
    mug_list = [i for i in range(100) if i not in mug_fmats['error_idxs']]
    mug_errs = []
    for i in mug_list:
        p = rand_pairs[i]
        F = mug_fmats['Fmats'][i]
        K = MUG_K
        inlier_points1 = mug_fmats['inlier_pts1'][i]
        inlier_points2 = mug_fmats['inlier_pts2'][i]
        E = K.T @ F @ K
        retval, R_vc, t_vc, mask = cv2.recoverPose(E, inlier_points2[:,:2].astype(np.float32), inlier_points1[:,:2].astype(np.float32), cameraMatrix=K)

        gt_RTs_0 = data['pkl_data'][p[0]]['gt_RTs']
        gt_RTs_1 = data['pkl_data'][p[1]]['gt_RTs']
        R_gt = relative_rotation(gt_RTs_0[0][:3, :3], gt_RTs_1[0][:3, :3]) # truth value for our case

        err = compute_rotation_error(R_vc, R_gt)
        print("Mug: ", i, err)
        mug_errs.append(err)
    
    OUTLIER_THRESH = 80
    cam_errs = np.array(cam_errs)
    mug_errs = np.array(mug_errs)
    cam_errs = cam_errs[cam_errs < OUTLIER_THRESH]
    mug_errs = mug_errs[mug_errs < OUTLIER_THRESH]
    aucs_cam = rot_auc(cam_errs)
    aucs_mug = rot_auc(mug_errs)
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t -- Superglue'.format(
        aucs_cam[0], aucs_cam[1], aucs_cam[2], np.mean(cam_errs)))
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t -- Superglue'.format(
        aucs_mug[0], aucs_mug[1], aucs_mug[2], np.mean(mug_errs)))


if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument('-d', '--data_dir', default='./data/', help='Path to data directory')
    ap.add_argument('-o', '--object', choices=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'],default='camera', help='Object set to use')
    ap.add_argument('-p', '--pred_type', choices=['gt', 'pred'], default='gt', help='Transforms to use')
    ap.add_argument('-n', '--num_samples', default=100, help='Transforms to use')
    ap.add_argument('-r', '--rand_list', default=True, help='Transforms to use')
    args = ap.parse_args()
    main(args)