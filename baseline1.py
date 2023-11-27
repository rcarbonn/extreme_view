import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse as ap
import _pickle as cPickle
import os
from scipy.spatial.transform import Rotation as sROT

def load_data(data_dir, object_type):
    image_path = os.path.join(data_dir, object_type, 'raw', 'images')
    wild6d_result_path = os.path.join(data_dir, object_type, 'results')
    image_list = [os.path.join(image_path, f) for f in os.listdir(image_path)]
    pkl_list = sorted([os.path.join(wild6d_result_path, f) for f in os.listdir(wild6d_result_path) if '.pkl' in f])
    pkl_data = [cPickle.load(open(pkl, 'rb')) for pkl in pkl_list]
    data = {
        'image_list': image_list,
        'pkl_data': pkl_data
    }
    return data

def main(args):
    data = load_data(args.data_dir, args.object)
    d0 = data['pkl_data'][0]
    gt_pose0 = d0['gt_RTs']
    pred_pose0 = d0['pred_RTs']
    gt_pose0_inv = np.linalg.inv(gt_pose0[0])
    pred_pose0_inv = np.linalg.inv(pred_pose0[0])
    rbase = gt_pose0_inv[:3, :3]
    pred_base = pred_pose0_inv[:3, :3]
    pts_3d_gt = []
    pts_3d_pred = []
    for i in range(1, len(data['pkl_data'])):
        d = data['pkl_data'][i]
        gt_pose = d['gt_RTs']
        pred_pose = d['pred_RTs']
        r_gt =  rbase @ gt_pose[0][:3, :3]
        r_pred = pred_base @ pred_pose[0][:3, :3]
        rvec_gt = np.zeros(3)
        rvec_pred = np.zeros(3)
        cv2.Rodrigues(r_gt, rvec_gt)
        cv2.Rodrigues(r_pred, rvec_pred)
        # rbase = np.linalg.inv(gt_pose[0][:3, :3])
        # pred_base = np.linalg.inv(pred_pose[0][:3, :3])
        print(np.linalg.norm(rvec_gt)*180/np.pi, np.linalg.norm(rvec_pred)*180/np.pi)
        # print(rvec)
        # to_base = gt_pose0_inv[:3, :3] @ gt_pose[0][:3, :3]
        # r = sROT.from_matrix(to_base)
        # r = sROT.from_matrix(gt_pose[0][:3, :3])
        # r = r.as_euler('xyz', degrees=True)
        # t = gt_pose[0][:3, 3]
    #     t_gt = np.linalg.norm(gt_pose[0][:3, 3])
    #     t_pred = np.linalg.norm(pred_pose[0][:3, 3])
    #     t_pred = pred_pose[0][:3,3] * t_gt / t_pred
    #     rvec = np.zeros(1,3)
    #     pts_3d_gt.append(gt_pose[0][:3, 3])
    #     pts_3d_pred.append(t_pred)
    # pts_3d_gt = np.array(pts_3d_gt)
    # pts_3d_pred = np.array(pts_3d_pred)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(pts_3d_gt[:, 0], pts_3d_gt[:, 1], pts_3d_gt[:, 2], c='r', marker='o')
    # ax.scatter(pts_3d_pred[:, 0], pts_3d_pred[:, 1], pts_3d_pred[:, 2], c='b', marker='o')
    # plt.show()

if __name__ == '__main__':
    ap = ap.ArgumentParser()
    ap.add_argument('-d', '--data_dir', default='./data/', help='Path to data directory')
    ap.add_argument('-o', '--object', choices=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'],default='bottle', help='Object set to use')
    args = ap.parse_args()
    main(args)