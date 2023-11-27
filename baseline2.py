import numpy as np
import cv2
import argparse as ap
import _pickle as cPickle
import os


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


def generate_random_pairs(num_pairs, min_val=1, max_val=200):
    pairs = []
    for i in range(num_pairs):
        pairs.append(np.random.randint(min_val, max_val, size=2))
    return np.array(pairs)

def generate_seq_pairs(num_pairs, min_val=1, max_val=200):
    pairs = []
    for i in range(num_pairs-1):
        pairs.append(np.array([0, i+1]))
    return np.array(pairs)


def recover_pose(kps1, kps2, K, method=cv2.RANSAC):
    E, mask = cv2.findEssentialMat(kps1, kps2, cameraMatrix=K, method=method)
    retval, R, t, mask = cv2.recoverPose(E, kps1, kps2, cameraMatrix=K)
    return E, R, t

def relative_pose(data1, data2):
    K = data1['K']
    kps1_pred = data1['pred_proj_box']
    kps2_pred = data2['pred_proj_box']
    kps1_gt = data1['gt_proj_box']
    kps2_gt = data2['gt_proj_box']
    E_pred, R_pred, t_pred = recover_pose(kps1_pred, kps2_pred, K)
    E_gt, R_gt, t_gt = recover_pose(kps1_gt, kps2_gt, K)
    return R_pred, R_gt

def relative_rotation(R1, R2):
    return np.linalg.inv(R2) @ R1
    # return R1 @ np.linalg.inv(R2)

def compute_rotation_error(sRT_1, sRT_2):
    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
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

def main(args):
    data = load_data(args.data_dir, args.object)
    rand_pairs = generate_random_pairs(100, max_val=len(data['pkl_data']))
    # rand_pairs = generate_seq_pairs(len(data['pkl_data']))
    for p in rand_pairs[:,:]:
        R_pred, R_gt_proj = relative_pose(data['pkl_data'][p[0]], data['pkl_data'][p[1]])
        gt_RTs_0 = data['pkl_data'][p[0]]['gt_RTs']
        gt_RTs_1 = data['pkl_data'][p[1]]['gt_RTs']
        pred_RTs_0 = data['pkl_data'][p[0]]['pred_RTs']
        pred_RTs_1 = data['pkl_data'][p[1]]['pred_RTs']
        R_gt = relative_rotation(gt_RTs_0[0][:3, :3], gt_RTs_1[0][:3, :3])
        R_preds = relative_rotation(pred_RTs_0[0][:3, :3], pred_RTs_1[0][:3, :3])
        err1 = compute_rotation_error(R_pred, R_gt)         # b/w green box and gt
        err2 = compute_rotation_error(R_gt_proj, R_gt)      # b/w red box and gt
        err3 = compute_rotation_error(R_preds, R_gt)
        print("Rotation error with predicions :: {:.3f} \t \
               Rotation error with gt :: {:.3f} \t \
               Rotation error baseline 1 :: {:.3f}".format(err1, err2, err3))


if __name__ == '__main__':
    ap = ap.ArgumentParser()
    ap.add_argument('-d', '--data_dir', default='./data/', help='Path to data directory')
    ap.add_argument('-o', '--object', choices=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'],default='camera', help='Object set to use')
    args = ap.parse_args()
    main(args)