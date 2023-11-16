import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse as ap
import _pickle as cPickle
import os

def load_data(data_dir, object_type):
    image_path = os.path.join(data_dir, object_type, 'raw', 'images')
    wild6d_result_path = os.path.join(data_dir, object_type, 'results')
    image_list = [os.path.join(image_path, f) for f in os.listdir(image_path)]
    pkl_list = [os.path.join(wild6d_result_path, f) for f in os.listdir(wild6d_result_path) if '.pkl' in f]
    pkl_data = [cPickle.load(open(pkl, 'rb')) for pkl in pkl_list]
    data = {
        'image_list': image_list,
        'pkl_data': pkl_data
    }
    return data

def main(args):
    data = load_data(args.data_dir, args.object)
    d1 = data['pkl_data'][0]
    gt_pose = d1['gt_RTs']
    pred_pose = d1['pred_RTs']
    print(gt_pose, pred_pose)

if __name__ == '__main__':
    ap = ap.ArgumentParser()
    ap.add_argument('-d', '--data_dir', default='./data/', help='Path to data directory')
    ap.add_argument('-o', '--object', choices=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'],default='bottle', help='Object set to use')
    args = ap.parse_args()
    main(args)