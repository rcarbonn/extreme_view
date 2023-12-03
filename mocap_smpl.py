import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse as ap
import _pickle as cPickle
import os


def load_data(data_dir):
    pkl_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if '.pkl' in f])
    pkl_data = [cPickle.load(open(pkl, 'rb')) for pkl in pkl_list]
    data = {
        'pkl_data': pkl_data
    }
    return data

def main(args):
    data = load_data(args.data_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    col_list = ['r', 'g', 'b', 'y', 'm', 'c']
    sample_list = np.arange(len(data['pkl_data']))
    # n = len(sample_list)//len(col_list)
    # small = sample_list[::n+1]
    # print(small)
    # small = [0, 60, 786]
    small = [786, 848]
    # small = sample_list
    for e,i in enumerate(small):
        pred_output_list = data['pkl_data'][i]['pred_output_list']
        # pred_vertices_bbox = pred_output_list[0]['pred_vertices_smpl']
        pred_vertices_bbox = pred_output_list[0]['pred_joints_vis']
        print(pred_vertices_bbox.shape)
        ax.scatter(pred_vertices_bbox[:, 0], pred_vertices_bbox[:, 1], pred_vertices_bbox[:, 2], c=col_list[e])
        ax.scatter(pred_vertices_bbox[39, 0], pred_vertices_bbox[39, 1], pred_vertices_bbox[39, 2], c='blue', s=200)
    ax.scatter([0], [0], [0], c='k', marker='o', s=100)
    plt.show()

if __name__ == '__main__':
    ap = ap.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='./data/mocap/han_long2')
    args = ap.parse_args()
    main(args)