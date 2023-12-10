import numpy as np
import cv2
import argparse as ap
import _pickle as cPickle
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D as l2d

np.random.seed(25)

def load_mesh(path_to_file):
    """ Load obj file.
    Args:
        path_to_file: path
    Returns:
        vertices: ndarray
        faces: ndarray, index of triangle vertices
    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    print(f"Number of vertices: {len(vertices)}, Number of faces: {len(faces)}")
    return vertices, faces


def load_data(data_dir, object_type):
    """ Load evaluation results and images from Wild6D
    Args:
        data_dir: results directory
        object_type: category from ['camera', 'laptop', 'mug']
    Returns:
        data: dictionary containing image_list, pkl_data and mesh_data
    
    Pickle file contains:
    ['gt_class_ids', 'gt_bboxes', 'gt_RTs', 
     'gt_scales', 'gt_handle_visibility', 'pred_class_ids',
     'pred_bboxes', 'pred_scores', 'pred_RTs', 'pred_scales',
     'gt_3d_box', 'gt_proj_box', 'pred_3d_box', 'pred_proj_box',
     'K', 'mesh_deltas', 'network_pose'] 

    """
    image_path = os.path.join(data_dir, object_type, 'raw', 'images')
    wild6d_result_path = os.path.join(data_dir, object_type, 'results2')
    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if ".jpg" in f], key=lambda x: int(x.split('/')[-1].split('.')[0]))
    depth_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if 'depth' in f])
    depth_mask_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if 'mask' in f])
    pkl_list = sorted([os.path.join(wild6d_result_path, f) for f in os.listdir(wild6d_result_path) if '.pkl' in f])
    pkl_data = [cPickle.load(open(pkl, 'rb')) for pkl in pkl_list]
    print(f"Loading mesh data...{object_type}")
    mesh_path = os.path.join(data_dir, 'meshes', f"{object_type}.obj")
    vertices, faces = load_mesh(mesh_path)
    data = {
        'image_list': image_list,
        'depth_list': depth_list,
        'depth_mask_list': depth_mask_list,
        'pkl_data': pkl_data,
        'mesh_data': [vertices, faces]
    }
    return data

def generate_random_pairs(num_pairs, min_val=1, max_val=200):
    pairs = []
    for i in range(num_pairs):
        pairs.append(np.random.randint(min_val, max_val, size=2))
    return np.array(pairs)

def generate_random_pairs2(num_pairs, min_val=1, max_val=200):
    pairs = []
    for i in range(num_pairs):
        j = np.random.randint(min_val, min_val+20, size=1)
        k = np.random.randint(max_val-20, max_val, size=1)
        pairs.append([*j,*k])
    return np.array(pairs)

def generate_seq_pairs(num_pairs, min_val=1, max_val=200):
    pairs = []
    for i in range(num_pairs-1):
        pairs.append(np.array([0, i+1]))
    return np.array(pairs)

def get_3d_bbox(size, shift=0):
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    return bbox_3d.T

def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def visualize_correspondences(image1, image2, pts1, pts2, path):
    img1 = np.array(image1)
    img2 = np.array(image2)

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    cimage = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
    cimage[:h1, :w1, :] = img1
    cimage[:h2, w1:, :] = img2
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(cimage)
    for p1, p2 in zip(pts1,pts2):
        color = np.random.rand(3)
        ax.scatter(p1[0], p1[1], c=color, s=10)
        ax.scatter(p2[0]+w1, p2[1], c=color, s=10)
        l = l2d([p1[0], p2[0]+w1], [p1[1], p2[1]], linewidth=2.0, color=color, alpha=0.8)
        ax.add_line(l)
    ax.axis('off')
    # ax.set_title('Virtual Correspondences')
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()

def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    # if fast_viz:
    #     make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
    #                             color, text, path, show_keypoints, 10,
    #                             opencv_display, opencv_title, small_text)
    #     return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color=['g']*len(mkpts0))

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
    #     fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def rot_auc(errors, thresholds=[20,30,40]):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t*100)
    return aucs