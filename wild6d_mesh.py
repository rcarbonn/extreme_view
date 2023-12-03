import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse as ap
import _pickle as cPickle
import os
from PIL import Image
import trimesh


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
    image_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if '.jpg' in f])
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

def handle_intersections(idx_triangles, idx_ray, locs, directions):
    intersections = {}
    ordering = np.argsort(idx_ray)
    idx_ray = idx_ray[ordering]
    idx_triangles = idx_triangles[ordering]
    locs = locs[ordering]

    prev_idx = idx_ray[0]
    intersections[prev_idx] = []
    for i,idx in enumerate(idx_ray):
        if idx != prev_idx:
            prev_idx = idx
            intersections[idx] = []
        intersections[idx].append((np.dot(locs[i], directions[idx]), idx_triangles[i], locs[i]))
    
    filtered_intersections = {}
    for ray_dir in intersections.keys():
        inters = sorted(intersections[ray_dir], key=lambda x: x[0])
        filtered_intersections[ray_dir] = [inters[0], inters[-1]]

    return intersections, filtered_intersections


def virtual_correspondence(data, ids, base_verts, base_faces, viz=False):
    pkl_data = data['pkl_data']
    image_list = data['image_list']
    depth_list = data['depth_list']
    depth_mask_list = data['depth_mask_list']

    merged_mesh = base_verts.copy()
    mean_deltas = np.zeros_like(base_verts)
    transforms = []
    img_sizes = []
    img_grids = []
    K_list = []
    filter_masks = []
    for i,id_ in enumerate(ids):
        transforms.append(pkl_data[id_]['gt_RTs'][0])  # change to pred_RTs for predicted poses 
        K_list.append(np.linalg.inv(pkl_data[id_]['K']))
        img_sizes.append(np.array(Image.open(image_list[id_])).shape)
        depth_img = np.array(Image.open(depth_list[id_]))
        depth_mask = np.array(Image.open(depth_mask_list[id_]))
        mask = depth_mask > 0
        filter_masks.append(mask)
        # img_grids.append(np.mgrid[-img_sizes[i][0]//2:img_sizes[i][0]//2, -img_sizes[i][1]//2:img_sizes[i][1]//2].transpose(1,2,0))
        img_grids.append(np.mgrid[0:img_sizes[i][0], 0:img_sizes[i][1]].transpose(1,2,0))
        mesh_deltas = pkl_data[id_]['mesh_deltas'][0]  # (1,N,3)
        mean_deltas += mesh_deltas
    merged_mesh = merged_mesh + mean_deltas/len(ids)

    # TODO: check why the transformed mesh looks flipped
    mesh1 = trimesh.Trimesh(vertices=merged_mesh, faces=base_faces).apply_transform(transforms[0])
    mesh2 = trimesh.Trimesh(vertices=merged_mesh, faces=base_faces).apply_transform(transforms[1])

    triangle_mesh1 = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh1)
    triangle_mesh2 = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh2)
    # triangle_mesh1 = trimesh.ray.ray_triangle.RayMeshIntersector(mesh1)
    # triangle_mesh2 = trimesh.ray.ray_triangle.RayMeshIntersector(mesh2)

    # direction vectors for ray casting
    directions = []
    for i in range(2):
        img_grids[i] = img_grids[i].reshape(-1,2)
        img_grids[i] = np.hstack((img_grids[i], np.ones((img_grids[i].shape[0], 1))))
        img_grids[i] = img_grids[i][filter_masks[i].flatten()]
        dirs = (K_list[i] @ img_grids[i].T).T
        dirs = dirs / np.linalg.norm(dirs, axis=1).reshape(-1,1)
        directions.append(dirs)

    # ray casting
    idx_triangles1, idx_ray1, locs1 = triangle_mesh1.intersects_id(np.zeros_like(directions[0]), directions[0], return_locations=True)    
    idx_triangles2, idx_ray2, locs2 = triangle_mesh2.intersects_id(np.zeros_like(directions[1]), directions[1], return_locations=True)    

    intersections1, f_intersections1 = handle_intersections(idx_triangles1, idx_ray1, locs1, directions[0])
    intersections2, f_intersections2 = handle_intersections(idx_triangles2, idx_ray2, locs2, directions[1])

    if viz:
        mesh1.show()
        mesh2.show()


def main(args):
    data = load_data(args.data_dir, args.object)
    base_verts, base_faces = data['mesh_data']
    depth_list = data['depth_list']
    depth_mask_list = data['depth_mask_list']

    imgs_ids = [10, 230] # for camera sequence

    virtual_correspondence(data, imgs_ids, base_verts, base_faces, viz=False)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # col_list = ['r', 'b', 'g', 'y', 'm', 'c']
    # merged_mesh = base_verts.copy()
    # mean_deltas = np.zeros_like(base_verts)
    # gtRTs = []
    # depth_img = np.array(Image.open(depth_list[id_]))
    # depth_mask = np.array(Image.open(depth_mask_list[id_]))
    # mask = depth_mask > 0
    # depth_data = depth_img[mask]
    # ax.scatter(gt_3d_box[0, :], gt_3d_box[1,:], gt_3d_box[2,:], c=col_list[i], marker='o')
    
    # ax.scatter([0], [0], [0], c='k', marker='o', s=10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()



if __name__ == '__main__':
    ap = ap.ArgumentParser()
    ap.add_argument('-d', '--data_dir', default='./data/', help='Path to data directory')
    ap.add_argument('-o', '--object', choices=['camera', 'laptop', 'mug'],default='camera', help='Object set to use')
    args = ap.parse_args()
    main(args)