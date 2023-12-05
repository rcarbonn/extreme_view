import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse as ap
import _pickle as cPickle
import os
from PIL import Image
import trimesh
from scipy.spatial.transform import Rotation as sROT

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


def load_mesh(path_to_file, rotx=False):
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
    if rotx:
        Rx = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        vertices = (Rx @ vertices.T).T
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
    face_ids2ray = {}
    debug_segments = {}
    for ray_dir in intersections.keys():
        inters = sorted(intersections[ray_dir], key=lambda x: x[0])
        filtered_intersections[ray_dir] = [inters[0], inters[-1]]
        # debug_segments[ray_dir] = [[np.zeros(3),inters[0][2]], [np.zeros(3), inters[-1][2]]]
        debug_segments[ray_dir] = [np.zeros(3),inters[0][2]]
        if inters[0][1] not in face_ids2ray.keys():
            face_ids2ray[inters[0][1]] = []
        if inters[-1][1] not in face_ids2ray.keys():
            face_ids2ray[inters[-1][1]] = []
        face_ids2ray[inters[0][1]].append(ray_dir)
        face_ids2ray[inters[-1][1]].append(ray_dir)

    return intersections, filtered_intersections, face_ids2ray, debug_segments


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
    gt3dboxes = []
    filter_masks = []
    scales = []
    for i,id_ in enumerate(ids):
        # transforms.append(pkl_data[id_]['gt_RTs'][0])  # change to pred_RTs for predicted poses
        # gtscales = pkl_data[id_]['gt_scales'][0]
        transforms.append(pkl_data[id_]['pred_RTs'][0])  # change to pred_RTs for predicted poses
        gtscales = pkl_data[id_]['pred_scales'][0]
        scale_mat = np.eye(4)
        # print(gtscales/2)
        # scale_mat[:3,:3] = np.diag([gtscales[1], gtscales[1], gtscales[1]])
        # scale_mat[:3,:3] = np.diag([0,0,0])
        scales.append(gtscales)
        gt3dboxes.append(pkl_data[id_]['gt_3d_box'])
        K_list.append(np.linalg.inv(pkl_data[id_]['K']))
        img_sizes.append(np.array(Image.open(image_list[id_])).shape)
        depth_img = np.array(Image.open(depth_list[id_]))
        depth_mask = np.array(Image.open(depth_mask_list[id_]))
        mask = depth_mask > 0
        filter_masks.append(mask)
        # img_grids.append(np.mgrid[-img_sizes[i][0]//2:img_sizes[i][0]//2, -img_sizes[i][1]//2:img_sizes[i][1]//2].transpose(1,2,0))
        img_grids.append(np.mgrid[0:img_sizes[i][0], 0:img_sizes[i][1]].transpose(1,2,0))
        # img_grids.append(np.mgrid[0:img_sizes[i][0], 0:img_sizes[i][1]])
        mesh_deltas = pkl_data[id_]['mesh_deltas'][0]  # (1,N,3)
        mean_deltas += mesh_deltas
    merged_mesh = merged_mesh + mean_deltas/len(ids)

    # TODO: check why the transformed mesh looks flipped
    # print(scales[0])
    # mesh = trimesh.Trimesh(vertices=base_verts, faces=base_faces).apply_transform(scales[0])
    # mesh_s = trimesh.Trimesh(vertices=base_verts, faces=base_faces)
    merged_mesh1 = merged_mesh.copy()
    merged_mesh1[:,0]*=scales[0][0]/(merged_mesh1[:,0].max()-merged_mesh1[:,0].min())
    merged_mesh1[:,1]*=scales[0][1]/(merged_mesh1[:,1].max()-merged_mesh1[:,1].min())
    merged_mesh1[:,2]*=scales[0][2]/(merged_mesh1[:,2].max()-merged_mesh1[:,2].min())
    merged_mesh1_ = np.hstack((merged_mesh1, np.ones((merged_mesh1.shape[0], 1))))
    t1 = np.eye(4)
    t1[:3,:3] = transforms[0][:3,:3].T   
    # t1_ = np.eye(4)
    # t1_[:3,:3] = np.array([[0,0,-1],[0,1,0],[1,0,0]])
    # t1__ = np.eye(4)
    # t1__[:3,:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    # merged_mesh1_ = (t1 @ merged_mesh1_.T).T
    # merged_mesh1_ = (t1 @ merged_mesh1_.T).T
    # merged_mesh1_[:,1]*=-1
    # merged_mesh1_[:,0]*=-1
    # merged_mesh1_[:,2]*=-1
    # merged_mesh1_[:,:3]+=transforms[0][:3,3]

    merged_mesh2 = merged_mesh.copy()
    merged_mesh2[:,0]*=scales[1][0]/(merged_mesh2[:,0].max()-merged_mesh2[:,0].min())
    merged_mesh2[:,1]*=scales[1][1]/(merged_mesh2[:,1].max()-merged_mesh2[:,1].min())
    merged_mesh2[:,2]*=scales[1][2]/(merged_mesh2[:,2].max()-merged_mesh2[:,2].min())
    merged_mesh2_ = np.hstack((merged_mesh2, np.ones((merged_mesh2.shape[0], 1))))
    # t2 = np.eye(4)
    # t2[:3,:3] = transforms[1][:3,:3].T
    # merged_mesh2_ = (t2 @ merged_mesh2_.T).T
    # merged_mesh2_[:,1]*=-1
    # merged_mesh2_[:,0]*=-1
    # merged_mesh2_[:,2]*=-1
    merged_mesh2_[:,:3]-=transforms[1][:3,3]

    r = sROT.from_matrix(transforms[0][:3, :3])
    r = r.as_euler('xyz', degrees=True)
    # transforms[0][:,3] = np.array([0,0,0,1])
    # rotation about z axis by 90 degrees
    r1 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # rotation about y axis by 90 degrees
    # r2 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    # rotation about x axis by 180 degrees
    r3 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    r2 = np.array([[-1,0,0],[0,1,0],[0,0,1]]) # regquired by gt
    r = r2 @ r3
    t = np.eye(4)
    # t[:3,:3] = r
    r3 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    t2 = np.eye(4)
    # t2[:3,:3] = r3
    mesh1 = trimesh.Trimesh(vertices=merged_mesh1, faces=base_faces).apply_transform(transforms[0] @ t)
    mesh2 = trimesh.Trimesh(vertices=merged_mesh2, faces=base_faces).apply_transform(transforms[1] @ t)
    # mesh1 = trimesh.Trimesh(vertices=merged_mesh1, faces=base_faces)
    # mesh2 = trimesh.Trimesh(vertices=merged_mesh2, faces=base_faces)
    # mesh1 = trimesh.Trimesh(vertices=merged_mesh1_[:,:3], faces=base_faces)
    # mesh2 = trimesh.Trimesh(vertices=merged_mesh2_[:,:3], faces=base_faces)

    triangle_mesh1 = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh1)
    triangle_mesh2 = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh2)
    # triangle_mesh1 = trimesh.ray.ray_triangle.RayMeshIntersector(mesh1)
    # triangle_mesh2 = trimesh.ray.ray_triangle.RayMeshIntersector(mesh2)

    # direction vectors for ray casting
    directions = []
    filtered_grids = []
    for i in range(2):
        img_grids[i] = img_grids[i].reshape(-1,2)
        img_grids[i] = np.hstack((img_grids[i], np.ones((img_grids[i].shape[0], 1))))
        # filtered_grids.append(img_grids[i][filter_masks[i].flatten()])
        # img_grids[i] = img_grids[i][filter_masks[i].flatten()]
        filtered_grids.append(img_grids[i])
        # img_grids[i] = img_grids[i][filter_masks[i].flatten()]
        dirs = (K_list[i] @ img_grids[i].T).T
        dirs = dirs / np.linalg.norm(dirs, axis=1).reshape(-1,1)
        directions.append(dirs)

    # ray casting
    idx_triangles1, idx_ray1, locs1 = triangle_mesh1.intersects_id(np.zeros_like(directions[0]), directions[0], return_locations=True)    
    idx_triangles2, idx_ray2, locs2 = triangle_mesh2.intersects_id(np.zeros_like(directions[1]), directions[1], return_locations=True)    


    intersections1, f_intersections1, faceids2ray1, debug_segments1 = handle_intersections(idx_triangles1, idx_ray1, locs1, directions[0])
    intersections2, f_intersections2, faceids2ray2, debug_segments2 = handle_intersections(idx_triangles2, idx_ray2, locs2, directions[1])

    # print(img_grids[i].shape)
    # for idx in faceids2ray1.keys():
    #     print(idx, len(faceids2ray1[idx]))

    face2pixel1 = {}
    face2pixel2 = {}
    for idx in faceids2ray1.keys():
        face2pixel1[idx] = np.array([filtered_grids[0][ray_id].tolist() for ray_id in faceids2ray1[idx]])
        face2pixel1[idx] = np.unique(face2pixel1[idx], axis=0)
    for idx in faceids2ray2.keys():
        face2pixel2[idx] = np.array([filtered_grids[1][ray_id].tolist() for ray_id in faceids2ray2[idx]])
        face2pixel2[idx] = np.unique(face2pixel2[idx], axis=0)
    
    # print(face2pixel1.keys())
    # print(face2pixel2.keys())
    return face2pixel1, face2pixel2
    
    # print(face2pixel1.keys())
    # for k,v in face2pixel1.items():
    #     print(k, v)
    # print(face2pixel1[25], face2pixel1[300])
    # print(face2pixel2[25], face2pixel2[300])

    # save_data = {
    #     'origins' : np.zeros_like(directions[0]),
    #     'ray_pixel1' : filtered_grids[0],
    #     'vertices1' : mesh1.vertices,
    #     'faces1' : mesh1.faces,
    #     'vertices2' : mesh2.vertices,
    #     'faces2' : mesh2.faces,
    #     'base_vertices' : base_verts,
    #     'base_faces' : base_faces,
    #     # 'masked_directions' : directions[0][filter_masks[0].flatten()],
    #     'ray_pixel2' : filtered_grids[1],
    #     'raw_intersections1' : intersections1,
    #     'raw_intersections2' : intersections2,
    #     'filtered_intersections1' : f_intersections1,
    #     'filtered_intersections2' : f_intersections2,
    #     'face2pixel1' : face2pixel1,
    #     'face2pixel2' : face2pixel2,
    #     'img_ids' : ids
    # }
    # np.save('matching_data_camera2.npy', save_data)

    if viz:
        segments = []
        for ray_id in debug_segments1.keys():
            segments.append(np.array(debug_segments1[ray_id]))
        segments = np.array(segments)
        # print(segments.shape)
        # segments = np.random.random((100,2,3))
        # print(segments)
        p = trimesh.load_path(segments[::100])
        # # p.show()
        # print(gt3dboxes[0].shape)
        # p = trimesh.points.PointCloud(gt3dboxes[0].T)
        # mesh.show()
        # add colors to mesh in trimesh
        # mesh.visual.vertex_colors = [0, 0, 255, 255]
        # mesh.visual.face_colors = [0, 0, 255, 255]

        # trimesh.Scene([mesh]).show()
        mesh1.show()
        mesh1.export('./tmeshes/2.obj')
        mesh2.show()
        mesh2.export('./tmeshes/2_.obj')
        # trimesh.Scene([mesh2, p]).show()
        trimesh.Scene([mesh1, p]).show()
        # mesh1.show()
        # mesh2.show()


def main(args):
    data = load_data(args.data_dir, args.object)
    base_verts, base_faces = data['mesh_data']
    depth_list = data['depth_list']
    depth_mask_list = data['depth_mask_list']

    imgs_ids = [10, 230] # for camera sequence
    # imgs_ids = [9, 164] # for mug sequence
    # imgs_ids = [8, 225] # for laptop sequence

    _, _ = virtual_correspondence(data, imgs_ids, base_verts, base_faces, viz=True)
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