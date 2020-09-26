import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model
from os.path import join
import h5py
import numpy as np
import cv2
import pandas as pd
import pickle
import time

start_time = time.time()

h5_dir = r''
dataset_dir = r''
P_mat_dir = join(dataset_dir, 'P_matrices')

# costants
H = 260;
W = 344;
num_joints = 13

cam_ids = [2,3]

confidence_threshold = 0.1

# Gaussian blur filter
def decay_heatmap(heatmap, sigma2=4):
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma2)
    heatmap /= np.max(heatmap) # keep the max to 1
    return heatmap

# 2D mean square error
def mse2D(y_true, y_pred):
    mean_over_ch = k.mean(k.square(y_pred - y_true), axis=-1)
    mean_over_w = k.mean(mean_over_ch, axis=-1)
    mean_over_h = k.mean(mean_over_w, axis=-1)
    return mean_over_h


def get_2Dcoords_and_heatmaps_label(vicon_xyz, cam_id):
    # From 3D label, get 2D label coordinates and heatmaps for selected camera
    if cam_id==1: P_mat_cam = np.load(join(P_mat_dir,'P1.npy'))
    elif cam_id==3: P_mat_cam = np.load(join(P_mat_dir,'P2.npy'))
    elif cam_id==2: P_mat_cam = np.load(join(P_mat_dir,'P3.npy'))
    elif cam_id == 0: P_mat_cam = np.load(join(P_mat_dir, 'P4.npy'))
    # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1, 13])], axis=0)
    # vincoXYZ is a 4x13 matrix
    coord_pix_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
    coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
    u = coord_pix_homog_norm[0]
    v = H - coord_pix_homog_norm[1]  # flip v coordinate to match the image direction
    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[np.isnan(u)] = 0;
    mask[np.isnan(v)] = 0
    mask[u > W] = 0;
    mask[u <= 0] = 0;
    mask[v > H] = 0;
    mask[v <= 0] = 0
    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)
    # initialize, fill and smooth the heatmaps
    label_heatmaps = np.zeros((H, W, num_joints))
    for fmidx, zipd in enumerate(zip(v, u, mask)):
        if zipd[2] == 1: # write joint position only when projection within frame boundaries
            label_heatmaps[zipd[0], zipd[1], fmidx] = 1
            label_heatmaps[:, :, fmidx] = decay_heatmap(label_heatmaps[:, :, fmidx])
    return np.stack((v, u), axis=-1), mask, label_heatmaps

# import model
trained_model=load_model(r'', custom_objects={'mse2D': mse2D}, compile=False)

def move_mpjpe_calculation(test_set, label_set):
    # predict with CNN, extract predicted 2D coordinates, and return mpjpe for frames set of a specific move
    # cropping rightmost pixels
    frames = test_set.shape[0]
    mpjpe2=[]
    mpjpe3=[]
    p_max_coords = np.zeros((num_joints,2))
    for frame_id in range(frames):
        # frame_id:frame_id + 1 allow select specific frame and mantain correct shape of predict method's argument
        test_frames = [test_set[frame_id:frame_id+1, :, :344, 2], test_set[frame_id:frame_id+1, :, :344, 3]]
        #to expand dims for predict
        test_frames[0] = np.expand_dims(test_frames[0], axis=-1)
        test_frames[1] = np.expand_dims(test_frames[1], axis=-1)
        prediction = trained_model.predict(test_frames)
        for cam_idx in cam_ids:
            y_2d, gt_mask, y_heatmaps = get_2Dcoords_and_heatmaps_label(label_set['XYZ'][frame_id], cam_idx)
            np.reshape(p_max_coords, y_2d.shape)
            for j_id in range(y_2d.shape[0]):
                cam = cam_ids.index(cam_idx)
                pred_j_map = prediction[cam][0, :, :, j_id]  # every prediction saved at index 0 since first dimension cardinality not defined
                # predict max value for each heatmap and keep only the first one if more are present
                is_all_zero = np.all(p_max_coords[j_id] == 0)
                if np.max(pred_j_map) >= confidence_threshold or is_all_zero:
                    p_max_coords_tmp = np.argwhere(pred_j_map == np.max(pred_j_map))
                    p_max_coords[j_id] = p_max_coords_tmp[0]
            y_2d = y_2d.astype(np.float)
            # where mask is 0, set gt back to NaN
            y_2d[gt_mask == 0] = np.nan
            dist_2d = np.linalg.norm((y_2d - p_max_coords), axis=-1)
            if cam_idx == 2:
                mpjpe3.append(np.nanmean(dist_2d))
                p_coords3.append(np.ndarray.tolist(p_max_coords))
            else:
                mpjpe2.append(np.nanmean(dist_2d))
                p_coords2.append(np.ndarray.tolist(p_max_coords))
    return mpjpe2, mpjpe3

subjects = [13,14,15,16,17]
sessions = [1,2,3,4,5]
moves = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6,],[1,2,3,4,5,6,7]]

mpjpe_mtx_2=[]
mpjpe_mtx_3=[]
p_coords2=[]
p_coords3=[]

for subj in subjects:
    for session in sessions:
        for move in moves[sessions.index(session)]:
            if (subj == 14 and session == 5 and move == 3):
                mpjpe_mtx_2.append([])
                mpjpe_mtx_3.append([])
            else:
                datafile = r'S{}_session{}_mov{}_7500events'.format(subj, session, move)
                path_x = join(h5_dir, datafile + '.h5')
                path_y = join(h5_dir, datafile + '_label.h5')
                x_h5 = h5py.File(path_x, 'r')
                y_h5 = h5py.File(path_y, 'r')
                test_set = x_h5['DVS'][:, :, :344]
                mpjpe2, mpjpe3 = move_mpjpe_calculation(test_set, y_h5)                
                mpjpe_mtx_2.append(mpjpe2)
                mpjpe_mtx_3.append(mpjpe3)
        print('Subj{}_sess{}'.format(subj,session))


df = pd.DataFrame(mpjpe_mtx_2)
writer = pd.ExcelWriter(r'', engine='xlsxwriter')
df.to_excel(writer,index=False)
writer.save() 

with open(r'',"wb") as fp:  # Pickling
    pickle.dump(p_coords2, fp)

df = pd.DataFrame(mpjpe_mtx_3)
writer = pd.ExcelWriter(r'', engine='xlsxwriter')
df.to_excel(writer,index=False)
writer.save()

with open(r'',"wb") as fp:  # Pickling
    pickle.dump(p_coords3, fp)

print("--- %s seconds ---" % (time.time() - start_time))
