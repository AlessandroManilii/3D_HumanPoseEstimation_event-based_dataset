import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model
from os.path import join
import h5py
import numpy as np
import cv2
import pandas as pd
import pickle

#where .h5 files are stored
h5_dir = r'C:\ '
#where P_matrices and unprocessed data are stored
dataset_dir = r'C:\ '
P_mat_dir = join(dataset_dir, 'P_matrices')

# costants
H = 260;
W = 344;
num_joints = 13

subjects = [13,14,15,16,17]
sessions = [1,2,3,4,5]
moves = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6,],[1,2,3,4,5,6,7]]

# Cam_id in range [0,1,2,3] corresponds to camera [4,1,3,2]
cam_id = 3

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

# Transform Vicon labels to match DVS camera reference system
def get_2D_label_coords(vicon_xyz, cam_id):
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
    return np.stack((v, u), axis=-1), mask

def move_mpjpe_calculation(test_set, label_set, cam_id):
    # predict with CNN, extract predicted 2D coordinates, and return mpjpe for frames set of a specific move
    # cropping rightmost pixels
    frames = test_set.shape[0]
    mpjpe=[]
    for frame_id in range(frames):
        # frame_id:frame_id + 1 allow select specific frame and mantain correct shape of predict method's argument
        test_frame = test_set[frame_id:frame_id + 1, :, :344]
        prediction = trained_model.predict(np.expand_dims(test_frame, axis=-1))
        y_2d, gt_mask, y_heatmaps = get_2D_label_coords(label_set['XYZ'][frame_id], cam_id)
        p_max_coords = np.zeros(y_2d.shape)
        for j_id in range(y_2d.shape[0]):
            pred_j_map = prediction[0, :, :, j_id] # every prediction saved at index 0 since first dimension cardinality not defined
            # predict max value for each heatmap and keep only the first one if more are present
            p_max_coords_tmp = np.argwhere(pred_j_map == np.max(pred_j_map))
            p_max_coords[j_id] = p_max_coords_tmp[0]
        y_2d = y_2d.astype(np.float)
        # where mask is 0, set gt back to NaN
        y_2d[gt_mask == 0] = np.nan
        dist_2d = np.linalg.norm((y_2d - p_max_coords), axis=-1)
        mpjpe.append(np.nanmean(dist_2d))
        p_coords.append(np.ndarray.tolist(p_max_coords))
    return mpjpe

# Pretrained model import
trained_model=load_model(join(dataset_dir,dataset_dir+'\DHP_CNN.model'), custom_objects={'mse2D': mse2D})

mpjpe_mtx=[]
p_coords=[]

for subj in subjects:
    for session in sessions:
        for move in moves[sessions.index(session)]:
            if (subj == 14 and session == 5 and move == 3):
                mpjpe_mtx.append([])
            else:
                datafile = r'S{}_session{}_mov{}_7500events'.format(subj, session, move)
                path_x = join(h5_dir, datafile + '.h5')
                path_y = join(h5_dir, datafile + '_label.h5')
                x_h5 = h5py.File(path_x, 'r')
                y_h5 = h5py.File(path_y, 'r')
                test_set = x_h5['DVS'][:, :, :344, cam_id]
                mpjpe = move_mpjpe_calculation(test_set, y_h5, cam_id)
                mpjpe_mtx.append(mpjpe)
        print('Subj{}_sess{}'.format(subj,session))

# Save mpjpe values on excel file with each table's row corresponding to a single move
df = pd.DataFrame(mpjpe_mtx)
writer = pd.ExcelWriter(r'C:\ ', engine='xlsxwriter')
df.to_excel(writer,index=False)
writer.save()

# Save max predictions' coordinates in local file
with open(r'C:\ ', "wb") as fp:
    pickle.dump(p_coords, fp)

