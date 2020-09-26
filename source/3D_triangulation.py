import numpy as np
import h5py
from os.path import join
import pandas as pd
import pickle

h5_dir = r'...path...'
dataset_dir = r'...path...'
P_mat_dir = join(dataset_dir, 'P_matrices')

# Costants
H = 260;
W = 344;
num_joints = 13

subjects = [13,14,15,16,17]
sessions = [1,2,3,4,5]
moves = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6,7]]

# Importing projection matrix and camera position for used cameras
P_mat_cam2 = np.load(join(P_mat_dir,'P2.npy'))
P_mat_cam3 = np.load(join(P_mat_dir,'P3.npy'))
cameras_pos = np.load(join(P_mat_dir,'camera_positions.npy'))

# Importing saved max predictions' coordinates
with open(r'...path...\/file_name.txt', "rb") as fp2:
    p_max_coords_cam2_ = pickle.load(fp2)

with open(r'...path...\/file_name.txt', "rb") as fp3:
    p_max_coords_cam3_ = pickle.load(fp3)

p_max_coords_cam2 = np.array(p_max_coords_cam2_)
p_max_coords_cam3 = np.array(p_max_coords_cam3_)

# One of the two point used for triangulation is camera center
Point0 = (np.stack((cameras_pos[1],cameras_pos[2])))

def project_uv_xyz_cam(uv, M):
    N = len(uv)
    uv_homog = np.hstack((uv, np.ones((N, 1))))
    M_inv= np.linalg.pinv(M)
    xyz = np.dot(M_inv, uv_homog.T).T
    x = xyz[:, 0] / xyz[:, 3]
    y = xyz[:, 1] / xyz[:, 3]
    z = xyz[:, 2] / xyz[:, 3]
    return x,y,z

def find_intersection(P0,P1):
    # generate all line direction vectors
    n = (P1-P0) / np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized
    # generate the array of all projectors
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (np.matmul(projs,P0[:,:,np.newaxis])).sum(axis=0)
    # solve the least squares problem for the
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q, rcond=None)[0]
    return p.T

def caluculate_3D_mpjpe(label_3d, pred_3d):
    mpjpe_3d_joints = np.linalg.norm((label_3d - pred_3d), axis=-1)
    mpjpe_3d_sample = np.nanmean(mpjpe_3d_joints)
    return mpjpe_3d_sample

# Iterator variable - increment every new frame to take proper element in p_max_coords variables
count = 0

mpjpe_3D_mtx=[]

for subj in subjects:
    for sess in sessions:
        for mov in moves[sessions.index(sess)]:
            if (subj == 14 and sess == 5 and mov == 3):
                mpjpe_3D_mtx.append([])
            else:
                # load input and 3D label files
                datafile = 'S{}_session{}_mov{}_7500events'.format(subj, sess, mov)
                path_y = join(h5_dir,datafile+'_label.h5')
                y_h5 = h5py.File(path_y, 'r')
                frames = y_h5['XYZ'].shape[0]
                mpjpe_3D = []
                for frame in range(frames):
                    # 3D label
                    label_3d = y_h5['XYZ'][frame].T
                    # initialize empty sample of 3D prediction
                    pred_3d = np.zeros(label_3d.shape)
                    pred_2d_cam2_ = np.zeros(p_max_coords_cam2[count].shape)
                    pred_2d_cam3_ = np.zeros(p_max_coords_cam3[count].shape)
                    pred_2d_cam2_[:,0] = p_max_coords_cam2[count][:,1]
                    pred_2d_cam2_[:,1] = H - p_max_coords_cam2[count][:,0]
                    pred_2d_cam3_[:,0] = p_max_coords_cam3[count][:,1]
                    pred_2d_cam3_[:,1] = H - p_max_coords_cam3[count][:,0]
                    x_cam2_pred, y_cam2_pred, z_cam2_pred = project_uv_xyz_cam(pred_2d_cam2_, P_mat_cam2)
                    x_cam3_pred, y_cam3_pred, z_cam3_pred = project_uv_xyz_cam(pred_2d_cam3_, P_mat_cam3)
                    xyz_cam2 = np.stack((x_cam2_pred, y_cam2_pred, z_cam2_pred), axis=1)
                    xyz_cam3 = np.stack((x_cam3_pred, y_cam3_pred, z_cam3_pred), axis=1)
                    for joint_idx in range(13):
                        # coordinates for both cameras of 2nd point of triangulation line
                        Point1 = np.stack((xyz_cam2[joint_idx,:], xyz_cam3[joint_idx,:]), axis=1).T
                        intersection = find_intersection(Point0, Point1)
                        pred_3d[joint_idx] = intersection[0]
                    mpjpe_3D.append(caluculate_3D_mpjpe(label_3d, pred_3d))
                    count += 1
                mpjpe_3D_mtx.append(mpjpe_3D)
        print('Subj{}_Sess{}'.format(subj,sess))

# Save 3D mpjpe values (in mm) in excel file with each table's row corresponding to a single move
df = pd.DataFrame(mpjpe_3D_mtx)
#writer = pd.ExcelWriter(r'C:\Users\aless\Documents\Universita\MAGISTRALE\Computer vision-Deep learning\progetto\test\2D model\test 10 single frame _ save model per epoch\test_single_input_multi_conf\triangulation_conf0.15.xlsx', engine='xlsxwriter')
writer = pd.ExcelWriter(r'...path.../file_name.xlsx', engine='xlsxwriter')
df.to_excel(writer,index=False)
writer.save()
