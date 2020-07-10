import numpy as np
import h5py
from os.path import join
import cv2

# Input image dimensions 
img_rows, img_cols = 260, 344
# Joint number
joints = 13
# Set number of frames for a single minibatch
num_of_frames = 8

# Training set population
# Missing values for S1_4_2, S4_3_4, S4_3_6, S14_5_3
subjects = [1,2,3,4,5,6,7,8,9,10,11,12]
sessions = [1,2,3,4,5]
moves = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6,],[1,2,3,4,5,6,7]]

# Projection matrices
p_mat_cam3 = np.load('.../P_mtx/P2.npy')
p_mat_cam2 = np.load('.../P_mtx/P3.npy')

# Counters for data generator
count=0
count_ID=0

minibatch = np.empty(shape=(num_of_frames, img_rows, img_cols))

# Data generator: x_train
for subj in subjects:
  for session in sessions:
    for move in moves[sessions.index(session)]:
      if (subj == 1 and session == 4 and move == 2) or (subj == 4 and session == 3 and (move == 4 or move == 6)):
        continue
      else:
        # Path to h5 files    
        path = '.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
        x_path = join(path + '.h5')
        x_h5 = h5py.File(x_path, 'r')

        frames = x_h5['DVS'].shape[0]
        
        for cam_id in [2,3]:
          
          for frame in range(frames):
            # x_train generation
            minibatch[count%8] = x_h5['DVS'][frame, :, :344, cam_id]            
            count += 1
            if ((count%8) == 0):
              # Path to x files
              np.save('/'.format(count_ID),minibatch)
              count_ID += 1
  print('subject {}'.format(subj))

# Gaussian blur filter
def decay_mask(heatmap, sigma2=2):
    mask = cv2.GaussianBlur(heatmap,(0,0),sigma2)
    return mask

# Data generator: y_train
for subj in subjects:
  for session in sessions:
    for move in moves[sessions.index(session)]:
      if (subj == 1 and session == 4 and move == 2) or (subj == 4 and session == 3 and (move == 4 or move == 6)):
        continue
      else:
        path = '.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
        y_path = join(path + '_label.h5')
        y_h5 = h5py.File(y_path,'r')
        
        # Create label mask (260x344x13 array with ones in correspondence to joints predicted positions )
        frames = y_h5['XYZ'].shape[0]
        
        for cam_id in [2,3]:

          # Load projection matrix for specific cam
          if cam_id == 2:
            p_mat_cam = p_mat_cam2
          else:
            p_mat_cam = p_mat_cam3
          
          for frame in range(frames):
            # y_train generation
            y_pos = np.zeros(shape=(2,joints))
            y_homog = np.concatenate([y_h5['XYZ'][frame], np.ones([1, 13])], axis=0)
            y_frame = np.zeros(shape=(img_rows,img_cols,joints))
            y_blur = np.empty(shape=(260,344,13))
            for j_id in range(joints):
              y_pix_coords = np.matmul(p_mat_cam, y_homog[:,j_id])
              y_pix_coords = y_pix_coords / y_pix_coords[-1]
              h = (img_rows - y_pix_coords[1]).astype(np.int32)
              w = y_pix_coords[0].astype(np.int32)
              y_pos[1,j_id] = h
              y_pos[0,j_id] = w
              y_frame[h][w][j_id] = 1
              y_blur[:,:, j_id] = decay_mask(y_frame[:, :, j_id])

            minibatch[count%8] = y_blur
            count += 1 
            if ((count%8) == 0):
              #path where to save y files
              np.save('/'.format(count_ID),minibatch)
              count_ID += 1
          
    print('subject {}'.format(subj))
