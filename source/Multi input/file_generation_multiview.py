import numpy as np
import h5py
from os.path import join
import cv2

# Input image dimensions
img_rows, img_cols = 260, 344
# Joint number
joints = 13

# Training set population
# Missing values for S1_4_2, S4_3_4, S4_3_6, S14_5_3
# subjects = [1,2,3,4,5,6,7,8,9,10,11,12]
train_subjects = [1,2,3,4,5,6,7,8,9]
val_subjects = [10,11,12]
sessions = [1,2,3,4,5]
moves = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6,7]]

cam_ids = [2,3]

# Projection matrices
p_mat_cam3 = np.load('/.../P2.npy')
p_mat_cam2 = np.load('/.../P3.npy')

# Counter for data generator
count = 0

x_minibatch = np.empty(shape=(img_rows, img_cols))
y_minibatch = np.empty(shape=(img_rows, img_cols, joints))

# Gaussian blur filter
def decay_mask(heatmap, sigma2=2):
    mask = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    mask /= np.max(mask) # keep the max to 1
    return mask

for subj in train_subjects:
  for session in sessions:
    for move in moves[sessions.index(session)]:
      if (subj == 1 and session == 4 and move == 2) or (subj == 4 and session == 3 and (move == 4 or move == 6)):
        continue
      else:
        #path where to find h.5 files
        path = '/.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
        x_path = join(path + '.h5')
        x_h5 = h5py.File(x_path, 'r')
        y_path = join(path + '_label.h5')
        y_h5 = h5py.File(y_path, 'r')

        # Create label mask (260x344x13 array with ones in correspondence to joints predicted positions)
        frames = y_h5['XYZ'].shape[0]

        for frame in range(frames):
          y_blur = np.empty(shape=(2, 260, 344, 13))
          is_good = True
          
          for cam_id in [2,3]:
          #  if is_good == False:
           #   break
            #else:
            # Load projection matrix for specific cam
            if cam_id == 2:
              p_mat_cam = p_mat_cam2
            else:
              p_mat_cam = p_mat_cam3
            # y_train generation
            y_homog = np.concatenate([y_h5['XYZ'][frame], np.ones([1, joints])], axis=0)
            y_frame = np.zeros(shape=(img_rows, img_cols, joints))
            
            for j_id in range(joints):
              y_pix_coords = np.matmul(p_mat_cam, y_homog[:, j_id])
              y_pix_coords = y_pix_coords / y_pix_coords[-1]
              h = (img_rows - y_pix_coords[1]).astype(np.int32)
              w = y_pix_coords[0].astype(np.int32)
              if (np.isnan(y_pix_coords[0]) or np.isnan(y_pix_coords[1]) or h > 260 or w > 344 or h<=0 or w<=0):
                is_good = False
                break
              else:
                y_frame[h-1][w-1][j_id] = 1
                y_blur[cam_ids.index(cam_id), :, :, j_id] = decay_mask(y_frame[:, :, j_id])
                #is_good = True
          if is_good:
            for cam_id in [2,3]:
              x_minibatch = x_h5['DVS'][frame, :, :344, cam_id]
              y_minibatch = y_blur[cam_ids.index(cam_id)]
              #path where to save frames from 2 different cams
              np.save('/.../dataset_{}/x{}.npy'.format(cam_id,count), x_minibatch)
              np.save('/.../dataset_{}/y{}.npy'.format(cam_id,count), y_minibatch)  
            count += 1      
    print('subject {} sess {}'.format(subj,session))
print('training frames {}'.format(count))

val_file = 0

for subj in val_subjects:
  for session in sessions:
    for move in moves[sessions.index(session)]:
      if (subj == 1 and session == 4 and move == 2) or (subj == 4 and session == 3 and (move == 4 or move == 6)):
        continue
      else:
        path = '/.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
        x_path = join(path + '.h5')
        x_h5 = h5py.File(x_path, 'r')
        y_path = join(path + '_label.h5')
        y_h5 = h5py.File(y_path, 'r')

        # Create label mask (260x344x13 array with ones in correspondence to joints predicted positions)
        frames = y_h5['XYZ'].shape[0]

        for frame in range(frames):
          y_blur = np.empty(shape=(2, 260, 344, 13))
          is_good = True
          for cam_id in [2,3]:
            #if is_good == False:
             # break
            #else:
              # Load projection matrix for specific cam
            if cam_id == 2:
              p_mat_cam = p_mat_cam2
            else:
              p_mat_cam = p_mat_cam3
            # y_train generation
            y_homog = np.concatenate([y_h5['XYZ'][frame], np.ones([1, joints])], axis=0)
            y_frame = np.zeros(shape=(img_rows, img_cols, joints))
            
            for j_id in range(joints):
              y_pix_coords = np.matmul(p_mat_cam, y_homog[:, j_id])
              y_pix_coords = y_pix_coords / y_pix_coords[-1]
              h = (img_rows - y_pix_coords[1]).astype(np.int32)
              w = y_pix_coords[0].astype(np.int32)
              if (np.isnan(y_pix_coords[0]) or np.isnan(y_pix_coords[1]) or h > 260 or w > 344 or h<=0 or w<=0):
                is_good = False
                break
              else:
                y_frame[h-1][w-1][j_id] = 1
                y_blur[cam_ids.index(cam_id), :, :, j_id] = decay_mask(y_frame[:, :, j_id])
                #is_good = True
          if is_good:
            for cam_id in [2,3]:
              x_minibatch = x_h5['DVS'][frame, :, :344, cam_id]
              y_minibatch = y_blur[cam_ids.index(cam_id)]
              np.save('/.../dataset_{}/x{}.npy'.format(cam_id,count), x_minibatch)
              np.save('/.../dataset_{}/y{}.npy'.format(cam_id,count), y_minibatch)  
            count += 1      
            val_file += 1
    print('subject {} sess {}'.format(subj,session))
print('tot val frames {}'.format(val_file))
print('total frames {}'.format(count))
