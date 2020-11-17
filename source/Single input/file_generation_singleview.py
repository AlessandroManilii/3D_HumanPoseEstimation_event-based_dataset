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
train_subjects = [1,2,3,4,5,6,7,8,9]
val_subjects = [10,11,12]
sessions = [1,2,3,4,5]
moves = [[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6,],[1,2,3,4,5,6,7]]

# Projection matrices saved in DHP19 folder
p_mat_cam3 = np.load('/.../P_matrices/P2.npy')
p_mat_cam2 = np.load('/.../P_matrices/P3.npy')

# Counter for data generator
count = 0

# Gaussian blur filter
def decay_mask(heatmap, sigma2=2):
    mask = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    mask /= np.max(mask)
    return mask

for subj in train_subjects:
    for session in sessions:
        for move in moves[sessions.index(session)]:
            if (subj == 1 and session == 4 and move == 2) or (subj == 4 and session == 3 and (move == 4 or move == 6)):
                continue
            else:
                path_x = '/.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
                x_path = join(path_x + '.h5')
                x_h5 = h5py.File(x_path, 'r')

                path_y = '/.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
                y_path = join(path_y + '_label.h5')
                y_h5 = h5py.File(y_path, 'r')

                # Create label mask (260x344x13 array with ones in correspondence to joints predicted positions )
                frames = y_h5['XYZ'].shape[0]

                for frame in range(frames):
                    for cam_id in [2, 3]:
                        # Load projection matrix for specific cam
                        if cam_id == 2:
                            p_mat_cam = p_mat_cam2
                        else:
                            p_mat_cam = p_mat_cam3

                        # y_train generation
                        y_homog = np.concatenate([y_h5['XYZ'][frame], np.ones([1, joints])], axis=0)
                        y_frame = np.zeros(shape=(img_rows, img_cols, joints))
                        y_blur = np.empty(shape=(260, 344, 13))
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
                                y_blur[:, :, j_id] = decay_mask(y_frame[:, :, j_id])
                                is_good = True
                        if is_good == True:
                            # Path where to save y files
                            np.save('/.../x{}.npy'.format(count), x_h5['DVS'][frame, :, :344, cam_id])
                            np.save('/.../y{}.npy'.format(count), y_blur)

                            count += 1

        print('subject {} session {}'.format(subj,session))
    print('subject {} frames {}'.format(subj, count))
print('training frames'.format(count))

val_file = 0

for subj in val_subjects:
    for session in sessions:
        for move in moves[sessions.index(session)]:            
            path_x = '/.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
            x_path = join(path_x + '.h5')
            x_h5 = h5py.File(x_path, 'r')

            path_y = '/.../S{}_session{}_mov{}_7500events'.format(subj, session, move)
            y_path = join(path_y + '_label.h5')
            y_h5 = h5py.File(y_path, 'r')

            # Create label mask (260x344x13 array with ones in correspondence to joints predicted positions )
            frames = y_h5['XYZ'].shape[0]

            for frame in range(frames):
                for cam_id in [2, 3]:
                    # Load projection matrix for specific cam
                    if cam_id == 2:
                        p_mat_cam = p_mat_cam2
                    else:
                        p_mat_cam = p_mat_cam3

                    # y_train generation
                    y_homog = np.concatenate([y_h5['XYZ'][frame], np.ones([1, joints])], axis=0)
                    y_frame = np.zeros(shape=(img_rows, img_cols, joints))
                    y_blur = np.empty(shape=(260, 344, 13))
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
                            y_blur[:, :, j_id] = decay_mask(y_frame[:, :, j_id])
                            is_good = True
                    if is_good == True:
                    
                        # Path where to save y files
                        np.save('/.../x{}.npy'.format(count),x_h5['DVS'][frame, :, :344, cam_id])
                        np.save('/.../y{}.npy'.format(count), y_blur)

                        count += 1
                        val_file += 1

        print('subject {} session {}'.format(subj,session))
    print('subject {} frames {}'.format(subj, val_file))
print('tot val frames {}'.format(val_file))
print('tot frames {}'.format(count))

