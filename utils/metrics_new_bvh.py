
import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
from smplx_fk import SMPLX_Skeleton
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)

# kinetic, manual
import os
def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def quantized_metrics(predicted_npz_root, gt_npz_root):


    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []


    # for npz in os.listdir(predicted_npz_root):
    #     pred_features_k.append(np.load(os.path.join(predicted_npz_root, 'kinetic_features', npz))) 
    #     pred_features_m.append(np.load(os.path.join(predicted_npz_root, 'manual_features_new', npz)))
    #     gt_freatures_k.append(np.load(os.path.join(predicted_npz_root, 'kinetic_features', npz)))
    #     gt_freatures_m.append(np.load(os.path.join(predicted_npz_root, 'manual_features_new', npz)))

    pred_features_k = [np.load(os.path.join(predicted_npz_root, 'kinetic_features', npz)) for npz in os.listdir(os.path.join(predicted_npz_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_npz_root, 'manual_features_new', npz)) for npz in os.listdir(os.path.join(predicted_npz_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_npz_root, 'kinetic_features', npz)) for npz in os.listdir(os.path.join(gt_npz_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_npz_root, 'manual_features_new', npz)) for npz in os.listdir(os.path.join(gt_npz_root, 'manual_features_new'))]
    
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 

#   T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize(gt_freatures_k)
    # gt_freatures_m = normalize(gt_freatures_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_g': fid_m, 'div_k': div_k, 'div_g' : div_m}
    return metrics


def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    # gt_list = []
    pred_list = []

    for npz in os.listdir(root):
        print(npz)
        if os.path.isdir(os.path.join(root, npz)):
            continue
        joint3d = process_predmotion(os.path.join(root, npz))
        # print(extract_manual_features(joint3d.reshape(-1, 22, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root
        # print('==============after fix root ============')
        # print(extract_manual_features(joint3d.reshape(-1, 22, 3)))
        # print('==============bla============')
        # print(extract_manual_features(joint3d.reshape(-1, 22, 3)))
        # np_dance[:, :3] = root

        joint3d_relative = joint3d.copy()
        joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
        joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]

        np.save(os.path.join(root, 'kinetic_features', npz), extract_kinetic_features(joint3d_relative.reshape(-1, 22, 3)))
        np.save(os.path.join(root, 'manual_features_new', npz), extract_manual_features(joint3d_relative.reshape(-1, 22, 3)))


def process_predmotion(motion_path):
    import numpy as np
    import torch

    data = np.load(motion_path)

    root_pos = data["trans"]  # [T, 3]
    local_q = data["poses"]    # [T, 24*3]
    print(local_q.shape)

    smplx_model = SMPLX_Skeleton()

    root_pos = torch.Tensor(root_pos)                          # [T, 3]
    local_q = torch.Tensor(local_q).view(-1, 24, 3)            # [T, 24, 3]
    length = root_pos.shape[0]

    local_q_72 = local_q.view(length, -1)                     # [T, 72]
    positions = smplx_model.forward(local_q_72, root_pos)     # [T, 24, 3]

    # Step 6: Extract only 22 FineDance joints
    smplx_to_22 = [
        0,   # pelvis
        1,   # left_hip
        2,   # right_hip
        3,   # spine1
        4,   # left_knee
        5,   # right_knee
        6,   # spine2
        7,   # left_ankle
        8,   # right_ankle
        9,   # spine3
        10,  # left_foot
        11,  # right_foot
        12,  # neck
        13,  # left_collar
        14,  # right_collar
        15,  # head
        16,  # left_shoulder
        17,  # right_shoulder
        18,  # left_elbow
        19,  # right_elbow
        20,  # left_wrist
        21   # right_wrist
    ]

    # just for reference since the 21 joints remove index 9 (DO NOT USE)
    # smplx_to_21 = [
    #     0,             # pelvis
    #     1, 4, 7, 9,   # left hip, knee, ankle, foot
    #     2, 5, 8, 10,   # right hip, knee, ankle, foot
    #     3, 6,             # spine1, spine2, spine3
    #     11, 14,        # neck, head
    #     12, 13,        # left_collar, right_collar
    #     15, 17, 19,    # left shoulder, elbow, wrist
    #     16, 18, 20     # right shoulder, elbow, wrist
    # ]

    positions = positions[:, smplx_to_22, :]         # [T, 22, 3]
    positions = positions.reshape(positions.shape[0], -1)  # [T, 63]
    return positions.numpy()


def calc_and_save_feats_gt(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    # gt_list = []
    pred_list = []

    for npz in os.listdir(root):
        print(npz)
        if os.path.isdir(os.path.join(root, npz)):
            continue
        joint3d = process_motion(os.path.join(root, npz))
        # joint3d = np.load(os.path.join(root, npz), allow_pickle=True).item()['pred_position'][:1200,:]
        # print(extract_manual_features(joint3d.reshape(-1, 22, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root
        # print('==============after fix root ============')
        # print(extract_manual_features(joint3d.reshape(-1, 22, 3)))
        # print('==============bla============')
        # print(extract_manual_features(joint3d.reshape(-1, 22, 3)))
        # np_dance[:, :3] = root

        joint3d_relative = joint3d.copy()
        joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
        joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]

        np.save(os.path.join(root, 'kinetic_features', npz), extract_kinetic_features(joint3d_relative.reshape(-1, 22, 3)))
        np.save(os.path.join(root, 'manual_features_new', npz), extract_manual_features(joint3d_relative.reshape(-1, 22, 3)))


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


def set_on_ground(root_pos, local_q_156, smplx_model):
    # root_pos = root_pos[:, :] - root_pos[:1, :]
    floor_height = 0
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    positions = smplx_model.forward(local_q_156, root_pos)
    positions = positions.view(length, -1, 3)   # bxt, j, 3
    
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos, local_q_156

def process_motion(motion_path):
    import numpy as np
    import torch

    data = np.load(motion_path)

    # Step 1: Parse root position and 6D local rotation
    if data.shape[1] == 315:
        root_pos = data[:, :3]
        local_q = data[:, 3:]
    elif data.shape[1] == 319:
        root_pos = data[:, 4:7]
        local_q = data[:, 7:]
    else:
        raise ValueError(f"Unexpected input shape: {data.shape}")

    # Step 2: Initialize FK skeleton model
    smplx_model = SMPLX_Skeleton()

    # Step 3: Convert to tensor and reshape
    root_pos = torch.Tensor(root_pos)                          # [T, 3]
    local_q = torch.Tensor(local_q).view(-1, 52, 6)            # [T, 52, 6]
    local_q = ax_from_6v(local_q)                              # [T, 52, 3]
    length = root_pos.shape[0]

    # Step 4: Set root on ground (optional normalization)
    local_q_156 = local_q.view(length, -1)                     # [T, 156]
    root_pos, local_q_156 = set_on_ground(root_pos, local_q_156, smplx_model)

    # Step 5: Forward kinematics (full 52-joint output)
    positions = smplx_model.forward(local_q_156, root_pos)     # [T, 52, 3]

    # Step 6: Extract only 22 FineDance joints
    smplx_to_22 = [
        0,   # pelvis
        1,   # left_hip
        2,   # right_hip
        3,   # spine1
        4,   # left_knee
        5,   # right_knee
        6,   # spine2
        7,   # left_ankle
        8,   # right_ankle
        9,   # spine3
        10,  # left_foot
        11,  # right_foot
        12,  # neck
        13,  # left_collar
        14,  # right_collar
        15,  # head
        16,  # left_shoulder
        17,  # right_shoulder
        18,  # left_elbow
        19,  # right_elbow
        20,  # left_wrist
        21   # right_wrist
    ]

    # just for reference since the 21 joints remove index 9 (DO NOT USE)
    # smplx_to_21 = [
    #     0,             # pelvis
    #     1, 4, 7, 9,   # left hip, knee, ankle, foot
    #     2, 5, 8, 10,   # right hip, knee, ankle, foot
    #     3, 6,             # spine1, spine2, spine3
    #     11, 14,        # neck, head
    #     12, 13,        # left_collar, right_collar
    #     15, 17, 19,    # left shoulder, elbow, wrist
    #     16, 18, 20     # right shoulder, elbow, wrist
    # ]

    positions = positions[:, smplx_to_22, :]         # [T, 22, 3]
    positions = positions.reshape(positions.shape[0], -1)  # [T, 63]
    return positions.numpy()

if __name__ == '__main__':


    gt_root = '/host_data/van/Danceba/finedance/motion'
    pred_root = '/host_data/van/bvh2smpl/finedance_lda_smpl'
    print('Calculating and saving features')
    calc_and_save_feats(pred_root)
    # calc_and_save_feats_gt(gt_root) # uncomment if not done yet

    print('Calculating metrics')
    # print(gt_root)
    # print(pred_root)
    print(quantized_metrics(pred_root, gt_root))