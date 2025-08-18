import os
import numpy as np
from tqdm import tqdm

def is_bvh(fname: str) -> bool:
    if fname[-4:] == ".bvh":
        return True
    return False

def is_npy(fname: str) -> bool:
    if fname[-4:] == ".npy":
        return True
    return False

def list_bvh_in_dir(dir):
    '''
    List all files in dir.

    Args:
        dir: directory to retreive file names from
    '''

    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(f"{dir}/{f}") and is_bvh(f)]
    files.sort()
    return files

def list_npy_in_dir(dir):
    '''
    List all files in dir.

    Args:
        dir: directory to retreive file names from
    '''

    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(f"{dir}/{f}") and is_npy(f)]
    files.sort()
    return files

def measure_jitter(joint_pos, fps):
    jitter = (joint_pos[3:] - 3 * joint_pos[2:-1] + 3 * joint_pos[1:-2] - joint_pos[:-3]) * (fps ** 3) 
    jitter = np.linalg.norm(jitter, axis=2) # [297, 19]
    jitter = jitter.mean() 
    return jitter

def measure_jitter_npy(dir:str, fps:int):
    print('Computing jitter metric:')
    print(f' - {dir}')
    file_list = list_npy_in_dir(dir)
    total_jitter = np.zeros([len(file_list)]) # one jitter metric for one motion data

    jitter_bar = tqdm(range(len(file_list)))
    for i in jitter_bar:
        fname = file_list[i]
        full_data_dir = f"{dir}/{fname}"

        data = np.load(full_data_dir, allow_pickle=True).item()['pred_position'][:, :] 
        joint_pos = np.array(data).reshape(-1, 22, 3)
        jitter = measure_jitter(joint_pos, fps)
        total_jitter[i] = jitter

    jitter_mean = total_jitter.mean()
    print(f"Total mean of jitter of {len(file_list)} motions: {jitter_mean}")


if __name__ == "__main__":
    ''' 
    Preprocess setup
    '''
    fps = 30
    '''
    Compute jitter metric for all motions and get mean
    '''
    data_dir = "/host_data/van/Danceba/finedance/cc_motion_gpt/eval/pkl/ep000300"
    
    # npy
    measure_jitter_npy(data_dir, fps)
    