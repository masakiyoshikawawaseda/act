import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

'''
Load data from npy file
'''

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, camera_names, norm_stats, mode):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.mode = mode
        # self.is_sim = None
        self.is_sim = False
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        dataset_path = os.path.join(self.dataset_dir, self.mode)

        self.len = np.load(os.path.join(dataset_path, 'images_raw.npy')).shape[0]
        images = np.load(os.path.join(dataset_path, 'images_raw.npy'))[index]
        hand_poses = np.load(os.path.join(dataset_path, 'robot_states.npy'))[index]

        original_action_shape = hand_poses.shape
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        qpos = hand_poses[start_ts]
        # import ipdb;ipdb.set_trace()
        image_dict = dict()
        for cam_name in self.camera_names:
                image_dict[cam_name] = images[start_ts]
        # get all actions after and including start_ts
        if self.is_sim:
            action = hand_poses[start_ts:]
            action_len = episode_len - start_ts
        else:
            action = hand_poses[max(0, start_ts - 1):] # hack, to make timesteps more aligned
            action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    # all_action_data = []
    for mode in ['train', 'test']:
        dataset_path = os.path.join(dataset_dir, mode)
        qpos = np.load(os.path.join(dataset_path, 'robot_states.npy'))
        # action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        # all_action_data.append(torch.from_numpy(action))
    # all_qpos_data = torch.stack(all_qpos_data)
    all_qpos_data = torch.cat((all_qpos_data[0], all_qpos_data[1]), dim=0)
    # import ipdb;ipdb.set_trace()
    # all_action_data = torch.stack(all_action_data)
    # all_action_data = all_action_data

    # # normalize action data
    # action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    # action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # action data is just same as qpos data
    action_mean, action_std = qpos_mean, qpos_std

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # # obtain train test split
    # train_ratio = 0.8
    # shuffled_indices = np.random.permutation(num_episodes)
    # train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    # val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_dir, camera_names, norm_stats, mode='train')
    val_dataset = EpisodicDataset(dataset_dir, camera_names, norm_stats, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
