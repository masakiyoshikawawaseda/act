#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import glob
import sys
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim

import pickle
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from imitate_episodes import make_policy
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed



def inv_quantize(x, n_sample, vmin=-1, vmax=1):
    """converts again x quantized, into continuous data"""
    bins = np.linspace(vmin, vmax, n_sample)
    steps, dims = x.shape
    dim = int(dims / n_sample)

    ret = []
    for d, x_split in enumerate(np.split(x, dim, axis=1)):
        idx = np.argmax(x_split, axis=1)
        y = np.tile(np.expand_dims(bins, 0), steps)
        ret.append(y[:, idx])

    return np.vstack(ret).T


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=1)
parser.add_argument("--mode", type=str, default="test")

# parser.add_argument('--eval', action='store_true')
parser.add_argument('--onscreen_render', action='store_true')
parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT')
parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

# for ACT
parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
parser.add_argument('--temporal_agg', action='store_true')

# args = parser.parse_args()
args = vars(parser.parse_args())

idx = args['idx']

# load dataset
images = np.load('../../data_0704/{}/images_raw.npy'.format(args['mode']))[idx]
# import ipdb;ipdb.set_trace()
robot_states = np.load('../../data_0704/{}/robot_states.npy'.format(args['mode']))[idx]
robot_state_dim = robot_states.shape[-1] # 4


# define model
set_seed(1)
# command line parameters
ckpt_dir = args['ckpt_dir']
policy_class = args['policy_class']
onscreen_render = args['onscreen_render']
task_name = args['task_name']
batch_size_train = args['batch_size']
batch_size_val = args['batch_size']
num_epochs = args['num_epochs']

# get task parameters
is_sim = task_name[:4] == 'sim_'
# if is_sim:
#     from constants import SIM_TASK_CONFIGS
#     task_config = SIM_TASK_CONFIGS[task_name]
# else:
#     from aloha_scripts.constants import TASK_CONFIGS
#     task_config = TASK_CONFIGS[task_name]
# dataset_dir = task_config['dataset_dir']
# num_episodes = task_config['num_episodes']
# episode_len = task_config['episode_len']
# camera_names = task_config['camera_names']

dataset_dir = '/home/yoshikawa/job/2024/airec/data_0704'
# num_episodes = 3
episode_len = 400
camera_names = ['images']

# fixed parameters
# state_dim = 14
state_dim = 4 # xyz, hand_state
lr_backbone = 1e-5
backbone = 'resnet18'
if policy_class == 'ACT':
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }
elif policy_class == 'CNNMLP':
    policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                        'camera_names': camera_names,}
else:
    raise NotImplementedError

config = {
    'num_epochs': num_epochs,
    'ckpt_dir': ckpt_dir,
    'episode_len': episode_len,
    'state_dim': state_dim,
    'lr': args['lr'],
    'policy_class': policy_class,
    'onscreen_render': onscreen_render,
    'policy_config': policy_config,
    'task_name': task_name,
    'seed': args['seed'],
    'temporal_agg': args['temporal_agg'],
    'camera_names': camera_names,
    'real_robot': not is_sim
}

# ckpt_name = f'policy_best.ckpt'
ckpt_name = 'policy_epoch_5800_seed_0.ckpt'
ckpt_path = os.path.join(ckpt_dir, ckpt_name)

policy = make_policy(policy_class, policy_config)
loading_status = policy.load_state_dict(torch.load(ckpt_path))
print(loading_status)
policy.cuda()
policy.eval()

stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']

query_frequency = policy_config['num_queries']
if config['temporal_agg']:
    query_frequency = 1
    num_queries = policy_config['num_queries']


# Inference
if args['temporal_agg']:
    all_time_actions = torch.zeros([episode_len, episode_len+num_queries, state_dim]).cuda()
target_qpos_list = []
nloop = len(images)
for loop_ct in range(nloop):
    # load data and normalization
    img_t = np.expand_dims(images[loop_ct].transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t / 255.0).float().cuda().unsqueeze(0)
    joint_t = robot_states[loop_ct]
    joint_t = pre_process(joint_t)
    joint_t = torch.from_numpy(joint_t).float().cuda().unsqueeze(0)

    # prediction
    if loop_ct % query_frequency == 0:
        all_actions = policy(joint_t, img_t)
    if args['temporal_agg']:
        all_time_actions[[loop_ct], loop_ct:loop_ct+num_queries] = all_actions
        actions_for_curr_step = all_time_actions[:, loop_ct]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
    else:
        raw_action = all_actions[:, loop_ct % query_frequency]

    # denormalization
    # raw_action = raw_action.squeeze(0).cpu().numpy()
    raw_action = raw_action.squeeze(0).cpu().detach().numpy()
    action = post_process(raw_action)
    target_qpos = action

    # append data
    target_qpos_list.append(target_qpos)
    
    print("loop_ct:{}, target_qpos:{}".format(loop_ct, target_qpos))

target_qpos = np.array(target_qpos_list)

# plot images
T = len(images)
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=60)


def anim_update(i):
    for j in range(2):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[i, :, :, ::-1])
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot pose
    # ax[2].set_ylim(-np.pi / 2, np.pi / 2)
    # ax[2].set_ylim(-2.0, 2.0)
    ax[1].set_xlim(0, T)
    ax[1].plot(robot_states[1:], linestyle="dashed", c="k")
    for robot_state_idx in range(robot_state_dim):
        ax[1].plot(np.arange(i + 1), target_qpos[: i + 1, robot_state_idx])
    ax[1].set_xlabel("Step")
    ax[1].set_title("Robot States")


ani = anim.FuncAnimation(fig, anim_update, frames=T)
ani.save("./output/{}_{}_{}.mp4".format(os.path.split(args['ckpt_dir'])[-1], args['mode'], idx), fps=10, writer="ffmpeg")
