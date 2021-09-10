import argparse
import numpy as np
import torchvision
import torch
import os.path as osp
import pickle
import json
import os
from envs.env import Env
from softgym.utils.visualization import save_numpy_as_gif
# Ravens dataset generation imports
import cv2
from PIL import Image
from ravens.dataset import Dataset
from absl import app
from absl import flags
# Segmentation imports
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def segment_depthmap(d_map):
	
	#https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
	
    image = d_map * 100 # Denormalize
    img_aux = Image.fromarray(image, 'F') 
    img_aux.show()

    image = image.astype(np.int64)
    print("MAX DEPTH FROM image", np.amax(image))
    print("MIN DEPTH FROM image", np.amin(image))
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    return

def cem_make_gif(env, initial_states, action_trajs, configs, save_dir, save_name, img_size=720):
    all_frames = []
    for i in range(len(action_trajs)):
        frames = []
        env.reset(config=configs[i], initial_state=initial_states[i])
        frames.append(env.get_image(img_size, img_size))
        for action in action_trajs[i]:
            _, reward, _, info = env.step(action, record_continuous_video=True, img_size=img_size)
            frames.extend(info['flex_env_recorded_frames'])
        all_frames.append(frames)
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=5).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))

def make_ravens_db(demo_dir, env, initial_states, action_trajs, configs, seed, img_size=128):
    #C reate ravens Dataset
    print("Save demos to: ", demo_dir)
    dataset = Dataset(demo_dir)
    episode= [] # ravens dataset

    for i in range(len(action_trajs)):
        env.reset(initial_state=initial_states[i])
        color = env.get_image(img_size, img_size) #TODO Hardcoded ik

        j = 0
        for action in action_trajs[i]:
            obs, reward, _, info = env.step(action, img_size=img_size)
            # ravens action encapsules pick&place -- see ravens/environment.py
            # softgym even actions are pick, odd are place
            if j%2==0: 
                action0 = info['picker_pos'][0], [0,0,0,1]
            else: 
                color = env.get_image(img_size, img_size)                
                depth = info['depth_map']
                img = Image.fromarray(depth, 'I') # debug

                obs_dict = {'color': color , 'depth': depth}
                action1= info['picker_pos'][0], [0,0,0,1] # Dont consider rotations
                #reward = info[] # get normalized reward from info.... not used
                action_dict = {'pose0': action0, 'pose1': action1}
                episode.append((obs_dict, action_dict, reward, info))
            j+=1

        #segment_depthmap(depth) #debug

        episode.append((obs_dict, None, reward, info))
        dataset.add(seed, episode)

def get_env(variant_path):
    with open(variant_path, 'r') as f:
        vv = json.load(f)

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'
    vv['env_kwargs']['render'] = True
    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'action_repeat': 1,
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    env = env_class(**env_kwargs)
    return env, vv['env_name']


def get_variant_file(traj_path):
    traj_dir = osp.dirname(traj_path)
    return osp.join(traj_dir, 'variant.json')


def generate_video(file_paths):
    envs, env_names = [], []
    for file_path in file_paths:
        variant_path = get_variant_file(file_path)
        env, env_name = get_env(variant_path)
        envs.append(env)
        env_names.append(env_name)
    for env, env_name, file_path in zip(envs, env_names, file_paths):
        with open(file_path, 'rb') as f:
            traj_dict = pickle.load(f)
        initial_states, action_trajs, configs = traj_dict['initial_states'], traj_dict['action_trajs'], traj_dict['configs']
        cem_make_gif(env, initial_states, action_trajs, configs, save_dir='data/videos/cem', save_name=env_name + '.gif')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()
    generate_video([osp.join(args.exp_dir, subdir, 'cem_traj.pkl') for subdir in os.listdir(args.exp_dir)])
    # file_path = osp.join(args.exp_dir, 'cem_traj.pkl')
    # with open(file_path, 'rb') as f:
    #     traj_dict = pickle.load(f)
    # initial_states, action_trajs, configs = traj_dict['initial_states'], traj_dict['action_trajs'], traj_dict['configs']
    # for action in action_trajs[0]:
    #     print(action)
