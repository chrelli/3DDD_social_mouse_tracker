# MAKE A PIPELINE FOR PROCESSING THE COLOR IMAGES!
# FROM TIRAMISU
# IDEA: Add neck to the posture map?
# %matplotlib inline
# %matplotlib widget
# %matplotlib qt
# %load_ext autoreload
# %autoreload 2

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import the tiramisu models
from pytorch_tiramisu.models import tiramisu

# from pose.models import hourglass
import deepfly.pose2d.models as flymodels

# from datasets import camvid
from pytorch_tiramisu.datasets import joint_transforms
import pytorch_tiramisu.utils.imgs
import pytorch_tiramisu.utils.training as train_utils

import sys, os, pickle
import cv2
from colour import Color
import h5py


# Check CUDA

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch_device)


# load the model and set the weights!
from deepfly.pose2d.ArgParse import create_parser

parser = create_parser()
# args = parser.parse_args()
# HACK, set epochs to 200
args = parser.parse_args("--epochs 10 --num-classes 4".split())
args.img_res = [448, 640]
args.num_classes = 11
#
print(args)

global best_acc

# create model
print(
    "==> creating model '{}', stacks={}, blocks={}".format(
        args.arch, args.stacks, args.blocks
    )
)

model = flymodels.__dict__[args.arch](
    num_stacks=args.stacks,
    num_blocks=args.blocks,
    num_classes=args.num_classes,
    num_feats=args.features,
    inplanes=args.inplanes,
    init_stride=args.stride,
)

model = torch.nn.DataParallel(model).cuda()



# A HELPER FUNCTION WHICH SAVES THE STATE OF THE NETWORK, maybe every 10 epochs or smth?
import os
import sys
import math
import string
import random
import shutil
import glob
epoch = 55


# save_weights(model, 0, 1000)
# def load_weights(model, epoch):
WEIGHTS_PATH = '/media/chrelli/Elements/Example3D_compressed/weights/'
# the the most recent from that epoch
all_options = sorted( glob.glob(WEIGHTS_PATH + '/singlecore_weights_epoch_'+str(epoch)+'*' ) )
print(all_options)
weights_fpath = all_options[0]
print("loading weights '{}'".format(weights_fpath))
model.load_state_dict( torch.load(weights_fpath) )
model.eval()
print('loaded!')




# load a stacvk of some sorted frames!

import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
# from deepposekit import VideoReader, KMeansSampler
import sys
# sys.path.append('/home/chrelli/git/3d_sandbox/mouseposev0p1/chrelli_annotator/')
# sys.path.append('/home/chrelli/git/3d_sandbox/mousepose_0p1/deepposekit-annotator/')

# from dpk_annotator import VideoReader, KMeansSampler
import tqdm
import glob
import itertools

from os.path import expanduser
home = expanduser("~")


# set tracking path
tracking_folder = '/home/chrelli/Documents/Example3D_compressed/'
d_files = sorted(glob.glob(tracking_folder + '*d.hdf5'))
c_files = sorted(glob.glob(tracking_folder + '*c.hdf5'))
di_files = sorted(glob.glob(tracking_folder + '*di.hdf5'))
ci_files = sorted(glob.glob(tracking_folder + '*ci.hdf5'))
import pickle
geometry = pickle.load( open( tracking_folder+'/geometry.pkl', "rb" ) )
timing = pickle.load( open( tracking_folder+'/timing.pkl', "rb" ) )
print(d_files)
print(geometry.keys())
print(timing.keys())


import h5py

frame = 0
which_device = 3
show_plots = False
frame_height,frame_width = 480,640

def load_dc_frames(frame,which_device,show_plots = False):
    d_file = d_files[which_device]
    c_file = c_files[which_device]
    di_file = di_files[which_device]
    ci_file = ci_files[which_device]

    with h5py.File(d_file, 'r') as hf:
        d = hf['data'][frame]
    with h5py.File(c_file, 'r') as hf:
        c = hf['data'][frame]
        c = c.reshape((-1,3))
    with h5py.File(di_file, 'r') as hf:
        di = hf['data'][frame]
    with h5py.File(ci_file, 'r') as hf:
        ci = hf['data'][frame]

    #initia;lize
    frame_height,frame_width = 480,640
    d_image = np.zeros((frame_height*frame_width)).astype('uint16')
    rr_image = np.zeros((frame_height*frame_width)).astype('uint8')
    gg_image = np.zeros((frame_height*frame_width)).astype('uint8')
    bb_image = np.zeros((frame_height*frame_width)).astype('uint8')


    d_image[di] = d

    #todo make more efficient
    rr = c[:,0]
    gg = c[:,1]
    bb = c[:,2]

    rr_image[ci] = rr
    gg_image[ci] = gg
    bb_image[ci] = bb


    d_image = d_image.reshape((frame_height,frame_width))
    rr_image = rr_image.reshape((frame_height,frame_width))
    gg_image = gg_image.reshape((frame_height,frame_width))
    bb_image = bb_image.reshape((frame_height,frame_width))
    c_image = np.stack((rr_image,gg_image,bb_image),axis=2)


    if show_plots:
        plt.figure(figsize = (10,5))
        for i, im in zip([1,2,5,6],[d_image,rr_image,gg_image,bb_image]):
            plt.subplot(2,4,i)
            plt.imshow(im)
        plt.subplot(1,2,2)
        plt.imshow(c_image.astype('uint8'))
        plt.title(frame)
        plt.show()
    # RETURNS THE IMAGES IN RGB!
    return c_image,d_image


# WRAP inside of a pytorch dataset
import torch
import torch.utils.data as data
import imgaug.augmenters as iaa

from c_utils.utils_hour import gaussian

class ReadDataset(data.Dataset):
    # todo add augmentation here, clean up and make faster
    # todo remove stupid side effects etc
    def __init__(self, dev):
        '''Initialization'''
        self.dev = dev

        with h5py.File(c_files[dev], 'r') as hf:
            self.n_frames = len(hf['data'])

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_frames

    def __getitem__(self, index):
        # returns the image in RBG
        c_image, _ = load_dc_frames(index,self.dev,show_plots = False)
        # REMEMBER TO CUT DOWN THE TOP of the revolution
        return index, np.moveaxis( c_image[32:,:,[2,1,0]], 2, 0)

# # we shuffle, so that we always see different dumps
dev = 0
FrameLoader = data.DataLoader( ReadDataset(dev) , batch_size=10, shuffle=False, num_workers = 1)


# loop over the dataset and generate the score maps!

# pass through the network
def im_batch_2_scores(im_batch,model):
    inputs = im_batch.float().div(255.).cuda()
    model.eval()
    with torch.no_grad():
        # compute model output
        output = model(inputs)
        # get the resulting scores out! Drop the affinity maps
        scores = output[-1][:,:4,:,:]
    return scores.cpu().numpy()





# prepare the h5py file for the score maps
score_height,score_width = int((480-32)/4),int(640/4)
n_frames = FrameLoader.dataset.__len__()

dev = 3

FrameLoader = data.DataLoader( ReadDataset(dev) , batch_size=50, shuffle=False, num_workers = 1)
if False:
    with h5py.File(tracking_folder+'/'+str(dev)+'_sco.hdf5','w') as h5_dump:
        dset = h5_dump.create_dataset("scores", (1000,4,score_height,score_width), dtype='float32')
        for idx, im_batch in FrameLoader:
            scores = im_batch_2_scores(im_batch,model)
            dset[idx.numpy(),:,:,:] = scores
            if torch.max(idx) > 1000:
                break
        print('DONE!')


# play some tracking!
dev = 0
with h5py.File(tracking_folder+'/'+str(dev)+'_sco.hdf5','r') as h5_dump:
    for i in range(1000):
        cv2.imshow('hmmm',h5_dump['scores'][i,dev,:,:])
        cv2.waitKey(1)

cv2.destroyAllWindows()



from skimage.feature import peak_local_max

frame = 210
def load_score_points(frame,dev):
    xy_list = [None]*4
    pxy_list = [None]*4
    score_idx_list = [None]*4
    with h5py.File(tracking_folder+'/'+str(dev)+'_sco.hdf5','r') as h5_dump:
        sco = h5_dump['scores'][frame,:,:,:]
        for key in range(4):
            xy = peak_local_max(sco[key,:,:],threshold_abs = 0.5,num_peaks = 6)
            xy_list[key] = xy
            pxy_list[key] = sco[key,xy[:,0],xy[:,1]]
            score_idx_list[key] = key * np.ones_like(xy)
#             plt.figure()
#             plt.imshow(sco[key,:,:])
#             plt.plot(xy[:,1],xy[:,0],'or')
#             plt.show()
    return np.concatenate(xy_list), np.concatenate(pxy_list), np.concatenate(score_idx_list)
# load_score_points(frame,dev)



# convert the COLOR IMAGE TO POINTS!
tracking_folder = '/home/chrelli/Documents/Example3D_compressed'

# load the depth scale
geometry = pickle.load( open( tracking_folder+'/geometry.pkl', "rb" ) )
timing = pickle.load( open( tracking_folder+'/timing.pkl', "rb" ) )




# plt.close('all')



#%% also set up the cylinder filtering!
c_cylinder = geometry['c_cylinder']
r_cylinder = geometry['r_cylinder']
floor_point = geometry['floor_point']
floor_normal = geometry['floor_normal']
M0 = geometry['M0']


def apply_rigid_transformation(positions,R,t):
    # takes postions as a Nx3 vector and applies rigid transformation
    # make matrices
    A = np.asmatrix(positions)
    R = np.asmatrix(R)
    t = np.asmatrix(t).T

    # Matrix way:
    n = A.shape[0]
    A2 = np.matmul(R,A.T) + np.tile(t, (1, n))

    # print(str(i)+' after transform: '+str(A2.shape))
    # make it an array?
    return np.asarray(A2.T)


def cut_by_floor_roof(positions,floor_point,floor_normal,floor_cut=0.005,roof_cut=0.01):
    """
    Function to cut away the floor w/o a need to rotate the points fikst, just use the dot product trick
    # cut away floor?
    # use the equation of the plane: http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfPlanes.aspx
    # and evaluate this to check if it's above or below: https://stackoverflow.com/questions/15688232/check-which-side-of-a-plane-points-are-on

    """
    # find the first coefficients of the equation of the plane!
    plane_coeffs = floor_normal

        # find a point above the plane!
    hover_point = floor_point + floor_normal * floor_cut
    roof_point = floor_point + floor_normal * roof_cut
        # calculate d, which is the dot product between a point on the plane and the normal
    floor_d = np.dot(floor_normal,hover_point)
    roof_d = np.dot(floor_normal,roof_point)

    # the idea is to calc ax+by+cz+d where abc is the normal and xyz is the point being tested
    # now do the dot product as the logic to pflip on the sign (don't care about equal to)
    #test_prod = np.dot(positions,plane_coeffs[0:3])
    # einsum is faster!
    test_prod = np.einsum('j,ij->i',plane_coeffs,positions)


    above_logic = (test_prod > floor_d) * (test_prod < roof_d)
    return above_logic

def cheap3d(positions,rgb = None, new=True):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, Y, Z = positions[:,0],positions[:,1],positions[:,2]

    #   3D plot of Sphere
    if new:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.gca()
        ax = ax.add_subplot(111, projection='3d')


    if rgb is None:
        ax.scatter(X, Y, Z, zdir='z', s=10, c='b',rasterized=True)
    else:
        ax.scatter(X, Y, Z, zdir='z', s=6, c=rgb/255,rasterized=True)
#     ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def align_by_floor(positions,floor_point,M0):
    positions = positions - floor_point
    # rotate!
    #TODO desperate need to convert everything to 4D transformations!! Here translation is first, then rotate. Above it's the other way around Yikes!!
    positions = np.transpose(np.matmul(M0,positions.T))

    # cut_logic = (positions[:,2] > 0.01 ) * (positions[:,2] < 0.1 )
    return positions

def cut_by_cylinder(positions,showplot = False):
    dd = np.sqrt( (positions[:,0] - c_cylinder[0])**2 + (positions[:,1] - c_cylinder[1])**2 )

    logic = dd < .99*r_cylinder

    if showplot:

        # easy3d(positions[::10,:])
        positions = positions[logic,:]
        easy3d(positions[:,:])

        plt.figure()
        plt.hist(dd)
        plt.show()

    return logic

def align_d_to_c(d_image,c_image,dev,geometry):
    # todo can be sped up
    pi,pj = np.where( (d_image>0) ) # simply all
    dij = d_image[pi,pj]

    fx,fy,ppx,ppy,depth_scale,fps,frame_width,frame_height = geometry['d_cam_params'][dev]
    fps,frame_width,frame_height = fps.astype('int'),frame_width.astype('int'),frame_height.astype('int')
    fx_c,fy_c,ppx_c,ppy_c,_,_,_,_ = geometry['c_cam_params'][dev]

    z_m = dij*depth_scale # +1e-6

    # and now use pinhole cam function to get the x and y
    x_m = (pj - ppx) * z_m / fx
    y_m = (pi - ppy) * z_m / fy

    # and pack to a stack of positions!
    positions_depth_space = np.vstack((x_m,y_m,z_m)).T

    # swing the depth positions to the color space
    R_extr = geometry['R_extrinsics'][dev]
    t_extr = geometry['t_extrinsics'][dev]
    positions_color_space = np.einsum('ij,aj->ai',R_extr,positions_depth_space) + t_extr

    # now we can caculate cu and cj, the index in the color frame of each point
    ci = np.round(positions_color_space[:,1] * fy_c / positions_color_space[:,2] + ppy_c)
    cj = np.round(positions_color_space[:,0] * fx_c / positions_color_space[:,2] + ppx_c)

    # make sure that they are good (actually, should probably set to zero outside)
    ci = np.clip(ci,0,frame_height-1).astype(int)
    cj = np.clip(cj,0,frame_width-1).astype(int)

    # depth aligned to color

    dac_image = np.zeros((frame_height,frame_width))
    dac_mask = np.zeros((frame_height,frame_width))
    # return the depth in meters
    dac_image[ci,cj] = dij
    dac_mask[ci,cj] = 1
    sigma_g = 3
    # dac_image = cv2.medianBlur(dac_image.astype('uint16'),5)
    # dac_image = cv2.GaussianBlur(dac_image,(sigma_g,sigma_g),0)/cv2.GaussianBlur(dac_mask,(sigma_g,sigma_g),0)
    # dac_image = dac_image[:,:,0]/dac_image[:,:,1]
    return dac_image.astype('uint16')


def cheap4d(pos,keyp,keyp_idx,rgb = None, new=True):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, Y, Z = pos[:,0],pos[:,1],pos[:,2]

    #   3D plot of Sphere
    if new:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = plt.gca()
        ax = ax.add_subplot(111, projection='3d')


    if rgb is None:
        ax.scatter(X, Y, Z, zdir='z', s=10, c='k', alpha = .1,rasterized=True)
    else:
        ax.scatter(X, Y, Z, zdir='z', s=6, c=rgb/255,alpha = .5,rasterized=True)
#     ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)

    body_colors = ['dodgerblue','red','lime','orange']
    for i,body in enumerate(keyp_idx):
        ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[body],rasterized=True)

    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
# plt.close('all')
# cheap4d(pos[::9],keyp,keyp_idx)






# load the point cloud, the keypoints fof a frame!

def load_d_and_keyp(frame,dev):
    # dev = 0
    frame = timing['master_frame_table'][frame,dev]
    c_image, d_image = load_dc_frames(frame,dev,show_plots = False)
    dac_image =  align_d_to_c(d_image,c_image,dev,geometry)
    xy, pxy, score_idx = load_score_points(frame,dev)
    # convert the score points to depth!
    # scale up to the color space, remember the 32 offset
    xy_cij = 4 * xy + np.array([32,0])

    #TODO take average around point!
    xy_d = dac_image[xy_cij[:,0],xy_cij[:,1]]
    xy_d = np.zeros_like(xy[:,0])

#     print("xy")
#     print(xy)
#     print(xy.shape)

#     print("xy_cij")
#     print(xy_cij)
#     print(xy_cij.shape)



    for i in range(xy.shape[0]):
        pixels = dac_image[np.meshgrid( np.arange(-3,4) + xy_cij[i,0], np.arange(-3,4) + xy_cij[i,1])]
#         print(pixels)
        xy_d[i] = np.nanmax( [np.median(pixels[pixels > 0].ravel()) , 0])



    # convert the keypoints to XYZ
    fx,fy,ppx,ppy,depth_scale,fps,frame_width,frame_height = geometry['d_cam_params'][dev]
    fps,frame_width,frame_height = fps.astype('int'),frame_width.astype('int'),frame_height.astype('int')
    fx_c,fy_c,ppx_c,ppy_c,_,_,_,_ =  geometry['c_cam_params'][dev]


    z_c = xy_d*depth_scale # +1e-6

    # and now use pinhole cam function to get the x and y
    x_c = (xy_cij[:,1] - ppx_c) * z_c / fx_c
    y_c = (xy_cij[:,0] - ppy_c) * z_c / fy_c

    # # and pack to a stack of positions!
    keyp_color_space = np.vstack((x_c,y_c,z_c)).T

    # SWING THESE POSITIONS TO THE DEPTH SPACE
    R_extr = geometry['R_extrinsics'][dev]
    t_extr = geometry['t_extrinsics'][dev]
    keyp_depth_space = np.einsum('ij,aj->ai',R_extr.T,(keyp_color_space - t_extr ))

    # also unpack the depth points!
    # get the expanded once more
    pi,pj = np.where( (d_image>0) )

    # pi,pj = np.where( (d>0 ) ) # simply all
    # get the depth of the masked pixels as a raveled list
    dij = d_image[pi,pj]

    # z is easy to calculate, it's just the depth
    z_m = dij*depth_scale # +1e-6
    # z_m = np.clip(z_m,0.,.5)

    # and now use pinhole cam function to get the x and y
    x_m = (pj - ppx) * z_m / fx
    y_m = (pi - ppy) * z_m / fy

    d_positions = np.vstack((x_m,y_m,z_m)).T

    d_world = apply_rigid_transformation(d_positions,geometry['R_world'][dev],geometry['t_world'][dev])
    keyp_world = apply_rigid_transformation(keyp_depth_space,geometry['R_world'][dev],geometry['t_world'][dev])

    return d_world,keyp_world,pxy,score_idx

def load_d_and_keyp_all(frame):
    d_world_list = [None]*4
    keyp_list = [None]*4
    pkeyp_list = [None]*4
    score_idx_list = [None]*4
    for dev in range(4):
        d_world,keyp_world,pkeyp,score_idx = load_d_and_keyp(frame,dev)
        d_world_list[dev] = d_world
        keyp_list[dev] = keyp_world
        score_idx_list[dev] = score_idx
        pkeyp_list[dev] = pkeyp

    return np.concatenate(d_world_list), np.concatenate(keyp_list), np.concatenate(pkeyp_list), np.concatenate(score_idx_list)

def load_full_frame(frame):
    pos, keyp, pkeyp, keyp_idx = load_d_and_keyp_all(frame)

    cut_logic = cut_by_floor_roof(pos,floor_point,floor_normal,floor_cut=0.04,roof_cut=0.25)

    pos = align_by_floor(pos,floor_point,M0)
    keyp = align_by_floor(keyp,floor_point,M0)

    cyl_logic = cut_by_cylinder(pos,showplot = False)


    pos = pos[cyl_logic*cut_logic,:]

    keyp_logic = pkeyp > .6

    return pos, keyp[keyp_logic,:], pkeyp[keyp_logic], keyp_idx[keyp_logic,0]


if __name__ == '__main__':

    pos, keyp, pkeyp, keyp_idx = load_full_frame(358)


    print(pos.shape)
    # pos = np.unique(np.round(pos,axis=0)
    resolution = 0.002
    pos, weights = np.unique(np.round(pos/resolution),axis=0,return_counts=True)
    # pos = pos[weights > 2]*resolution
    pos = pos*resolution

    print(pos.shape)
    print(pkeyp)
    plt.close('all')
    cheap4d(pos[::9,:],keyp,keyp_idx)

# # swing the depth positions to the color space
# AND convert the depth to XYZ

# plt.close('all')
# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(c_image)
# plt.plot(xy_cij[:,1],xy_cij[:,0],'or')
# plt.subplot(2,2,2)
# plt.imshow(dac_image)
# plt.plot(xy_cij[:,1],xy_cij[:,0],'or')
# plt.show()

# load_keys_and_cloud(frame)
