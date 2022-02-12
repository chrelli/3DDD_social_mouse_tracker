#%% Import the nescessary stuff
# basic OS stuff
import time, os, sys, shutil

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# small utilities
import csv
from colour import Color
from itertools import compress # for list selection with logical
from tqdm import tqdm

# for image manipulation
import cv2

# for recording and connecting to the intel realsense librar
#import pyrealsense as pyrs

#import multiprocessing
from multiprocessing import Process

# for cloud handling
from pyntcloud import PyntCloud

# import handy Functions
from utils.common_utils import *
from utils.recording_utils import *
from utils.cloud_utils import *
from utils.reading_utils import *
from utils.localization_utils import *

import h5py

#TODO ideas: save the depth and depth index only as uint 16, as one long string:
# d, di, dj
# for the color, also

#%% check for the most recent recording folder

from utils.reading_utils import most_recent_recording_folders

top_folder_0, top_folder_1 = most_recent_recording_folders()
scene_folders = [top_folder_0,top_folder_0,top_folder_1,top_folder_1]

output_folder = '/media/chrelli/Main SSD/compressed2/'
output_folder = top_folder_0 +'/'
# path_to_dir = path_to_dir.replace(" ", "\\ ")

print(scene_folders)

print('real')
print([check_folder_if_present(ji) for ji in scene_folders])


###################
# Block for reading from npy saving
###################
d_lists = [get_file_shortlist(i,scene_folders[i]+'/npy_raw','d') for i in range(4)]
c_lists = [get_file_shortlist(i,scene_folders[i]+'/npy_raw','cad') for i in range(4)]


###################
# LOAD the camara intrinsics
###################

def read_cam_params(which_device,top_folder,tag = ''):
    # reads the camera parameters of that camera
    this_name = top_folder+'/parameters_'+tag+str(which_device)+'.csv'
    if os.path.exists(this_name):
        raw_list = np.genfromtxt(this_name, delimiter=',')[1,:]
        cam_params = raw_list
        fps_choice,frame_width,frame_height = raw_list[5],raw_list[6],raw_list[7]
    else:
        print('ERROR: '+ this_name+' not found!')
        sys.exit(0)

    return cam_params

d_cam_param_list = [read_cam_params(i,scene_folders[i],'d_') for i in range(4)]
c_cam_param_list = [read_cam_params(i,scene_folders[i],'c_') for i in range(4)]

# ALSO load the extrinsics

R_extr_list = [np.load( scene_folders[i] +'/extr_R_depth_to_color_' + str(i) + '.npy' ).reshape((3,3)) for i in range(4)]
t_extr_list = [np.load( scene_folders[i] +'/extr_t_depth_to_color_' + str(i) + '.npy' ) for i in range(4)]



###################
# LOAD the alignment of the arena
###################
# also load the transformational matrices
from utils.reading_utils import most_recent_calibration_folders
calib_folder_0, calib_folder_1 = most_recent_calibration_folders()
# make a list, a bit silly
calib_folders = [calib_folder_0,calib_folder_0,calib_folder_1,calib_folder_1]
# get a list of the transformations
transformation_list = [load_transformation(i,calib_folders[i]) for i in range(4)]
# tupe will have the content: R_0,t_0,_ =
R_matrices = [transformation_list[i][0] for i in range(4)]
t_vectors = [transformation_list[i][1] for i in range(4)]

# run the function to get the alignment of the arena floor!
M0,floor_point,floor_normal,refined_corners = load_arena_geometry(scene_folders,load_corners=False)

#%% AND also set up the cylinder filtering!
c_cylinder = np.load(scene_folders[0]+'/c_cylinder.npy')
r_cylinder = np.load(scene_folders[0]+'/r_cylinder.npy')



#%% BUT FIRST we save the geometry variables to a json file!
# we will also save the times, even though we don't really need them

# load the master frame table to display!

master_frame_table, reference_time_cam, reference_stamps = load_master_frame_table(scene_folders, allow_reload = False)


time_stamps = [ np.genfromtxt( scene_folders[i]+'/timestamps_'+str(i)+'.csv', delimiter=',') for i in range(4)]

shifted_stamps = [ np.genfromtxt( scene_folders[i]+'/shiftedstamps_'+str(i)+'.csv', delimiter=',') for i in range(4)]


#%% NOW we can loop over the frames!
which_device = 0
show_plots = False
start_frame = 0
# do five mins
end_frame = 60*60*5

geometry = {
    "start_frame": start_frame,
    "end_frame": end_frame,
    "d_cam_params": d_cam_param_list,
    "c_cam_params": c_cam_param_list,
    "R_extrinsics": R_extr_list,
    "t_extrinsics": t_extr_list,
    "R_world": R_matrices,
    "t_world": t_vectors,
    "M0": M0,
    "floor_point": floor_point,
    "floor_normal": floor_normal,
    "c_cylinder": c_cylinder,
    "r_cylinder": r_cylinder,
}

timing = {
    "master_frame_table": master_frame_table,
    "reference_time_cam": reference_time_cam,
    "reference_stamps": reference_stamps,
    "time_stamps": time_stamps,
    "shifted_stamps": shifted_stamps
}

# from here https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
import pickle
with open(output_folder +'geometry.pkl', 'wb+') as f:
    pickle.dump(geometry,f)
with open(output_folder +'timing.pkl', 'wb+') as f:
    pickle.dump(timing,f)


# with open(output_folder +'geometry.pkl', 'rb') as f:
#     geometry = pickle.load(f)
