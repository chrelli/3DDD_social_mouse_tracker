#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from IPython import get_ipython


import time, os, sys, shutil
# from utils.fitting_utils import *

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting

import sys, os, pickle
# import cv2
# from colour import Color

import h5py

import glob
import itertools


# and pytorch
import torch


# In[2]:


import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display
# %matplotlib inline

get_ipython().run_line_magic('matplotlib', 'widget')


# In[ ]:





# In[3]:


# def unpack_from_jagged(jagged_line):
#     ''' THE REVESER SO HERE IT UNPACKS AGAIN SO THE DATA CAN BE SAVED
#     AS A JAGGED H5PY DATASET
#     FROM OTHER: Takes the NX3, N, Mx3, M, M shapes and packs to a single float16
#     We ravel the position, ravel the keyp, stack everything and
#     - importantly - we also save M, the number of keypoints'''
#     n_keyp = int(jagged_line[-1])
#     keyp_idx2 = jagged_line[-(1+n_keyp):-1].astype('int')
#     pkeyp2 = jagged_line[-(1+2*n_keyp):-(1+n_keyp)]
#     keyp2 = jagged_line[-(1+5*n_keyp):-(1+2*n_keyp)].reshape((n_keyp,3))
#     block2 = jagged_line[:-(1+5*n_keyp)].reshape((-1,4))
#     pos2,pos_weights2 = block2[:,:3], block2[:,3]
#     # HACK to cut the floor
#     floor_logic = pos2[:,2] > .012
#     pos2 = pos2[floor_logic,:]
#     pos_weights2 = pos_weights2[floor_logic]

#     return pos2, pos_weights2, keyp2, pkeyp2, keyp_idx2

from utils.analysis_tools import unpack_from_jagged
from utils.analysis_tools import particles_to_body_supports_cuda


class data_storage(object):
    def __init__(self):
        # TODO update all this properly
        self.data_path = None
        self.tracking_path = None

        self.jagged_lines = None
        self.has_implant = True
        self.is_running = False

    def load_jagged(self):
        with h5py.File(self.data_path, mode='r') as hdf5_file:
            print("Loading jagged lines from " + self.data_path + "...")
#             print(hdf5_file.keys())
#             print(len(hdf5_file['dataset']))
            self.jagged_lines = hdf5_file['dataset'][...]
        print("Loaded {} jagged lines.".format(len(self.jagged_lines)) )

    def load_tracking(self):
        with open(self.tracking_path, 'rb') as f:
            tracked_behavior = pickle.load(f)
        print(tracked_behavior.keys())
        self.tracked_behavior = tracked_behavior
        self.has_implant = tracked_behavior['has_implant']
        self.start_frame = tracked_behavior['start_frame']
        self.end_frame = tracked_behavior['end_frame']

        # get the raw tracking data!
        part = self.tracked_behavior['tracking_holder']

        # unpack all the 3D coordinates!
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)

        if self.has_implant:
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
            # and the spine length
            s_0 = part[:,2].cpu().numpy()
            s_1 = part[:,2+9].cpu().numpy()

        else:
            body_support_0 = particles_to_body_supports_cuda(part[:,:8],implant = False)
            body_support_1 = particles_to_body_supports_cuda(part[:,8:],implant = False)
            # and the spine length
            s_0 = part[:,2].cpu().numpy()
            s_1 = part[:,2+8].cpu().numpy()

        # add the raw and smoothed coordinates as numpy arrays
        self.body_support_0_raw = [i.cpu().numpy().squeeze() for i in body_support_0]
#         self.body_support_0_smooth = body_support_0_smooth
        self.s_0_raw = s_0
#         self.s_0_smooth = s_0_smooth
        self.body_support_1_raw = [i.cpu().numpy().squeeze() for i in body_support_1]
#         self.body_support_1_smooth = body_support_1_smooth
        self.s_1_raw = s_1
#         self.s_1_smooth = s_1_smooth

    def make_3d_axis(self):
        #   3D plot of the
        fig = plt.figure(figsize = (4.5,4.5))
        ax = fig.add_subplot(111, projection='3d')
        # add to self for use later
        self.fig = fig
        self.ax = ax

    def add_raw_data(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])

        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]

        # add to axis 3D plot of Sphere
        self.h_pc = self.ax.scatter(X, Y, Z, zdir='z', s=2, c='k', alpha = .05,rasterized=False)
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        # loop over the types of body, and make emptyscatter plots
        self.h_kp_list = []
        for body in body_indices:
            h_kp = self.ax.scatter([],[],[], zdir='z', s=25, c=body_colors[body],rasterized=False)
            self.h_kp_list.append(h_kp)

        # THEN set the 3d values to be what the shoud be
        for body in body_indices:
            self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])

        # for axis adjustment
        self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        self.mid_x = (X.max()+X.min()) * 0.5
        self.mid_y = (Y.max()+Y.min()) * 0.5
        self.mid_z = (Z.max()+Z.min()) * 0.5

    def update_raw_data(self,frame):
        # get new raw data!
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
        # update the pointcloud
        self.h_pc._offsets3d = (X,Y,Z)
        # and update the keypoints
        for body in range(4):
            self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])


    def plot_skeleton(self,body_support,color = 'k',body_idx = 0,has_implant = False):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        #print("c_hip is {}".format(c_hip))
        if has_implant:
            p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]
            p_line = [c_nose,c_nose,c_mid,c_impl,c_impl]
            q_line = [c_mid,c_tip,c_ass,c_nose,c_tip]
        else:
            p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip]
            p_line = [c_nose,c_nose,c_mid]
            q_line = [c_mid,c_tip,c_ass]

        # add the body points
        for p in p_skel:
            h_bp = self.ax.scatter(p[0],p[1],p[2],zdir='z', s=50, alpha = 1 , c=color,rasterized=False)
            self.h_bp_list[body_idx].append(h_bp)

        # and the lines between body parts
        for p,q in zip(p_line,q_line):
            h_skel = self.ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],c=color,lw = 4)
            self.h_skel_list[body_idx].append(h_skel)


    def add_skel_fit(self,frame,fit='raw',plot_ellipsoids = True):
        # frame index
        i_frame = frame-self.start_frame

        if fit =='raw':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_raw]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_raw]
            s_0 = self.s_0_raw[i_frame]
            s_1 = self.s_1_raw[i_frame]
        elif fit =='smooth':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_smooth]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_smooth]
            s_0 = self.s_0_smooth[i_frame]
            s_1 = self.s_1_smooth[i_frame]
        else:
            return


        # and plot!
        self.h_skel_list = [[],[]]
        self.h_bp_list = [[],[]]

        self.plot_skeleton(body_support_0,color = 'k',body_idx = 0,has_implant = self.has_implant)
        self.plot_skeleton(body_support_1,color = 'peru',body_idx = 1,has_implant = False)

    def update_skeleton(self,body_support,body_idx = 0, has_implant = False):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        if has_implant :
            p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]
            p_line = [c_nose,c_nose,c_mid,c_impl,c_impl]
            q_line = [c_mid,c_tip,c_ass,c_nose,c_tip]
        else:
            p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip]
            p_line = [c_nose,c_nose,c_mid]
            q_line = [c_mid,c_tip,c_ass]

        # update the body points
        for j,p in enumerate(p_skel):
            self.h_bp_list[body_idx][j]._offsets3d = ([p[0]],[p[1]],[p[2]])

        # update the lines between body parts
        for j,(p,q) in enumerate(zip(p_line,q_line)):
#             # lines are an extra level deep for some stupid matplotlib reason
#             self.h_skel_list[body_idx][j][0].set_xdata([p[0],q[0]])
#             self.h_skel_list[body_idx][j][0].set_ydata([p[1],q[1]])
#             self.h_skel_list[body_idx][j][0].set_3d_properties([p[2],q[2]])
            # new matplotlilb has changed how this is done:

            self.h_skel_list[body_idx][j][0].set_data_3d([p[0],q[0]],[p[1],q[1]],[p[2],q[2]])


    def update_skel_fit(self,frame,fit='raw'):
        # get the data out frame index
        i_frame = frame-self.start_frame
        # speed up this list nonsense
        if fit =='raw':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_raw]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_raw]
            s_0 = self.s_0_raw[i_frame]
            s_1 = self.s_1_raw[i_frame]
        elif fit =='smooth':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_smooth]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_smooth]
            s_0 = self.s_0_smooth[i_frame]
            s_1 = self.s_1_smooth[i_frame]
        else:
            return
        self.update_skeleton(body_support_0,body_idx = 0, has_implant = self.has_implant)
        self.update_skeleton(body_support_1,body_idx = 1, has_implant = False)


    def add_ellip_fit(self,frame,fit='raw',plot_ellipsoids = True):
        # frame index
        i_frame = frame-self.start_frame

        if fit =='raw':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_raw]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_raw]
            s_0 = self.s_0_raw[i_frame]
            s_1 = self.s_1_raw[i_frame]
        elif fit =='smooth':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_smooth]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_smooth]
            s_0 = self.s_0_smooth[i_frame]
            s_1 = self.s_1_smooth[i_frame]
        else:
            return

        self.h_hip_list = [[],[]]
        self.plot_ellipsoids(body_support_0,s_0,color = 'k',body_idx = 0,has_implant=self.has_implant)
        self.plot_ellipsoids(body_support_1,s_1,color = 'peru',body_idx = 1,has_implant=False)

    def add_wireframe_to_axis(self,ax,R_body,c_hip, a_nose,b_nose,a_hip,b_hip,r_impl,style='hip',this_color='k',this_alpha=.4):
        # FIRST PLOT THE ELLIPSE, which is the hip
        # generate points on a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        # get the mesh, by using the equation of an ellipsoid
        if style == 'hip':
            x=np.cos(u)*a_hip
            y=np.sin(u)*np.sin(v)*b_hip
            z=np.sin(u)*np.cos(v)*b_hip
            this_color = 'grey'
        if style == 'nose':
            x=np.cos(u)*a_nose
            y=np.sin(u)*np.sin(v)*b_nose
            z=np.sin(u)*np.cos(v)*b_nose
        if style == 'impl':
            x=np.cos(u)*r_impl
            y=np.sin(u)*np.sin(v)*r_impl
            z=np.sin(u)*np.cos(v)*r_impl

        # pack to matrix of positions
        posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

        # apply the rotatation and unpack
        # posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
        # REMEBRE BODY SUPPORTS ARE [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
        posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip[:,np.newaxis]

        x = posi_rotated[0,:]
        y = posi_rotated[1,:]
        z = posi_rotated[2,:]

        # reshape for wireframe
        x = np.reshape(x, (u.shape) )
        y = np.reshape(y, (u.shape) )
        z = np.reshape(z, (u.shape) )

        h_hip = ax.plot_wireframe(x, y, z, color=this_color,alpha = this_alpha)
        return h_hip

    def plot_ellipsoids(self,body_support,s,color = 'k',body_idx = 0,has_implant=False):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        # this is not so elegant, hm hm
        _, a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl= self.tracked_behavior['body_constants']
        a_hip_delta = a_hip_max - a_hip_min
        b_hip_delta = b_hip_max - b_hip_min
        a_hip_0 = a_hip_min
        b_hip_0 = b_hip_min

        a_hip = a_hip_0 + a_hip_delta * s
        b_hip = b_hip_0 + b_hip_delta * (1.-s)
        d_hip = .75 * a_hip

        if has_implant:
            RRs,ccs,styles = [R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']
        else:
            RRs,ccs,styles = [R_body,R_nose],[c_hip,c_nose],['hip','nose']

        for RR,cc,style in zip(RRs,ccs,styles):

            h_hip = self.add_wireframe_to_axis(self.ax,RR,
                                                   cc,
                                                   a_nose,
                                                   b_nose,
                                                   a_hip,
                                                   b_hip,
                                                   r_impl,
                                                   style=style,this_color=color)
            self.h_hip_list[body_idx].append(h_hip)


    def update_wireframe_lines(self,h_hip,X,Y,Z):
        # h_hip is the handle to the lines3dcollection
        # much of the code is taken from the source of the marplotlib wireframe plotting
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape

        rstride = 1
        cstride = 1

        # We want two sets of lines, one running along the "rows" of
        # Z and another set of lines running along the "columns" of Z.
        # This transpose will make it easy to obtain the columns.
        tX, tY, tZ = np.transpose(X), np.transpose(Y), np.transpose(Z)

        if rstride:
            rii = list(range(0, rows, rstride))
            # Add the last index only if needed
            if rows > 0 and rii[-1] != (rows - 1):
                rii += [rows-1]
        else:
            rii = []
        if cstride:
            cii = list(range(0, cols, cstride))
            # Add the last index only if needed
            if cols > 0 and cii[-1] != (cols - 1):
                cii += [cols-1]
        else:
            cii = []

        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]

        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]

        lines = ([list(zip(xl, yl, zl))
                  for xl, yl, zl in zip(xlines, ylines, zlines)]
                + [list(zip(xl, yl, zl))
                   for xl, yl, zl in zip(txlines, tylines, tzlines)])

        h_hip.set_segments(lines)

    def calculate_wireframe_points(self,R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip'):
        # FIRST PLOT THE ELLIPSE, which is the hip
        # generate points on a sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        # get the mesh, by using the equation of an ellipsoid
        if style == 'hip':
            x=np.cos(u)*a_hip
            y=np.sin(u)*np.sin(v)*b_hip
            z=np.sin(u)*np.cos(v)*b_hip
        if style == 'nose':
            x=np.cos(u)*a_nose
            y=np.sin(u)*np.sin(v)*b_nose
            z=np.sin(u)*np.cos(v)*b_nose
        if style == 'impl':
            x=np.cos(u)*r_impl
            y=np.sin(u)*np.sin(v)*r_impl
            z=np.sin(u)*np.cos(v)*r_impl

        # pack to matrix of positions
        posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

        # apply the rotatation and unpack
        # posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
        # REMEBRE BODY SUPPORTS ARE [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
        posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip[:,np.newaxis]

        x = posi_rotated[0,:]
        y = posi_rotated[1,:]
        z = posi_rotated[2,:]

        # reshape for wireframe
        x = np.reshape(x, (u.shape) )
        y = np.reshape(y, (u.shape) )
        z = np.reshape(z, (u.shape) )

        return x,y,z

    def update_ellipsoids(self,body_support,s,body_idx = 0, has_implant = False):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        # this is not so elegant, hm hm
        # this is STILL not so elegant, hm hm
        _, a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl= self.tracked_behavior['body_constants']
        a_hip_delta = a_hip_max - a_hip_min
        b_hip_delta = b_hip_max - b_hip_min
        a_hip_0 = a_hip_min
        b_hip_0 = b_hip_min

        a_hip = a_hip_0 + a_hip_delta * s
        b_hip = b_hip_0 + b_hip_delta * (1.-s)
        d_hip = .75 * a_hip


        if has_implant:
            RRs,ccs,styles = [R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']
        else:
            RRs,ccs,styles = [R_body,R_nose],[c_hip,c_nose],['hip','nose']

        for jj, (RR,cc,style) in enumerate(zip(RRs,ccs,styles)):

            X,Y,Z = self.calculate_wireframe_points(RR,
                                                   cc,
                                                   a_nose,
                                                   b_nose,
                                                   a_hip,
                                                   b_hip,
                                                   r_impl,
                                                   style=style)
            h_hip = self.h_hip_list[body_idx][jj]
            self.update_wireframe_lines(h_hip,X,Y,Z)

    def update_ellip_fit(self,frame,fit = 'raw'):
        # get the data out frame index
        i_frame = frame-self.start_frame
        # speed up this list nonsense
        if fit =='raw':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_raw]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_raw]
            s_0 = self.s_0_raw[i_frame]
            s_1 = self.s_1_raw[i_frame]
        elif fit =='smooth':
            body_support_0 = [ d[i_frame,...] for d in self.body_support_0_smooth]
            body_support_1 = [ d[i_frame,...] for d in self.body_support_1_smooth]
            s_0 = self.s_0_smooth[i_frame]
            s_1 = self.s_1_smooth[i_frame]
        else:
            return

        self.update_ellipsoids(body_support_0,s_0,body_idx = 0,has_implant = self.has_implant)
        self.update_ellipsoids(body_support_1,s_1,body_idx = 1,has_implant = False)



    def unpack_trace(self,body_support,trace_indices,body_idx = 0,what_type=['hip'],color='k'):
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        type_list = np.array(['hip','ass','mid','nose','tip','impl'])
        c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
        ii_c_list = np.arange(len(type_list))
        # TODO make the decay work!

        for ttt in what_type:
            # this is also not so elegant
            selecta = np.arange(len(type_list))[type_list == ttt]
            dat = c_list[selecta[0]].squeeze()
            X,Y,Z = dat[trace_indices,0],dat[trace_indices,1],dat[trace_indices,2]
            h_trace = self.ax.plot(X,Y,Z,lw=2,c=color,alpha = .65)
            self.h_trace_list[body_idx][ii_c_list[type_list == ttt][0]] = h_trace


    def add_trace(self,frame,trace='raw',trace_length=90,trace_clip = None,decay_factor=.9, type_list = ['nose']):
        # get the particle, convert to torch tensor, calculate body supports
        i_frame = frame-self.start_frame
        # make a holder for the lines
        self.h_trace_list = [[None]*5,[None]*5]

        if trace_clip is not None:
            i_clip = trace_clip-self.start_frame
            i_trace_start = np.max([i_clip, i_frame-trace_length])
        else:
            i_trace_start = np.max([0, i_frame-trace_length])

        #print("i_trace_start is {} and i_frame is {}".format(i_trace_start,i_frame))
        trace_indices = np.arange(i_trace_start,i_frame)

        if trace == 'raw':
            self.unpack_trace(self.body_support_0_raw,trace_indices, body_idx = 0,what_type=type_list,color='black')
            self.unpack_trace(self.body_support_1_raw,trace_indices, body_idx = 1,what_type=type_list,color='peru')
        if trace == 'smooth':
            self.unpack_trace(self.body_support_0_smooth,trace_indices, body_idx = 0,what_type=type_list,color='black')
            self.unpack_trace(self.body_support_1_smooth,trace_indices, body_idx = 1,what_type=type_list,color='peru')

    def update_trace_3dlines(self,body_support,trace_indices,body_idx=0,what_type=['hip']):
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        type_list = np.array(['hip','ass','mid','nose','tip','impl'])
        c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
        ii_c_list = np.arange(len(type_list))
        # TODO make the decay work!

        for ttt in what_type:
            # this is also not so elegant
            selecta = np.arange(len(type_list))[type_list == ttt]
            dat = c_list[selecta[0]].squeeze()
            X,Y,Z = dat[trace_indices,0],dat[trace_indices,1],dat[trace_indices,2]
#             self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_xdata(X)
#             self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_ydata(Y)
#             self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_3d_properties(Z)
            # Ugh matplotlib changed the api, the new way makes much more sense though, so fine..
            self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_data_3d(X,Y,Z)

    def update_trace_fit(self,frame,trace='raw',trace_length=90,trace_clip = None,decay_factor=.9, type_list = None):
        # get the particle, convert to torch tensor, calculate body supports
        i_frame = frame-self.start_frame

        if trace_clip is not None:
            i_clip = trace_clip-self.start_frame
            i_trace_start = np.max([i_clip, i_frame-trace_length])
        else:
            i_trace_start = np.max([0, i_frame-trace_length])

        # these are the indices to plot
        trace_indices = np.arange(i_trace_start,i_frame)

        if trace =='raw':
            body_support_0 = self.body_support_0_raw
            body_support_1 = self.body_support_1_raw
        elif trace =='smooth':
            body_support_0 = self.body_support_0_smooth
            body_support_1 = self.body_support_1_smooth
        else:
            return


        if len(trace_indices)== 0:
            # just skip if there is no trace
            return


        self.update_trace_3dlines(body_support_0,trace_indices,body_idx=0,what_type = type_list)
        self.update_trace_3dlines(body_support_1,trace_indices,body_idx=1,what_type = type_list)

    def finish_3d_axis(self,view_style = 'ex', zoom = False, dump = False):
        # finish the labeling, plot adjustments, dump and show
        ax = self.ax

        if self.max_range is not None:
            ax.set_xlim(self.mid_x - self.max_range, self.mid_x + self.max_range)
            ax.set_ylim(self.mid_y - self.max_range, self.mid_y + self.max_range)
            ax.set_zlim(0, 2*self.max_range)

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        if view_style == 'top':
            az = -30
            el = 90
        if view_style == 'side':
            az = -15
            el = 9
        if view_style == 'mix':
            az = 150
            el = 50
        if view_style == 'ex':
            az = -14
            el = 46

        if view_style == 'ex':
            az = -46
            el = 23

        ax.view_init(elev=el, azim=az)

storage = data_storage()


# In[4]:



play = widgets.Play(
    value=0,
    min=0,
    max=10000,
    step=10,
    interval=100,
    description="Press play",
    disabled=False
)
slider = widgets.IntSlider(value=0,
    min=0,
    max=10000)



def on_value_change(change):
    frame = int(change['new'])
    storage.update_raw_data( change['new'] )
    storage.update_skel_fit( int(change['new']) )
    storage.update_ellip_fit( int(change['new']) )
#     storage.update_trace_fit( int(change['new']) )
    # storage.update_trace_fit(frame)


    storage.fig.canvas.draw()

slider.observe(on_value_change, 'value')

widgets.jslink((play, 'value'),(slider, 'value'))


# In[5]:


data_path_textbox = widgets.Text(
    value='/media/chrelli/SSD4TB/Data0_backup/recording_20201110-105540/pre_processed_frames.hdf5',
    description='Path:'
)

tracking_path_textbox = widgets.Text(
    value='/media/chrelli/SSD4TB/Data0_backup/recording_20201110-105540/tracked_behavior_in_progress.pkl',
    description='Path:'
)


load_button = widgets.Button(
    description='Load data',
)

load_behavior_button = widgets.Button(
    description='Load tracking',
)


# In[6]:


@load_button.on_click
def plot_on_click(b):
    storage.data_path = data_path_textbox.value
    storage.load_jagged()
    # and make the plot
    storage.add_raw_data( int(play.value) )
    storage.finish_3d_axis()
    storage.fig.canvas.draw()
    # set the min and max time to the behavior!
    play.min = 0
    play.max = len(storage.jagged_lines)
    slider.min = 0
    slider.max = len(storage.jagged_lines)

@load_behavior_button.on_click
def plot_on_click2(b):
    storage.tracking_path = tracking_path_textbox.value
    storage.load_tracking()
    storage.add_skel_fit( int(play.value) )
    storage.add_ellip_fit( int(play.value) )
    # storage.add_trace( int(play.value) )

    play.min = storage.tracked_behavior['start_frame']
    play.max = storage.tracked_behavior['end_frame']
    slider.min = storage.tracked_behavior['start_frame']
    slider.max = storage.tracked_behavior['end_frame']


#     # set the min and max time to the tracked behavior!
#     play.min = 0
#     play.max = len(storage.jagged_lines)

    storage.fig.canvas.draw()




# In[7]:


frame_textbox = widgets.BoundedIntText(
    value=0,
    min = 0,
    max = 10000,
    description='Frame #:'
)

jump_frame_button = widgets.Button(
    description='Jump to frame',
)


# In[8]:


@jump_frame_button.on_click
def update_frame(b):
    play.value = frame_textbox.value

#     storage.update_raw_data( frame_textbox.value)
#     storage.fig.canvas.draw()


# In[9]:


fps = 60

time_textbox = widgets.BoundedFloatText(
    value=0,
    min = 0,
    max = 10000/60,
    description='Time [s]:'
)

jump_time_button = widgets.Button(
    description='Jump to time',
)


# In[10]:


@jump_time_button.on_click
def update_time(b):
    play.value = int(time_textbox.value * fps)
#     storage.update_raw_data( int(time_textbox.value * fps) )
#     storage.fig.canvas.draw()


# In[ ]:





# In[11]:


# widgets.jslink((play, 'value'),(frame_textbox, 'value'))


# In[12]:


raw_ok =widgets.Valid(
    value=True,
    indent = True,
    description='Raw data',
)

track_ok = widgets.Valid(
    value=True,
    description='Tracking'
)


# In[13]:




check_raw = widgets.Checkbox(
    value=True,
    description='Display raw data',
    disabled=False,
    indent=True
)

check_skel = widgets.Checkbox(
    value=True,
    description='Display skeleton',
    disabled=False,
    indent=False
)

check_ellip = widgets.Checkbox(
    value=True,
    description='Display ellipsoids',
    disabled=False,
    indent=True
)

check_trace = widgets.Checkbox(
    value=False,
    description='Display trace',
    disabled=False,
    indent=False
)


# In[14]:


sub10_button = widgets.Button(
    description='<< 10',
)

sub5_button = widgets.Button(
    description='< 5',
)

add10_button = widgets.Button(
    description='10 >>',
)

add5_button = widgets.Button(
    description='5 >',
)

@sub10_button.on_click
def update_frame(b):
    play.value = play.value - 10

@sub5_button.on_click
def update_frame(b):
    play.value = play.value - 5

@add5_button.on_click
def update_frame(b):
    play.value = play.value + 5

@add10_button.on_click
def update_frame(b):
    play.value = play.value + 10


# In[15]:


from ipywidgets import AppLayout, GridspecLayout
item_layout = widgets.Layout(margin='0 0 10px 10px')

dashboard = VBox([
    HBox([data_path_textbox, load_button],  layout = item_layout) ,

    HBox([tracking_path_textbox, load_behavior_button], layout = item_layout) ,

    HBox([track_ok, raw_ok], layout = item_layout) ,

    HBox([play, slider], layout = item_layout) ,

    HBox([sub10_button,sub5_button,add5_button,add10_button]) ,

    HBox([frame_textbox,jump_frame_button], layout = item_layout) ,

    HBox([time_textbox,jump_time_button] , layout = item_layout) ,

    HBox([check_raw,check_skel]),
    HBox([check_ellip,check_trace])

        ])

output = widgets.Output()
with output:
    storage.make_3d_axis()
    storage.fig.canvas.toolbar_position = 'bottom'


# In[ ]:





# In[16]:


from ipywidgets import AppLayout
from ipywidgets import HTML, Layout, Dropdown, Output, Textarea, VBox, Label, Text
from ipywidgets import Label, Layout, HBox
from IPython.display import display


# header = HTML("<h1><center><\"(__)~~.. MousePlayer  <\"(__)~~....</center></h1>")
# header = HTML("<h1><center><\"(__)~~.. ʍօʊֆɛ քʟǟʏɛʀ  <\"(__)~~....</center></h1>")
header = HTML("<h1><center>🐭 ʍօʊֆɛ քʟǟʏɛʀ 🐭</center></h1>")


# board = VBox( [header, HBox([output,dashboard]) ], layout=Layout(justify_content = 'center') )

board = AppLayout(header=None,
          left_sidebar=None,
          center=output,
          right_sidebar=dashboard,
          footer=None,
                 pane_widths=[0,2, 2])

app = VBox( [header, board ], layout=Layout(justify_content = 'center') )


# In[ ]:





# In[17]:




# In[ ]:





# In[ ]:





# In[18]:


# TODO toggles to show trace, ellipsoids, skeleton, raw data,
# Labeles showing if data is loaded or tracking is loaded
# Tracking without the raw data (get the xy limits from the xy data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
