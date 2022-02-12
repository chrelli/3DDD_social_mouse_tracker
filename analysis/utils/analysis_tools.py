# IDEA: Add neck to the posture map?
from IPython import get_ipython

# QT for movable plots
# %load_ext autoreload
# %autoreload 2

import time, os, sys, shutil
# from utils.fitting_utils import *

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline

# %matplotlib widget
# %matplotlib qt

from itertools import compress # for list selection with logical
from tqdm import tqdm

from multiprocessing import Process

# ALLSO JIT STUFF
from numba import jit, njit

# and pytorch
import torch

import sys, os, pickle
# import cv2
from colour import Color
import h5py
from tqdm import tqdm, tqdm_notebook
import glob
import itertools


# setup for pretty plotting

import matplotlib

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('text', usetex='false') 
matplotlib.rcParams.update({'font.size': 13})

# SET UP MATPLOTLIB
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('text', usetex=False) 
matplotlib.rcParams.update({'font.size': 13})

# AHHH, Ok: https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Liberation Sans'
matplotlib.rcParams['mathtext.it'] = 'Liberation Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Liberation Sans:bold'
matplotlib.rcParams['mathtext.fallback_to_cm'] = False
from palettable.cmocean.sequential import Algae_6
cmpl = Algae_6.mpl_colors

from palettable.cmocean.sequential import Algae_6
cmpl = Algae_6.mpl_colors

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def adjust_spines2(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
#             spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])        
        
# %debug
# MAKE a function that takes a Frame of raw data and plots a particle
# Maybe make this an object, even though I hate those...

from utils.cuda_tracking_utils_weights_for_figures import *
# from utils.cuda_tracking_utils import unpack_from_jagged, cheap4d
import matplotlib.animation as animation

# def particles_to_body_supports_cuda(part,implant = False):
#     if implant:
#         beta = part[:,0]
#         gamma = part[:,1]
#         s = part[:,2]
#         #todo naming here is off
#         psi = part[:,3]
#         theta = part[:,4]
#         phi = part[:,5]
#         t_body = part[:,6:9]
#     else:
#         beta = part[:,0]
#         gamma = part[:,1]
#         s = part[:,2]
#         #todo naming here is off
#         theta = part[:,3]
#         phi = part[:,4]
#         t_body = part[:,5:8]


#     # calculate vectors holding the hip values!
#     # the values are n_particles long
#     a_hip = a_hip_0 + a_hip_delta * s
#     b_hip = b_hip_0 + b_hip_delta * (1.-s)
#     d_hip = .75 * a_hip

#     # we need to do cos an sin on the angles!
#     # and other places too over and over, let's just keep them in mem?
#     one  = torch.ones_like(s)
#     zero = torch.zeros_like(s)

#     # this is the rotation matrix of the body - it does not need
#     R_body = make_xyz_rotation(zero,beta,gamma,one,zero)

#     ### COORDS IN MOUSE BODY FRAME ###
#     # c_hip is zero
#     # c_mid is the hinge point
#     c_mid = torch.stack([d_hip,zero,zero],dim=1)

#     # the nose-pointing vector, make more
#     cos_theta = torch.cos(theta)
#     sin_theta = torch.sin(theta)
#     cos_phi = torch.cos(phi)
#     sin_phi = torch.sin(phi)
#     # a unit vector pointing to the nose
#     nose_pointer = torch.stack([cos_theta,   sin_theta*cos_phi,     sin_theta*sin_phi], dim=1)
#     # a unit vector along the x-axis
#     x_pointer = torch.stack([one, zero,zero], dim=1)
#     # use the nose-pointing vector to calculate the nose rotation matrix
#     R_head = rotation_matrix_vec2vec(x_pointer,nose_pointer)
#     c_nose = c_mid +  torch.einsum('aij,aj->ai',[R_head, d_nose * x_pointer ])

#     # for the implant, we  allow rotation about x also (maybe limit this to realistic implant-up-scenarios?)
#     if implant:
#         cos_psi = torch.cos(psi)
#         sin_psi = torch.sin(psi)
#         c_impl = c_mid +  torch.einsum('aij,aj->ai',[R_head, torch.stack([ x_impl*one, sin_psi*z_impl,cos_psi*z_impl ],dim=1)  ])

#         c_impl = torch.einsum('aij,aj->ai',[R_body,c_impl]) + t_body
#         c_impl = torch.unsqueeze(c_impl,1).transpose(-2,-1)

#     else:
#         c_impl = c_mid *2.

#     # and these are the rest of the anchor points
#     c_ass = torch.stack([-a_hip,zero,zero],dim=1)
#     c_tip = c_mid +  torch.einsum('aij,aj->ai',[R_head, (d_nose+a_nose) * x_pointer ])
#     # c_hip = torch.stack([zero,   zero,      zero], dim=1)

#     ### CONVERT FROM BODY FRAME TO WORLD FRAME ###
#     # todo maybe pack these and batch in some way?
#     c_hip = t_body
#     c_nose = torch.einsum('aij,aj->ai',[R_body,c_nose]) + t_body

#     c_ass = torch.einsum('aij,aj->ai',[R_body,c_ass]) + t_body
#     c_tip = torch.einsum('aij,aj->ai',[R_body,c_tip]) + t_body
#     c_mid = torch.einsum('aij,aj->ai',[R_body,c_mid]) + t_body

#     # unsqueeze for auto broadcasting
#     c_hip = torch.unsqueeze(c_hip,1).transpose(-2,-1)
#     c_nose = torch.unsqueeze(c_nose,1).transpose(-2,-1)

#     c_ass = torch.unsqueeze(c_ass,1).transpose(-2,-1)
#     c_tip = torch.unsqueeze(c_tip,1).transpose(-2,-1)
#     c_mid = torch.unsqueeze(c_mid,1).transpose(-2,-1)


#     # Make the matrices for the ellipsoids, nose is always the same
#     R_nose = torch.einsum('aij,ajk->aik',[R_body,R_head])

#     body_support = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
    
#     return body_support



def particles_to_body_supports_cuda(part,implant = False):
    if implant:
        beta = part[:,0]
        gamma = part[:,1]
        s = part[:,2]
        #todo naming here is off
        psi = part[:,3]
        theta = part[:,4]
        phi = part[:,5]
        t_body = part[:,6:9]
    else:
        beta = part[:,0]
        gamma = part[:,1]
        s = part[:,2]
        #todo naming here is off
        theta = part[:,3]
        phi = part[:,4]
        t_body = part[:,5:8]


    # calculate vectors holding the hip values!
    # the values are n_particles long
    a_hip = a_hip_0 + a_hip_delta * s
    b_hip = b_hip_0 + b_hip_delta * (1.-s)
    d_hip = .75 * a_hip

    # we need to do cos an sin on the angles!
    # and other places too over and over, let's just keep them in mem?
    one  = torch.ones_like(s)
    zero = torch.zeros_like(s)

    # this is the rotation matrix of the body - it does not need
    R_body = make_xyz_rotation(zero,beta,gamma,one,zero)

    ### COORDS IN MOUSE BODY FRAME ###
    # c_hip is zero
    # c_mid is the hinge point
    c_mid = torch.stack([d_hip,zero,zero],dim=1)
    
    # OLD style of unpacking
#     # the nose-pointing vector, make more
#     cos_theta = torch.cos(theta)
#     sin_theta = torch.sin(theta)
#     cos_phi = torch.cos(phi)
#     sin_phi = torch.sin(phi)
#     # a unit vector pointing to the nose
#     nose_pointer = torch.stack([cos_theta,   sin_theta*cos_phi,     sin_theta*sin_phi], dim=1)
#     # a unit vector along the x-axis
#     x_pointer = torch.stack([one, zero,zero], dim=1)
#     # use the nose-pointing vector to calculate the nose rotation matrix
#     R_head = rotation_matrix_vec2vec(x_pointer,nose_pointer)
#     c_nose = c_mid +  torch.einsum('aij,aj->ai',[R_head, d_nose * x_pointer ])
    # New style of unpacking the head!
    R_head = make_xyz_rotation(zero,theta,phi,one,zero)
    x_pointer = torch.stack([one, zero,zero], dim=1)
    c_nose = c_mid +  torch.einsum('aij,aj->ai',[R_head, d_nose * x_pointer ])


    # for the implant, we  allow rotation about x also (maybe limit this to realistic implant-up-scenarios?)
    if implant:
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        c_impl = c_mid +  torch.einsum('aij,aj->ai',[R_head, torch.stack([ x_impl*one, sin_psi*z_impl,cos_psi*z_impl ],dim=1)  ])

        c_impl = torch.einsum('aij,aj->ai',[R_body,c_impl]) + t_body
        c_impl = torch.unsqueeze(c_impl,1).transpose(-2,-1)

    else:
        c_impl = c_mid *2.

    # and these are the rest of the anchor points
    c_ass = torch.stack([-a_hip,zero,zero],dim=1)
    c_tip = c_mid +  torch.einsum('aij,aj->ai',[R_head, (d_nose+a_nose) * x_pointer ])
    # c_hip = torch.stack([zero,   zero,      zero], dim=1)

    ### CONVERT FROM BODY FRAME TO WORLD FRAME ###
    # todo maybe pack these and batch in some way?
    c_hip = t_body
    c_nose = torch.einsum('aij,aj->ai',[R_body,c_nose]) + t_body

    c_ass = torch.einsum('aij,aj->ai',[R_body,c_ass]) + t_body
    c_tip = torch.einsum('aij,aj->ai',[R_body,c_tip]) + t_body
    c_mid = torch.einsum('aij,aj->ai',[R_body,c_mid]) + t_body

    # unsqueeze for auto broadcasting
    c_hip = torch.unsqueeze(c_hip,1).transpose(-2,-1)
    c_nose = torch.unsqueeze(c_nose,1).transpose(-2,-1)

    c_ass = torch.unsqueeze(c_ass,1).transpose(-2,-1)
    c_tip = torch.unsqueeze(c_tip,1).transpose(-2,-1)
    c_mid = torch.unsqueeze(c_mid,1).transpose(-2,-1)


    # Make the matrices for the ellipsoids, nose is always the same
    R_nose = torch.einsum('aij,ajk->aik',[R_body,R_head])

    body_support = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
    
    return body_support




class PlotMachine(object):
    def __init__(self,tracked_behavior,jagged_lines,what_to_plot = 'guess'):
        self.tracked_behavior = tracked_behavior
        self.jagged_lines = jagged_lines
        # useful functions, get the raw data in plottable format
        # pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[start_frame])
        # useful functions, get the fitted data in plottable format
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Say, "the default sans-serif font is COMIC SANS"
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        # Then, "ALWAYS use sans-serif fonts"
        matplotlib.rcParams['font.family'] = "sans-serif"

        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('text', usetex='false') 
        matplotlib.rcParams.update({'font.size': 13})

        from palettable.cmocean.sequential import Algae_6
        cmpl = Algae_6.mpl_colors
        
        # unpack 
        self.guessing_holder = tracked_behavior['guessing_holder']
        self.tracking_holder = tracked_behavior['tracking_holder']
        self.start_frame = tracked_behavior['start_frame']
        
        self.track_or_guess = what_to_plot
        self.n_frames = self.tracking_holder.shape[1]
        
        self.v_ed = None
        self.v_ed_reject = None
        
    def kernel_smoothing(self):
        def easy_kernel(kernel_width = 30):
            from scipy import stats
            kernel = stats.norm.pdf(np.arange(-3*kernel_width,3*kernel_width+1),scale=kernel_width)
            kernel = kernel/np.sum(kernel)
            return kernel
        kernel = easy_kernel(3)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')
    
    def make_3d_axis(self):
        #   3D plot of the 
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        # add to self for use later
        self.fig = fig
        self.ax = ax
        
    def add_raw_data(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]

        # add to axis 3D plot of Sphere
        self.h_pc = self.ax.scatter(X, Y, Z, zdir='z', s=10, c='k', alpha = .05,rasterized=False)
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        # loop over the types of body, and make emptyscatter plots
        self.h_kp_list = []        
        for body in body_indices:
            h_kp = self.ax.scatter([],[],[], zdir='z', s=100, c=body_colors[body],rasterized=False)
            self.h_kp_list.append(h_kp)
        
        # THEN set the 3d values to be what the shoud be
        for body in body_indices:
            self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])
        
        
        
#         self.h_kp_list = []
#         for i,body in enumerate(ikeyp):
#             h_kp = self.ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[body],rasterized=False)
#             self.h_kp_list.append(h_kp)
        # for axis adjustment
        self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        self.mid_x = (X.max()+X.min()) * 0.5
        self.mid_y = (Y.max()+Y.min()) * 0.5
        self.mid_z = (Z.max()+Z.min()) * 0.5
            
    def plot_skeleton(self,body_support,color = 'k',body_idx = 0):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
       
        # add the body points
        for p in [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]:
            if (len(p.shape) >= 3):
                h_bp = self.ax.scatter(p[:,0,0],p[:,1,0],p[:,2,0],zdir='z', s=50, alpha = 1 , c=color,rasterized=False)
                self.h_bp_list[body_idx].append(h_bp)
                
        # and the lines between body parts
        for p,q in zip([c_nose,c_nose,c_mid,c_impl,c_impl],[c_mid,c_tip,c_ass,c_nose,c_tip]):
            if (len(p.shape) >= 3) and (len(q.shape) >= 3):
                for ii in range(p.shape[0]):
                    h_skel = self.ax.plot([p[ii,0,0],q[ii,0,0]],[p[ii,1,0],q[ii,1,0]],[p[ii,2,0],q[ii,2,0]],c=color,lw = 4)
                    self.h_skel_list[body_idx].append(h_skel)
                    
    def add_wireframe_to_axis(self,ax,R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip',this_color='k',this_alpha=.4):
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
        posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip

        x = posi_rotated[0,:]
        y = posi_rotated[1,:]
        z = posi_rotated[2,:]

        # reshape for wireframe
        x = np.reshape(x, (u.shape) )
        y = np.reshape(y, (u.shape) )
        z = np.reshape(z, (u.shape) )

        h_hip = ax.plot_wireframe(x, y, z, color=this_color,alpha = this_alpha)
        return h_hip        
                    
    def plot_ellipsoids(self,part,body_support,color = 'k',body_idx = 0):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
        # this is not so elegant, hm hm
        s = part[:,2]
        a_hip = a_hip_0 + a_hip_delta * s
        b_hip = b_hip_0 + b_hip_delta * (1.-s)
        d_hip = .75 * a_hip
        
        for RR,cc,style in zip([R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']):
            if (len(cc.shape) >= 3):
                h_hip = self.add_wireframe_to_axis(self.ax,RR[0,...],
                                                   cc[0,...],
                                                   a_nose.cpu().numpy(),
                                                   b_nose.cpu().numpy(),
                                                   a_hip.cpu().numpy(),
                                                   b_hip.cpu().numpy(),
                                                   r_impl.cpu().numpy(),
                                                   style=style,this_color=color)
                self.h_hip_list[body_idx].append(h_hip)

    def add_fit(self,frame,plot_ellipsoids = True):
        # get the particle, convert to torch tensor, calculate body supports
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

        # store
        self.body_support_0 = body_support_0
        self.body_support_1 = body_support_1
        
        # and plot!
        self.h_skel_list = [[],[]]
        self.h_bp_list = [[],[]]
        
        self.plot_skeleton(body_support_0,color = 'k',body_idx = 0)
        self.plot_skeleton(body_support_1,color = 'peru',body_idx = 1)                

        self.h_hip_list = [[],[]]
        if plot_ellipsoids:
            self.plot_ellipsoids(part,body_support_0,color = 'k',body_idx = 0)
            self.plot_ellipsoids(part,body_support_1,color = 'peru',body_idx = 1)
            
    def add_trace(self,frame,trace_length=90,trace_clip = None,decay_factor=.9, type_list = ['tip']):
        # get the particle, convert to torch tensor, calculate body supports
        print(frame)
        i_frame = frame-self.start_frame
        print(i_frame)
        # make a holder for the lines
        self.h_trace_list = [[None]*5,[None]*5]
     
        if trace_clip is not None:
            i_clip = trace_clip-self.start_frame
            i_trace_start = np.max([i_clip, i_frame-trace_length])
        else:
            i_trace_start = np.max([0, i_frame-trace_length])
            
        print("i_trace_start is {} and i_frame is {}".format(i_trace_start,i_frame))
        
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,(i_trace_start):(i_frame)]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,(i_trace_start):(i_frame)]
        
        if part.shape[1] == 0:
            # just skip if there is no trace
            print(part.shape)
            return
        
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        
#         type_list = ['ass','hip','nose','tip']
#         self.unpack_trace(body_support_0,what_type=type_list,color='aqua')     
#         self.unpack_trace(body_support_1,what_type=type_list,color='fuchsia')     
        
        self.unpack_trace(body_support_0,body_idx = 0,what_type=type_list,color='black')     
        self.unpack_trace(body_support_1,body_idx = 1,what_type=type_list,color='peru')
        
    def finish_3d_axis(self,view_style = 'top', zoom = False, dump = False):
        # finish the labeling, plot adjustments, dump and show
        ax = self.ax
#         ax.set_xlabel('$x$ (mm)')
#         ax.set_ylabel('\n$y$ (mm)')
#         zlabel = ax.set_zlabel('\n$z$ (mm)')

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
            
        if zoom:
            def mean_point(body_support,impl = False):
                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
                if impl==True:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip,c_impl])
                else:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip])
                mean_0 = np.mean(points_0,0)
                return mean_0
            mean_0 = mean_point(self.body_support_0,impl= True)
            mean_1 = mean_point(self.body_support_1, impl = False)
            
            overall_mean = np.hstack([mean_0,mean_1])
            
            mu_zoom = np.mean(overall_mean,1)
            scaling = 2
            d_zoom = ( mean_0.ravel() - mean_1.ravel() )*scaling
#             print(d_zoom)
#             print(mu_zoom)
#             self.ax.scatter(mu_zoom[0],mu_zoom[1],mu_zoom[2],zdir='z', s=500, alpha = 1 , c='pink',rasterized=False)
            
            pp = np.vstack([mu_zoom-d_zoom,mu_zoom,mu_zoom+d_zoom])
#             print(pp)
#             self.ax.plot(pp[:,0],pp[:,1],pp[:,2],zdir='z',alpha = 1,lw=10 , c='pink',rasterized=False)
            
            X,Y,Z = pp[:,0],pp[:,1],pp[:,2]
            
            # for axis adjustment
            self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            self.mid_x = (X.max()+X.min()) * 0.5
            self.mid_y = (Y.max()+Y.min()) * 0.5
            self.mid_z = (Z.max()+Z.min()) * 0.5
            
            ax.set_xlim(self.mid_x - self.max_range, self.mid_x + self.max_range)
            ax.set_ylim(self.mid_y - self.max_range, self.mid_y + self.max_range)
            ax.set_zlim(0, 2*self.max_range)
            
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            
        ax.view_init(elev=el, azim=az)

    def unpack_trace(self,body_support,body_idx = 0,what_type=['hip'],color='k'):
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
        type_list = np.array(['hip','ass','mid','nose','tip','impl'])
        c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
        ii_c_list = np.arange(len(type_list))
        # TODO make the decay work!
        
        for ttt in what_type:
            # this is also not so elegant
            selecta = np.arange(len(type_list))[type_list == ttt]
            dat = c_list[selecta[0]].squeeze()
            X,Y,Z = dat[:,0],dat[:,1],dat[:,2]
            h_trace = self.ax.plot(X,Y,Z,lw=2,c=color,alpha = .65)
            self.h_trace_list[body_idx][ii_c_list[type_list == ttt][0]] = h_trace
            print(what_type)        

    def dump_to_disk(self,tag = ""):
        print("saving w/tag: "+tag)
        plt.tight_layout()
        plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/Plotter'+tag+'.pdf',transparent=True)  

    def add_frame_number(self,frame):
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.74, 0.07, "Frame:" ,ha='left',fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_number = self.ax.text2D(0.95, 0.07, str(frame),ha='right', fontsize = 20, transform=self.ax.transAxes)
        pass
    
    def add_frame_time(self,frame,start_frame):
        cam_fps = 60 #Hz
        t_now = (frame-start_frame)/cam_fps
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.71, 0.03, "Time:" ,ha='left', fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_time = self.ax.text2D(0.95, 0.03, "{:0.2f} s".format(t_now) ,ha='right',fontsize = 20, transform=self.ax.transAxes)
        pass    

    def add_ed(self,frame,case='ed'):
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
        c_nose = np.squeeze(c_mid)
        if case == 'ed':
            v_ed = self.v_ed[frame-self.start_frame,:]
            self.h_ed = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c='r',lw = 4)
        if case == 'ed_reject':
            v_ed = self.v_ed_reject[frame-self.start_frame,:]
            ccc = 'blueviolet'
            self.h_ed_reject = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c=ccc,lw = 4)

        pass
    
    def make(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
            self.add_raw_data(frame)
            self.track_or_guess = 'track' #hk
            self.add_fit(frame)
            self.track_or_guess = 'track'
            self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
            
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
            
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()

    def make_skel1(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
            self.add_raw_data(frame)
            self.track_or_guess = 'track'
            self.add_fit(frame,plot_ellipsoids = True)
            self.track_or_guess = 'track'
#             self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
        
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()        


    def make_skel2(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
            self.add_raw_data(frame)
            self.track_or_guess = 'track'
            self.add_fit(frame,plot_ellipsoids = False)
            self.track_or_guess = 'track'
#             self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
        
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)            
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()        
        
        
        
    def make_skel3(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
#             self.add_raw_data(frame)
            self.track_or_guess = 'track'
            self.add_fit(frame,plot_ellipsoids = False)
            self.track_or_guess = 'track'
#             self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
        
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()          
        
        
    def video(self,frame_list=30*60+np.arange(100)*100, savepath = None,view_override = None):
        
        # MAKE STARTING PLOT
        
        self.trace_length = 960
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        self.track_or_guess = 'track'
        self.add_fit(frame_list[0])
#         self.track_or_guess = 'guess'
        self.add_trace(frame_list[0],trace_length=self.trace_length,trace_clip = self.trace_clip ,type_list = self.type_list)
        self.track_or_guess = 'track'
        
        if self.v_ed is not None:
            self.add_ed(frame_list[0],case='ed')
            self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])

        
        # LOTS OF FUNCTIONS FOR ANIMATION
        
        def update_frame_number(frame):
#             self.h_frame_number.set_text("Frame: {}".format(frame))
            self.h_frame_number.set_text(str(frame))

        def update_frame_time(frame,start_frame = frame_list[0]):
            cam_fps = 60. #Hz
            t_now = (frame-start_frame)/cam_fps
            self.h_frame_time.set_text("{:0.2f} s".format(t_now))
            
        def update_skeleton(body_support,body_idx):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]            
            # add the body points
            for j,p in enumerate([c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]):
                if (len(p.shape) >= 3):
                    # update the location of the body point!
                    self.h_bp_list[body_idx][j]._offsets3d = (p[:,0,0],p[:,1,0],p[:,2,0])

            # and the lines between body parts
            for j, (p,q) in enumerate( zip([c_nose,c_nose,c_mid,c_impl,c_impl],[c_mid,c_tip,c_ass,c_nose,c_tip]) ):
                if (len(p.shape) >= 3) and (len(q.shape) >= 3):
                    for ii in range(p.shape[0]):
                        # lines are an extra level deep for some stupid matplotlib reason
                        self.h_skel_list[body_idx][j][0].set_xdata([p[ii,0,0],q[ii,0,0]])
                        self.h_skel_list[body_idx][j][0].set_ydata([p[ii,1,0],q[ii,1,0]])
                        self.h_skel_list[body_idx][j][0].set_3d_properties([p[ii,2,0],q[ii,2,0]])

        def update_wireframe_lines(h_hip,X,Y,Z):
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
        
        def calculate_wireframe_points(R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip'):
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
            posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip

            x = posi_rotated[0,:]
            y = posi_rotated[1,:]
            z = posi_rotated[2,:]

            # reshape for wireframe
            x = np.reshape(x, (u.shape) )
            y = np.reshape(y, (u.shape) )
            z = np.reshape(z, (u.shape) )
            
            return x,y,z
        
        def update_ellipsoids(part,body_support,body_idx):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
            # this is not so elegant, hm hm
            s = part[:,2]
            a_hip = a_hip_0 + a_hip_delta * s
            b_hip = b_hip_0 + b_hip_delta * (1.-s)
            d_hip = .75 * a_hip

            for jj,(RR,cc,style) in enumerate( zip([R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']) ):
                if (len(cc.shape) >= 3):
                    X,Y,Z = calculate_wireframe_points(RR[0,...],
                                                   cc[0,...],
                                                   a_nose.cpu().numpy(),
                                                   b_nose.cpu().numpy(),
                                                   a_hip.cpu().numpy(),
                                                   b_hip.cpu().numpy(),
                                                   r_impl.cpu().numpy(),
                                                      style)
                    h_hip = self.h_hip_list[body_idx][jj]
                    update_wireframe_lines(h_hip,X,Y,Z) 

            
        def update_trace_3dlines(body_support,body_idx=0,what_type=['hip']):
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
            type_list = np.array(['hip','ass','mid','nose','tip','impl'])
            c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
            ii_c_list = np.arange(len(type_list))
            # TODO make the decay work!

            for ttt in what_type:
                # this is also not so elegant
                selecta = np.arange(len(type_list))[type_list == ttt]
                dat = c_list[selecta[0]].squeeze()
                X,Y,Z = dat[:,0],dat[:,1],dat[:,2]
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_xdata(X)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_ydata(Y)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_3d_properties(Z)
                
        
        def calculate_new_trace(frame,trace_length=None,trace_clip = None,decay_factor=.9, type_list = None):
            # get the particle, convert to torch tensor, calculate body supports
            i_frame = frame-self.start_frame

            if trace_clip is not None:
                i_clip = trace_clip-self.start_frame
                i_trace_start = np.max([i_clip, i_frame-trace_length])
            else:
                i_trace_start = np.max([0, i_frame-trace_length])

            if self.track_or_guess == 'track':
                part = self.tracking_holder[:-1,(i_trace_start):(i_frame)]
            if self.track_or_guess == 'guess':
                part = self.guessing_holder[:-1,(i_trace_start):(i_frame)]
                
            # overwrite HACK for now
            part = self.guessing_holder[:-1,(i_trace_start):(i_frame)]

            if part.shape[1] == 0:
                # just skip if there is no trace
                return

            part = torch.from_numpy(part).float().cuda()
            part = torch.transpose(part,0,1)
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
            update_trace_3dlines(body_support_0,body_idx=0,what_type = type_list)
            update_trace_3dlines(body_support_1,body_idx=1,what_type = type_list)


        if self.v_ed is not None:
            def update_ed(frame,case='ed'):
                if self.track_or_guess == 'track':
                    part = self.tracking_holder[:-1,frame-self.start_frame]
                if self.track_or_guess == 'guess':
                    part = self.guessing_holder[:-1,frame-self.start_frame]

                part = torch.from_numpy(part).float().unsqueeze(0).cuda()
                body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
                body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
                c_nose = np.squeeze(c_mid)
                if case == 'ed':
                    v_ed = self.v_ed[frame-self.start_frame,:]
                    self.h_ed[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])
                if case == 'ed_reject':
                    v_ed = self.v_ed_reject[frame-self.start_frame,:]
                    self.h_ed_reject[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed_reject[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed_reject[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])

                pass            
        
        def update_pc(frame):
            # get new raw data!
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            # update 
            self.h_pc._offsets3d = (X,Y,Z)
            
            # THEN set the 3d values to be what the shoud be
            for body in range(4):
                self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])

            # update the fit as well!
            # get the particle, convert to torch tensor, calculate body supports
            self.track_or_guess = 'track'

            if self.track_or_guess == 'track':
                part = self.tracking_holder[:-1,frame-self.start_frame]
            if self.track_or_guess == 'guess':
                part = self.guessing_holder[:-1,frame-self.start_frame]

            part = torch.from_numpy(part).float().unsqueeze(0).cuda()
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
            update_skeleton(body_support_0,body_idx=0)
            update_skeleton(body_support_1,body_idx=1)
            update_ellipsoids(part,body_support_0,body_idx=0)
            update_ellipsoids(part,body_support_1,body_idx=1)
            
            calculate_new_trace(frame,trace_length=self.trace_length,trace_clip = self.trace_clip,decay_factor=.9, type_list = self.type_list)
            
            update_frame_number(frame)
            update_frame_time(frame)
            if self.v_ed is not None:
                update_ed(frame,case='ed')
                update_ed(frame,case='ed_reject')
                
        # trick for updating line3d collection
        # https://mail.python.org/pipermail/matplotlib-users/2015-October/000066.html

        
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, frame_list[1:], interval=10)

        fps = 10
        if savepath is None:
            fn = '<"(, ,)~~'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        
        ani.save(savepath,writer='imagemagick',fps=fps)
        plt.rcParams['animation.html'] = 'html5'
        ani

    def rotation(self,frame_list=30*60+np.arange(100)*100, savepath = None,view_override = None):
        
        # MAKE STARTING PLOT
        
        self.trace_length = 960
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        self.track_or_guess = 'track'
        self.add_fit(frame_list[0])
#         self.track_or_guess = 'guess'
        self.add_trace(frame_list[0],trace_length=self.trace_length,trace_clip = self.trace_clip ,type_list = self.type_list)
        self.track_or_guess = 'track'
        
        if self.v_ed is not None:
            self.add_ed(frame_list[0],case='ed')
            self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])

        
        # LOTS OF FUNCTIONS FOR ANIMATION
        def update_pc(i):
            # get new raw data!
            
            offset = el*np.sin(i*2*np.pi / 360)
            plt.gca().view_init(elev=el+offset, azim=az+i)
            
                
        # trick for updating line3d collection
        # https://mail.python.org/pipermail/matplotlib-users/2015-October/000066.html

        
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, np.linspace(0,360,40), interval=10)

        fps = 10
        if savepath is None:
            fn = '<"(, ,)~~_rotation'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD_rotation'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        
        ani.save(savepath,writer='imagemagick',fps=fps)
        plt.rcParams['animation.html'] = 'html5'
        ani        
        
        
        
    def plot_residuals(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
        
        # and get the corresponding fit: the particle, convert to torch tensor, calculate body supports
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
#         body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
#         body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        pos = torch.from_numpy(pos).float().cuda()
        
        # CALCULATE RESIDUAL
        dist0,_,self.body_support_0 = particles_to_distance_cuda(part[:,:9],pos[:,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(part[:,9:],pos[:,:])
        r = torch.min(dist0[0,:],dist1[0,:])
        r = r.cpu().numpy().squeeze()
        
        # Figure out the weighing stuff
        w2 = pos_weights**2

        rw=np.clip(r,0,.03)
        
        # or weigted mean?
#         rm = np.mean(rw)
        
        # correct
        rm = np.sum(w2*rw)/np.sum(w2)
        # wrong old way:
#         rm = np.mean(w2*rw)/np.median(w2)

        plt.figure()
        plt.plot(r,'.k',label='residual')
        plt.plot(rw,'.r',label='cut')
        plt.plot(w2,'.g',label='w2')
        plt.legend()
        plt.show()
        
#         plt.figure()
#         plt.hist(w2,100)
#         plt.show()
        
        plt.figure(figsize =(10,10))
#         plt.scatter(X,Y,c = 1-rw/np.max(rw))
        plt.scatter(X,Y,c = w2/np.max(w2))

        plt.axis('square')        
        plt.title("clipped: {}, weighted: {}".format(np.mean(rw),rm))
        plt.show()


        pass
# tips https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
# add raw data to plot

# add particle

    def kernel_smoothing(self,width = 3):
        kernel = easy_kernel(width)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')



class PlotMachine_noimpl(object):
    def __init__(self,tracked_behavior,jagged_lines,what_to_plot = 'guess', has_implant = False):
        self.tracked_behavior = tracked_behavior
        self.jagged_lines = jagged_lines
        # useful functions, get the raw data in plottable format
        # pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[start_frame])
        # useful functions, get the fitted data in plottable format
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Say, "the default sans-serif font is COMIC SANS"
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        # Then, "ALWAYS use sans-serif fonts"
        matplotlib.rcParams['font.family'] = "sans-serif"

        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('text', usetex='false') 
        matplotlib.rcParams.update({'font.size': 13})

        from palettable.cmocean.sequential import Algae_6
        cmpl = Algae_6.mpl_colors
        
        # unpack 
        self.guessing_holder = tracked_behavior['guessing_holder']
        self.tracking_holder = tracked_behavior['tracking_holder']
        self.start_frame = tracked_behavior['start_frame']
        
        self.has_implant = has_implant
        self.track_or_guess = what_to_plot
        self.n_frames = self.tracking_holder.shape[1]
        
        self.v_ed = None
        self.v_ed_reject = None
        
    def kernel_smoothing(self):
        def easy_kernel(kernel_width = 30):
            from scipy import stats
            kernel = stats.norm.pdf(np.arange(-3*kernel_width,3*kernel_width+1),scale=kernel_width)
            kernel = kernel/np.sum(kernel)
            return kernel
        kernel = easy_kernel(3)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')
    
    def make_3d_axis(self):
        #   3D plot of the 
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        # add to self for use later
        self.fig = fig
        self.ax = ax
        
    def add_raw_data(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]

        # add to axis 3D plot of Sphere
        self.h_pc = self.ax.scatter(X, Y, Z, zdir='z', s=10, c='k', alpha = .05,rasterized=False)
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        # loop over the types of body, and make emptyscatter plots
        self.h_kp_list = []        
        for body in body_indices:
            h_kp = self.ax.scatter([],[],[], zdir='z', s=100, c=body_colors[body],rasterized=False)
            self.h_kp_list.append(h_kp)
        
        # THEN set the 3d values to be what the shoud be
        for body in body_indices:
            self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])
        
        
        
#         self.h_kp_list = []
#         for i,body in enumerate(ikeyp):
#             h_kp = self.ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[body],rasterized=False)
#             self.h_kp_list.append(h_kp)
        # for axis adjustment
        self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        self.mid_x = (X.max()+X.min()) * 0.5
        self.mid_y = (Y.max()+Y.min()) * 0.5
        self.mid_z = (Z.max()+Z.min()) * 0.5
            
    def plot_skeleton(self,body_support,color = 'k',body_idx = 0):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
       
        # add the body points
        for p in [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]:
            if (len(p.shape) >= 3):
                h_bp = self.ax.scatter(p[:,0,0],p[:,1,0],p[:,2,0],zdir='z', s=50, alpha = 1 , c=color,rasterized=False)
                self.h_bp_list[body_idx].append(h_bp)
                
        # and the lines between body parts
        for p,q in zip([c_nose,c_nose,c_mid,c_impl,c_impl],[c_mid,c_tip,c_ass,c_nose,c_tip]):
            if (len(p.shape) >= 3) and (len(q.shape) >= 3):
                for ii in range(p.shape[0]):
                    h_skel = self.ax.plot([p[ii,0,0],q[ii,0,0]],[p[ii,1,0],q[ii,1,0]],[p[ii,2,0],q[ii,2,0]],c=color,lw = 4)
                    self.h_skel_list[body_idx].append(h_skel)
                    
    def add_wireframe_to_axis(self,ax,R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip',this_color='k',this_alpha=.4):
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
        posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip

        x = posi_rotated[0,:]
        y = posi_rotated[1,:]
        z = posi_rotated[2,:]

        # reshape for wireframe
        x = np.reshape(x, (u.shape) )
        y = np.reshape(y, (u.shape) )
        z = np.reshape(z, (u.shape) )

        h_hip = ax.plot_wireframe(x, y, z, color=this_color,alpha = this_alpha)
        return h_hip        
                    
    def plot_ellipsoids(self,part,body_support,color = 'k',body_idx = 0):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
        # this is not so elegant, hm hm
        s = part[:,2]
        a_hip = a_hip_0 + a_hip_delta * s
        b_hip = b_hip_0 + b_hip_delta * (1.-s)
        d_hip = .75 * a_hip
        
        for RR,cc,style in zip([R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']):
            if (len(cc.shape) >= 3):
                h_hip = self.add_wireframe_to_axis(self.ax,RR[0,...],
                                                   cc[0,...],
                                                   a_nose.cpu().numpy(),
                                                   b_nose.cpu().numpy(),
                                                   a_hip.cpu().numpy(),
                                                   b_hip.cpu().numpy(),
                                                   r_impl.cpu().numpy(),
                                                   style=style,this_color=color)
                self.h_hip_list[body_idx].append(h_hip)

    def add_fit(self,frame,plot_ellipsoids = True):
        # get the particle, convert to torch tensor, calculate body supports
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
        if self.has_implant:
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

        else:
            body_support_0 = particles_to_body_supports_cuda(part[:,:8],implant = False)
            body_support_1 = particles_to_body_supports_cuda(part[:,8:],implant = False)

        # store
        self.body_support_0 = body_support_0
        self.body_support_1 = body_support_1
        
        # and plot!
        self.h_skel_list = [[],[]]
        self.h_bp_list = [[],[]]
        
        self.plot_skeleton(body_support_0,color = 'k',body_idx = 0)
        self.plot_skeleton(body_support_1,color = 'peru',body_idx = 1)                

        self.h_hip_list = [[],[]]
        if plot_ellipsoids:
            self.plot_ellipsoids(part,body_support_0,color = 'k',body_idx = 0)
            self.plot_ellipsoids(part,body_support_1,color = 'peru',body_idx = 1)
            
    def add_trace(self,frame,trace_length=90,trace_clip = None,decay_factor=.9, type_list = ['tip']):
        # get the particle, convert to torch tensor, calculate body supports
        print(frame)
        i_frame = frame-self.start_frame
        print(i_frame)
        # make a holder for the lines
        self.h_trace_list = [[None]*5,[None]*5]
     
        if trace_clip is not None:
            i_clip = trace_clip-self.start_frame
            i_trace_start = np.max([i_clip, i_frame-trace_length])
        else:
            i_trace_start = np.max([0, i_frame-trace_length])
            
        print("i_trace_start is {} and i_frame is {}".format(i_trace_start,i_frame))
        
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,(i_trace_start):(i_frame)]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,(i_trace_start):(i_frame)]
        
        if part.shape[1] == 0:
            # just skip if there is no trace
            print(part.shape)
            return
        
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)
        
        if self.has_implant:
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

        else:
            body_support_0 = particles_to_body_supports_cuda(part[:,:8],implant = False)
            body_support_1 = particles_to_body_supports_cuda(part[:,8:],implant = False)
  
        #         type_list = ['ass','hip','nose','tip']
        #         self.unpack_trace(body_support_0,what_type=type_list,color='aqua')     
        #         self.unpack_trace(body_support_1,what_type=type_list,color='fuchsia')     
        
        self.unpack_trace(body_support_0,body_idx = 0,what_type=type_list,color='black')     
        self.unpack_trace(body_support_1,body_idx = 1,what_type=type_list,color='peru')
        
    def finish_3d_axis(self,view_style = 'top', zoom = False, dump = False):
        # finish the labeling, plot adjustments, dump and show
        ax = self.ax
#         ax.set_xlabel('$x$ (mm)')
#         ax.set_ylabel('\n$y$ (mm)')
#         zlabel = ax.set_zlabel('\n$z$ (mm)')

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
            
        if zoom:
            def mean_point(body_support,impl = False):
                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
                if impl==True:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip,c_impl])
                else:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip])
                mean_0 = np.mean(points_0,0)
                return mean_0
            mean_0 = mean_point(self.body_support_0,impl= True)
            mean_1 = mean_point(self.body_support_1, impl = False)
            
            overall_mean = np.hstack([mean_0,mean_1])
            
            mu_zoom = np.mean(overall_mean,1)
            scaling = 2
            d_zoom = ( mean_0.ravel() - mean_1.ravel() )*scaling
#             print(d_zoom)
#             print(mu_zoom)
#             self.ax.scatter(mu_zoom[0],mu_zoom[1],mu_zoom[2],zdir='z', s=500, alpha = 1 , c='pink',rasterized=False)
            
            pp = np.vstack([mu_zoom-d_zoom,mu_zoom,mu_zoom+d_zoom])
#             print(pp)
#             self.ax.plot(pp[:,0],pp[:,1],pp[:,2],zdir='z',alpha = 1,lw=10 , c='pink',rasterized=False)
            
            X,Y,Z = pp[:,0],pp[:,1],pp[:,2]
            
            # for axis adjustment
            self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            self.mid_x = (X.max()+X.min()) * 0.5
            self.mid_y = (Y.max()+Y.min()) * 0.5
            self.mid_z = (Z.max()+Z.min()) * 0.5
            
            ax.set_xlim(self.mid_x - self.max_range, self.mid_x + self.max_range)
            ax.set_ylim(self.mid_y - self.max_range, self.mid_y + self.max_range)
            ax.set_zlim(0, 2*self.max_range)
            
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            
        ax.view_init(elev=el, azim=az)

    def unpack_trace(self,body_support,body_idx = 0,what_type=['hip'],color='k'):
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
        type_list = np.array(['hip','ass','mid','nose','tip','impl'])
        c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
        ii_c_list = np.arange(len(type_list))
        # TODO make the decay work!
        
        for ttt in what_type:
            # this is also not so elegant
            selecta = np.arange(len(type_list))[type_list == ttt]
            dat = c_list[selecta[0]].squeeze()
            X,Y,Z = dat[:,0],dat[:,1],dat[:,2]
            h_trace = self.ax.plot(X,Y,Z,lw=2,c=color,alpha = .65)
            self.h_trace_list[body_idx][ii_c_list[type_list == ttt][0]] = h_trace
            print(what_type)        

    def dump_to_disk(self,tag = ""):
        print("saving w/tag: "+tag)
        plt.tight_layout()
        plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/Plotter'+tag+'.pdf',transparent=True)  

    def add_frame_number(self,frame):
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.74, 0.07, "Frame:" ,ha='left',fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_number = self.ax.text2D(0.95, 0.07, str(frame),ha='right', fontsize = 20, transform=self.ax.transAxes)
        pass
    
    def add_frame_time(self,frame,start_frame):
        cam_fps = 60 #Hz
        t_now = (frame-start_frame)/cam_fps
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.71, 0.03, "Time:" ,ha='left', fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_time = self.ax.text2D(0.95, 0.03, "{:0.2f} s".format(t_now) ,ha='right',fontsize = 20, transform=self.ax.transAxes)
        pass    

    def add_ed(self,frame,case='ed'):
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
        c_nose = np.squeeze(c_mid)
        if case == 'ed':
            v_ed = self.v_ed[frame-self.start_frame,:]
            self.h_ed = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c='r',lw = 4)
        if case == 'ed_reject':
            v_ed = self.v_ed_reject[frame-self.start_frame,:]
            ccc = 'blueviolet'
            self.h_ed_reject = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c=ccc,lw = 4)

        pass
    
    def make(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
            self.add_raw_data(frame)
            self.track_or_guess = 'track' #hk
            self.add_fit(frame)
            self.track_or_guess = 'track'
            self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
            
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
            
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()

    def make_skel1(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
            self.add_raw_data(frame)
            self.track_or_guess = 'track'
            self.add_fit(frame,plot_ellipsoids = True)
            self.track_or_guess = 'track'
#             self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
        
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()        


    def make_skel2(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
            self.add_raw_data(frame)
            self.track_or_guess = 'track'
            self.add_fit(frame,plot_ellipsoids = False)
            self.track_or_guess = 'track'
#             self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
        
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)            
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()        
        
        
        
    def make_skel3(self,frame=None,tag = "",savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        self.make_3d_axis()
        if frame is not None:
#             self.add_raw_data(frame)
            self.track_or_guess = 'track'
            self.add_fit(frame,plot_ellipsoids = False)
            self.track_or_guess = 'track'
#             self.add_trace(frame,trace_length=960,trace_clip = 30*60-960 ,type_list = ['nose'])
            if self.v_ed is not None:
                self.add_ed(frame,case='ed')
                self.add_ed(frame,case='ed_reject')
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)
        if len(tag) > 0:
            self.dump_to_disk(tag = tag)
        
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()          
        
        
    def video(self,frame_list=30*60+np.arange(100)*100, savepath = None,view_override = None):
        
        # MAKE STARTING PLOT
        
        self.trace_length = 960
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        self.track_or_guess = 'track'
        self.add_fit(frame_list[0])
#         self.track_or_guess = 'guess'
        self.add_trace(frame_list[0],trace_length=self.trace_length,trace_clip = self.trace_clip ,type_list = self.type_list)
        self.track_or_guess = 'track'
        
        if self.v_ed is not None:
            self.add_ed(frame_list[0],case='ed')
            self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])

        
        # LOTS OF FUNCTIONS FOR ANIMATION
        
        def update_frame_number(frame):
#             self.h_frame_number.set_text("Frame: {}".format(frame))
            self.h_frame_number.set_text(str(frame))

        def update_frame_time(frame,start_frame = frame_list[0]):
            cam_fps = 60. #Hz
            t_now = (frame-start_frame)/cam_fps
            self.h_frame_time.set_text("{:0.2f} s".format(t_now))
            
        def update_skeleton(body_support,body_idx):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]            
            # add the body points
            for j,p in enumerate([c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]):
                if (len(p.shape) >= 3):
                    # update the location of the body point!
                    self.h_bp_list[body_idx][j]._offsets3d = (p[:,0,0],p[:,1,0],p[:,2,0])

            # and the lines between body parts
            for j, (p,q) in enumerate( zip([c_nose,c_nose,c_mid,c_impl,c_impl],[c_mid,c_tip,c_ass,c_nose,c_tip]) ):
                if (len(p.shape) >= 3) and (len(q.shape) >= 3):
                    for ii in range(p.shape[0]):
                        # lines are an extra level deep for some stupid matplotlib reason
                        self.h_skel_list[body_idx][j][0].set_xdata([p[ii,0,0],q[ii,0,0]])
                        self.h_skel_list[body_idx][j][0].set_ydata([p[ii,1,0],q[ii,1,0]])
                        self.h_skel_list[body_idx][j][0].set_3d_properties([p[ii,2,0],q[ii,2,0]])

        def update_wireframe_lines(h_hip,X,Y,Z):
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
        
        def calculate_wireframe_points(R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip'):
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
            posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip

            x = posi_rotated[0,:]
            y = posi_rotated[1,:]
            z = posi_rotated[2,:]

            # reshape for wireframe
            x = np.reshape(x, (u.shape) )
            y = np.reshape(y, (u.shape) )
            z = np.reshape(z, (u.shape) )
            
            return x,y,z
        
        def update_ellipsoids(part,body_support,body_idx):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
            # this is not so elegant, hm hm
            s = part[:,2]
            a_hip = a_hip_0 + a_hip_delta * s
            b_hip = b_hip_0 + b_hip_delta * (1.-s)
            d_hip = .75 * a_hip

            for jj,(RR,cc,style) in enumerate( zip([R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']) ):
                if (len(cc.shape) >= 3):
                    X,Y,Z = calculate_wireframe_points(RR[0,...],
                                                   cc[0,...],
                                                   a_nose.cpu().numpy(),
                                                   b_nose.cpu().numpy(),
                                                   a_hip.cpu().numpy(),
                                                   b_hip.cpu().numpy(),
                                                   r_impl.cpu().numpy(),
                                                      style)
                    h_hip = self.h_hip_list[body_idx][jj]
                    update_wireframe_lines(h_hip,X,Y,Z) 

            
        def update_trace_3dlines(body_support,body_idx=0,what_type=['hip']):
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
            type_list = np.array(['hip','ass','mid','nose','tip','impl'])
            c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
            ii_c_list = np.arange(len(type_list))
            # TODO make the decay work!

            for ttt in what_type:
                # this is also not so elegant
                selecta = np.arange(len(type_list))[type_list == ttt]
                dat = c_list[selecta[0]].squeeze()
                X,Y,Z = dat[:,0],dat[:,1],dat[:,2]
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_xdata(X)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_ydata(Y)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_3d_properties(Z)
                
        
        def calculate_new_trace(frame,trace_length=None,trace_clip = None,decay_factor=.9, type_list = None):
            # get the particle, convert to torch tensor, calculate body supports
            i_frame = frame-self.start_frame

            if trace_clip is not None:
                i_clip = trace_clip-self.start_frame
                i_trace_start = np.max([i_clip, i_frame-trace_length])
            else:
                i_trace_start = np.max([0, i_frame-trace_length])

            if self.track_or_guess == 'track':
                part = self.tracking_holder[:-1,(i_trace_start):(i_frame)]
            if self.track_or_guess == 'guess':
                part = self.guessing_holder[:-1,(i_trace_start):(i_frame)]
                
            # overwrite HACK for now
            part = self.guessing_holder[:-1,(i_trace_start):(i_frame)]

            if part.shape[1] == 0:
                # just skip if there is no trace
                return

            part = torch.from_numpy(part).float().cuda()
            part = torch.transpose(part,0,1)
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
            update_trace_3dlines(body_support_0,body_idx=0,what_type = type_list)
            update_trace_3dlines(body_support_1,body_idx=1,what_type = type_list)


        if self.v_ed is not None:
            def update_ed(frame,case='ed'):
                if self.track_or_guess == 'track':
                    part = self.tracking_holder[:-1,frame-self.start_frame]
                if self.track_or_guess == 'guess':
                    part = self.guessing_holder[:-1,frame-self.start_frame]

                part = torch.from_numpy(part).float().unsqueeze(0).cuda()
                body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
                body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
                c_nose = np.squeeze(c_mid)
                if case == 'ed':
                    v_ed = self.v_ed[frame-self.start_frame,:]
                    self.h_ed[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])
                if case == 'ed_reject':
                    v_ed = self.v_ed_reject[frame-self.start_frame,:]
                    self.h_ed_reject[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed_reject[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed_reject[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])

                pass            
        
        def update_pc(frame):
            # get new raw data!
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            # update 
            self.h_pc._offsets3d = (X,Y,Z)
            
            # THEN set the 3d values to be what the shoud be
            for body in range(4):
                self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])

            # update the fit as well!
            # get the particle, convert to torch tensor, calculate body supports
            self.track_or_guess = 'track'

            if self.track_or_guess == 'track':
                part = self.tracking_holder[:-1,frame-self.start_frame]
            if self.track_or_guess == 'guess':
                part = self.guessing_holder[:-1,frame-self.start_frame]

            part = torch.from_numpy(part).float().unsqueeze(0).cuda()
            body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
            body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
            update_skeleton(body_support_0,body_idx=0)
            update_skeleton(body_support_1,body_idx=1)
            update_ellipsoids(part,body_support_0,body_idx=0)
            update_ellipsoids(part,body_support_1,body_idx=1)
            
            calculate_new_trace(frame,trace_length=self.trace_length,trace_clip = self.trace_clip,decay_factor=.9, type_list = self.type_list)
            
            update_frame_number(frame)
            update_frame_time(frame)
            if self.v_ed is not None:
                update_ed(frame,case='ed')
                update_ed(frame,case='ed_reject')
                
        # trick for updating line3d collection
        # https://mail.python.org/pipermail/matplotlib-users/2015-October/000066.html

        
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, frame_list[1:], interval=10)

        fps = 10
        if savepath is None:
            fn = '<"(, ,)~~'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        
        ani.save(savepath,writer='imagemagick',fps=fps)
        plt.rcParams['animation.html'] = 'html5'
        ani

    def rotation(self,frame_list=30*60+np.arange(100)*100, savepath = None,view_override = None):
        
        # MAKE STARTING PLOT
        
        self.trace_length = 960
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        self.track_or_guess = 'track'
        self.add_fit(frame_list[0])
#         self.track_or_guess = 'guess'
        self.add_trace(frame_list[0],trace_length=self.trace_length,trace_clip = self.trace_clip ,type_list = self.type_list)
        self.track_or_guess = 'track'
        
        if self.v_ed is not None:
            self.add_ed(frame_list[0],case='ed')
            self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])

        
        # LOTS OF FUNCTIONS FOR ANIMATION
        def update_pc(i):
            # get new raw data!
            
            offset = el*np.sin(i*2*np.pi / 360)
            plt.gca().view_init(elev=el+offset, azim=az+i)
            
                
        # trick for updating line3d collection
        # https://mail.python.org/pipermail/matplotlib-users/2015-October/000066.html

        
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, np.linspace(0,360,40), interval=10)

        fps = 10
        if savepath is None:
            fn = '<"(, ,)~~_rotation'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD_rotation'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        
        ani.save(savepath,writer='imagemagick',fps=fps)
        plt.rcParams['animation.html'] = 'html5'
        ani        
        
        
        
    def plot_residuals(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
        
        # and get the corresponding fit: the particle, convert to torch tensor, calculate body supports
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
#         body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
#         body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        pos = torch.from_numpy(pos).float().cuda()
        
        # CALCULATE RESIDUAL
        dist0,_,self.body_support_0 = particles_to_distance_cuda(part[:,:9],pos[:,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(part[:,9:],pos[:,:])
        r = torch.min(dist0[0,:],dist1[0,:])
        r = r.cpu().numpy().squeeze()
        
        # Figure out the weighing stuff
        w2 = pos_weights**2

        rw=np.clip(r,0,.03)
        
        # or weigted mean?
#         rm = np.mean(rw)
        
        # correct
        rm = np.sum(w2*rw)/np.sum(w2)
        # wrong old way:
#         rm = np.mean(w2*rw)/np.median(w2)

        plt.figure()
        plt.plot(r,'.k',label='residual')
        plt.plot(rw,'.r',label='cut')
        plt.plot(w2,'.g',label='w2')
        plt.legend()
        plt.show()
        
#         plt.figure()
#         plt.hist(w2,100)
#         plt.show()
        
        plt.figure(figsize =(10,10))
#         plt.scatter(X,Y,c = 1-rw/np.max(rw))
        plt.scatter(X,Y,c = w2/np.max(w2))

        plt.axis('square')        
        plt.title("clipped: {}, weighted: {}".format(np.mean(rw),rm))
        plt.show()


        pass
# tips https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
# add raw data to plot

# add particle

    def kernel_smoothing(self,width = 3):
        kernel = easy_kernel(width)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')


                 
            
from filterpy.common import kinematic_kf,Q_discrete_white_noise
from filterpy.kalman import FixedLagSmoother

def kalman1_3D(tr,sigma_process,sigma_measure,dt = 1/60):
    # make first order kinematic kalman filter
    cv = kinematic_kf(dim=3, order=1, dt = dt) 
    cv.R = np.eye(3) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0,2,4]:
        cv.Q[i:(i+2),i:(i+2)] = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001
    
    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0,0],0,tr[0,1],0,tr[0,2] ,0 ]]).T
    cv.update(tr[i,:])
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i,:][:,np.newaxis])

        kalman_estimate.append(cv.x)

    kalman_estimate = np.hstack(kalman_estimate)    
    tr_filtered = kalman_estimate[[0,2,4],:].T
    return tr_filtered

def fls1_3d(tr,sigma_process,sigma_measure,dt = 1/60,N_lag = 16):
    # make first order kinematic kalman filter
    cv = kinematic_kf(dim=3, order=1, dt = dt) 
    cv.R = np.eye(3) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0,2,4]:
        cv.Q[i:(i+2),i:(i+2)] = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001
    
    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0,0],0,tr[0,1],0,tr[0,2] ,0 ]]).T
    
    # also make an FLS smoother 
    fls = FixedLagSmoother(dim_x=6, dim_z=3, N=N_lag)

    fls.x = np.copy(cv.x)
    fls.F = np.copy(cv.F)
    fls.H = np.copy(cv.H)
    fls.P = np.copy(cv.P)
    fls.R = np.copy(cv.R)
    fls.Q = np.copy(cv.Q)
    
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i,:][:,np.newaxis])
        fls.smooth(tr[i,:][:,np.newaxis])
        kalman_estimate.append(cv.x)

    kalman_estimate = np.hstack(kalman_estimate)    
    fls_estimate = np.hstack(fls.xSmooth)
    tr_filtered = kalman_estimate[[0,2,4],:].T
    tr_smoothed = fls_estimate[[0,2,4],:].T
    return tr_smoothed

def fls2_3d(tr,sigma_process,sigma_measure,dt = 1/60,N_lag = 16):
    # make second order kinematic kalman filter
    cv = kinematic_kf(dim=3, order=2, dt = dt) 
    cv.R = np.eye(3) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0,3,6]:
        cv.Q[i:(i+3),i:(i+3)] = Q_discrete_white_noise(dim=3, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001
    
    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0,0],0,0,tr[0,1],0,0,tr[0,2],0 ,0 ]]).T

        
    # also make an FLS smoother 
    fls = FixedLagSmoother(dim_x=9, dim_z=3, N=N_lag)

    fls.x = np.copy(cv.x)
    fls.F = np.copy(cv.F)
    fls.H = np.copy(cv.H)
    fls.P = np.copy(cv.P)
    fls.R = np.copy(cv.R)
    fls.Q = np.copy(cv.Q)
    
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i,:][:,np.newaxis])
        fls.smooth(tr[i,:][:,np.newaxis])

        kalman_estimate.append(cv.x)

    kalman_estimate = np.hstack(kalman_estimate)    
    fls_estimate = np.hstack(fls.xSmooth)
    tr_filtered = kalman_estimate[[0,3,6],:].T
    tr_smoothed = fls_estimate[[0,3,6],:].T
    return tr_smoothed

def kalman_1D(tr,sigma_process,sigma_measure,dt = 1/60):
    # make first order kinematic kalman filter
    cv = kinematic_kf(dim=1, order=1, dt = dt) 
    cv.R = np.eye(1) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0]:
        cv.Q[i:(i+2),i:(i+2)] = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001

    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0],0 ]]).T
    cv.update(tr[i])
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i])

        kalman_estimate.append(cv.x)

    kalman_estimate = np.hstack(kalman_estimate)    
    tr_filtered = kalman_estimate[[0],:].T
    return tr_filtered

def kalman2_3D(tr,sigma_process,sigma_measure,dt = 1/60):
    # make second order kinematic kalman filter
    cv = kinematic_kf(dim=3, order=2, dt = dt) 
    cv.R = np.eye(3) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0,3,6]:
        cv.Q[i:(i+3),i:(i+3)] = Q_discrete_white_noise(dim=3, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001
    
    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0,0],0,0,tr[0,1],0,0,tr[0,2],0 ,0 ]]).T
    cv.update(tr[i,:])
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i,:][:,np.newaxis])

        kalman_estimate.append(cv.x)

    kalman_estimate = np.hstack(kalman_estimate)    
    tr_filtered = kalman_estimate[[0,3,6],:].T
    return tr_filtered

def fls2_1d(tr,sigma_process,sigma_measure,dt = 1/60,N_lag = 16):
    # make second order kinematic kalman filter
    cv = kinematic_kf(dim=1, order=2, dt = dt) 
    cv.R = np.eye(1) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0]:
        cv.Q[i:(i+2),i:(i+2)] = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001
    
    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0],0,0]]).T

        
    # also make an FLS smoother 
    fls = FixedLagSmoother(dim_x=3, dim_z=1, N=N_lag)

    fls.x = np.copy(cv.x)
    fls.F = np.copy(cv.F)
    fls.H = np.copy(cv.H)
    fls.P = np.copy(cv.P)
    fls.R = np.copy(cv.R)
    fls.Q = np.copy(cv.Q)
    
#     print(cv)
#     print(fls)
    
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i])
        fls.smooth(tr[i])

        kalman_estimate.append(cv.x)

    kalman_estimate = np.hstack(kalman_estimate)    
    fls_estimate = np.hstack(fls.xSmooth)
    tr_filtered = kalman_estimate[[0],:].T
    tr_smoothed = fls_estimate[[0],:].T
    return tr_smoothed

def fls1_1d(tr,sigma_process,sigma_measure,dt = 1/60,N_lag = 16):
    # make second order kinematic kalman filter
    cv = kinematic_kf(dim=1, order=1, dt = dt) 
    cv.R = np.eye(1) * sigma_measure**2
    G = np.array([[0.5 * dt**2, dt]], np.float32).T
    Q0 = np.matmul(G, G.T) * sigma_process**2

    for i in [0]:
        cv.Q[i:(i+2),i:(i+2)] = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_process**2)
    #     cv.Q[i:(i+2),i:(i+2)] = Q0
    cv.P = np.ones((cv.dim_x,cv.dim_x))*0.001 +.0001
    
    kalman_estimate = []
    # initialize
    cv.x =  np.array([[ tr[0],0]]).T

    
    # also make an FLS smoother 
    fls = FixedLagSmoother(dim_x=2, dim_z=1, N=N_lag)

    fls.x = np.copy(cv.x)
    fls.F = np.copy(cv.F)
    fls.H = np.copy(cv.H)
    fls.P = np.copy(cv.P)
    fls.R = np.copy(cv.R)
    fls.Q = np.copy(cv.Q)
    
    fls.x =  np.array([[ .9,0]]).T
    fls.P = np.array([[2.73445008e-04 ,2.49619926e-05],[2.49619926e-05, 4.56088374e-06]])
    
    for i in tqdm(range(tr.shape[0])):
        cv.predict()
        cv.update(tr[i])
        fls.smooth(tr[i])
        kalman_estimate.append(cv.x)
        
    kalman_estimate = np.hstack(kalman_estimate)    
    fls_estimate = np.hstack(fls.xSmooth)
    tr_filtered = kalman_estimate[[0],:].T
    tr_smoothed = fls_estimate[[0],:].T
    return tr_smoothed

            
def smooth_body_support(body_support,s):
    # unpack to 3D coordinates
    c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy().squeeze() for i in body_support]
    type_list = np.array(['hip','tail','mid','nose','tip','impl'])
    c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]

    # run function for 3D kalman filtering (body coordinates)
    sigma_process = .01
    sigma_measure = .015
    c_hip = fls2_3d(c_hip,sigma_process,sigma_measure,dt = 1/60)
    c_ass = fls2_3d(c_ass,sigma_process,sigma_measure,dt = 1/60)
    c_mid = fls2_3d(c_mid,sigma_process,sigma_measure,dt = 1/60)
    c_nose = fls2_3d(c_nose,sigma_process,sigma_measure,dt = 1/60)
    c_tip = fls2_3d(c_tip,sigma_process,sigma_measure,dt = 1/60)
    sigma_process = .01
    sigma_measure = .02
    c_impl = fls2_3d(c_impl,sigma_process,sigma_measure,dt = 1/60)


    # run function for 1D kalman filtering (s)
    sigma_process = .05
    sigma_measure = .3
    s = fls1_1d(s,sigma_process,sigma_measure,dt = 1/60)
    
    return [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose],s            
            
# calculate rotation matrices from the smoothed skeleton points, to be used for smooth video playback

def rotation_matrix_vec2vec_numpy(f,t):
    # from this paper, ffrom math stacj
    # but made batch-able for pytorch
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    #rotate vector f onto vector t
    # import numpy as np
    # v = np.cross(f, t)
    # u = v/np.linalg.norm(v)
    # c = np.dot(f, t)
    # h = (1 - c)/(1 - c**2)

    # vx, vy, vz = v
    # rot =[[c + h*vx**2, h*vx*vy - vz, h*vx*vz + vy],
    #       [h*vx*vy+vz, c+h*vy**2, h*vy*vz-vx],
    #       [h*vx*vz - vy, h*vy*vz + vx, c+h*vz**2]]

    # good disussion about smoothing rotation matrices later: https://www.cvl.isy.liu.se/education/graduate/geometry2010/lectures/Lecture7b.pdf
    # rotate f onto t
    # very fast, but slightly numerically unstable, so we add epsilon!



    epsilon = 1e-6
    # f = x_pointer
    # t = nose_pointer
    # cross product
    v = np.cross(f,t)
    u = v/(np.linalg.norm(v,axis=1)[:,np.newaxis] + epsilon)
    # dot product
    c = np.einsum('i,ai->a', f,t)
    # the factor h
    h = (1 - c)/(1 - c**2 + epsilon)

    vx, vy, vz = v[:,0],v[:,1],v[:,2]

    R = np.stack([np.stack([c + h*vx**2, h*vx*vy - vz, h*vx*vz + vy], axis=1),
                             np.stack([h*vx*vy+vz, c+h*vy**2, h*vy*vz-vx],      axis=1),
                             np.stack([h*vx*vz - vy, h*vy*vz + vx, c+h*vz**2], axis=1)], axis=1)

    return R

# use quarterneons to smooth the body ellipsoid rotations
# convert the rotation matrices to quarternions
from pyquaternion import Quaternion

def unpack_axis_angels(R):
    # calculate the axis-angle representation
    # https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
    angle_x = np.arctan2(R[:,2,1],R[:,2,2])
    angle_y = np.arctan2(-R[:,2,0],np.sqrt(R[:,2,1]**2 + R[:,2,2]**2  ) )
    angle_z = np.arctan2(R[:,1,0],R[:,0,0])
    return np.stack((angle_x,angle_y,angle_z),axis = 1)

def averageQuaternions(Q):
    # from https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].ravel())    

def quaternion_smoothing(R):
    angles_kalman = unpack_axis_angels(R)     
    q_x = [ Quaternion(axis = (1.,0.,0.),radians = r) for r in angles_kalman[:,0] ]
    q_y = [ Quaternion(axis = (0.,1.,0.),radians = r) for r in angles_kalman[:,1] ]
    q_z = [ Quaternion(axis = (0.,0.,1.),radians = r) for r in angles_kalman[:,2] ]
    # now, smooth the rotations
    q_all = []
    for i in range(len(q_x)):
        q_all.append(q_x[i]*q_y[i]*q_z[i])

    # convert to a matrix w qith w x y z    
    Q = np.stack([q.elements for q in q_all],axis = 0)

    # try a running average first!
    Q_run_av = np.copy(Q)
    w_length = 9 # must be uneven
    h_length = int(np.floor(w_length/2))
    for i in tqdm(np.arange(np.floor(w_length/2),Q.shape[0]-np.ceil(w_length/2))):
        i = int(i) 
        Q_run_av[i,:] = averageQuaternions(Q[(i-h_length):(i+h_length+1),:])


    # convert back to rotation matrices, to check that converion is fine
    R_q_list = [q.rotation_matrix for q in q_all]
    R_q = np.stack(R_q_list,axis = 0)

    R_q_smooth_list = [Quaternion(Q_run_av[i,:]).rotation_matrix for i in range(Q_run_av.shape[0])]
    R_q_smooth = np.stack(R_q_smooth_list,axis = 0)
    return R_q,R_q_smooth

def smooth_rotation_matrices(body_support_0_smooth):
    # first calculate body vectors from the smoothed skeleton points
    c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support_0_smooth

    #todo, maybe average across the both noisy estimates here, prob won't gain much though..
    v_nose = c_nose - c_mid
    v_hip = c_mid - c_hip

    v_nose = v_nose/np.linalg.norm(v_nose,axis=1)[:,np.newaxis]
    v_hip = v_hip/np.linalg.norm(v_hip,axis=1)[:,np.newaxis]


    # To calculate R_nose, we ask how we have to rotate a vector along x, so that it points along the hip or nose
    f = np.array([1,0,0])
    t = v_nose
    R_nose_smooth = rotation_matrix_vec2vec_numpy(f,t)

    t = v_hip
    R_body_smooth = rotation_matrix_vec2vec_numpy(f,t)
    return [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body_smooth,R_head,R_nose_smooth]

def smooth_rotation_matrices_quaternion(body_support_0_smooth):
    # first calculate body vectors from the smoothed skeleton points
    c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support_0_smooth

    #todo, maybe average across the both noisy estimates here, prob won't gain much though..
    v_nose = c_nose - c_mid
    v_hip = c_mid - c_hip

    v_nose = v_nose/np.linalg.norm(v_nose,axis=1)[:,np.newaxis]
    v_hip = v_hip/np.linalg.norm(v_hip,axis=1)[:,np.newaxis]


    # To calculate R_nose, we ask how we have to rotate a vector along x, so that it points along the hip or nose
    f = np.array([1,0,0])
    t = v_nose
    R_nose_smooth = rotation_matrix_vec2vec_numpy(f,t)

    t = v_hip
    R_body_smooth = rotation_matrix_vec2vec_numpy(f,t)
    
    R_q,R_body_smooth = quaternion_smoothing(R_body_smooth)
    R_q,R_nose_smooth = quaternion_smoothing(R_nose_smooth)
    
    return [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body_smooth,R_head,R_nose_smooth]

    
    
    

class VideoPlotMachine(object):
    def __init__(self,tracked_behavior,jagged_lines,what_to_plot = 'guess'):
        self.tracked_behavior = tracked_behavior
        self.jagged_lines = jagged_lines
        # useful functions, get the raw data in plottable format
        # pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[start_frame])
        # useful functions, get the fitted data in plottable format
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Say, "the default sans-serif font is COMIC SANS"
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        # Then, "ALWAYS use sans-serif fonts"
        matplotlib.rcParams['font.family'] = "sans-serif"

        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('text', usetex='false') 
        matplotlib.rcParams.update({'font.size': 13})

        from palettable.cmocean.sequential import Algae_6
        cmpl = Algae_6.mpl_colors
        
        # unpack 
        self.guessing_holder = tracked_behavior['guessing_holder']
        self.tracking_holder = tracked_behavior['tracking_holder']
        self.start_frame = tracked_behavior['start_frame']
        
        self.track_or_guess = what_to_plot
        self.n_frames = self.tracking_holder.shape[1]
        
        self.v_ed = None
        self.v_ed_reject = None

    def better_smoothing(self):
        # get the raw tracking data!
        part = self.tracking_holder

        # unpack all the 3D coordinates!
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        # and the spine length
        s_0 = part[:,2].cpu().numpy()
        s_1 = part[:,2+9].cpu().numpy()

        # and smooth the data
#         from utils.analysis_tools import smooth_body_support
        body_support_0_smooth,s_0_smooth = smooth_body_support(body_support_0,s_0)
        body_support_1_smooth,s_1_smooth = smooth_body_support(body_support_1,s_1)

        # add the smoothed coordinates as numpy arrays
        self.body_support_0_raw = [i.cpu().numpy().squeeze() for i in body_support_0]
        self.body_support_0_smooth = body_support_0_smooth
        self.s_0_raw = s_0
        self.s_0_smooth = s_0_smooth
        self.body_support_1_raw = [i.cpu().numpy().squeeze() for i in body_support_1]
        self.body_support_1_smooth = body_support_1_smooth
        self.s_1_raw = s_1
        self.s_1_smooth = s_1_smooth
        # also smooth the body ellipsoid rotations
        self.body_support_0_smooth = smooth_rotation_matrices(body_support_0_smooth)
        self.body_support_1_smooth = smooth_rotation_matrices(body_support_1_smooth)
        # self.body_support_0_smooth = smooth_rotation_matrices_quaternion(body_support_0_smooth)
        # self.body_support_1_smooth = smooth_rotation_matrices_quaternion(body_support_1_smooth)

    def kernel_smoothing(self):
        def easy_kernel(kernel_width = 30):
            from scipy import stats
            kernel = stats.norm.pdf(np.arange(-3*kernel_width,3*kernel_width+1),scale=kernel_width)
            kernel = kernel/np.sum(kernel)
            return kernel
        kernel = easy_kernel(3)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')
    
    def make_3d_axis(self):
        #   3D plot of the 
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        # add to self for use later
        self.fig = fig
        self.ax = ax
        
    def add_raw_data(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]

        # add to axis 3D plot of Sphere
        self.h_pc = self.ax.scatter(X, Y, Z, zdir='z', s=10, c='k', alpha = .05,rasterized=False)
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        # loop over the types of body, and make emptyscatter plots
        self.h_kp_list = []        
        for body in body_indices:
            h_kp = self.ax.scatter([],[],[], zdir='z', s=100, c=body_colors[body],rasterized=False)
            self.h_kp_list.append(h_kp)
        
        # THEN set the 3d values to be what the shoud be
        for body in body_indices:
            self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])
        
        
        
#         self.h_kp_list = []
#         for i,body in enumerate(ikeyp):
#             h_kp = self.ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[body],rasterized=False)
#             self.h_kp_list.append(h_kp)
        # for axis adjustment
        self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        self.mid_x = (X.max()+X.min()) * 0.5
        self.mid_y = (Y.max()+Y.min()) * 0.5
        self.mid_z = (Z.max()+Z.min()) * 0.5
            
    def plot_skeleton(self,body_support,color = 'k',body_idx = 0):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        #print("c_hip is {}".format(c_hip))
        if body_idx == 0 :
            p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl] 
            p_line = [c_nose,c_nose,c_mid,c_impl,c_impl]
            q_line = [c_mid,c_tip,c_ass,c_nose,c_tip]
        elif body_idx == 1:
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
                    
    def add_wireframe_to_axis(self,ax,R_body,c_hip, a_nose,b_nose,a_hip,b_hip,r_impl,style='hip',this_color='k',this_alpha=.4):
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

        h_hip = ax.plot_wireframe(x, y, z, color=this_color,alpha = this_alpha)
        return h_hip        
                    
    def plot_ellipsoids(self,body_support,s,color = 'k',body_idx = 0):
        # unpack
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
        # this is not so elegant, hm hm
        a_hip = a_hip_0.cpu().numpy() + a_hip_delta.cpu().numpy() * s
        b_hip = b_hip_0.cpu().numpy() + b_hip_delta.cpu().numpy() * (1.-s)
        d_hip = .75 * a_hip

        if body_idx == 0:
            RRs,ccs,styles = [R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']
        if body_idx == 1:            
            RRs,ccs,styles = [R_body,R_nose],[c_hip,c_nose],['hip','nose']
        
        for RR,cc,style in zip(RRs,ccs,styles):

            h_hip = self.add_wireframe_to_axis(self.ax,RR,
                                                   cc,
                                                   a_nose.cpu().numpy(),
                                                   b_nose.cpu().numpy(),
                                                   a_hip,
                                                   b_hip,
                                                   r_impl.cpu().numpy(),
                                                   style=style,this_color=color)
            self.h_hip_list[body_idx].append(h_hip)
            
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
        
        self.plot_skeleton(body_support_0,color = 'k',body_idx = 0)
        self.plot_skeleton(body_support_1,color = 'peru',body_idx = 1)                

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
        self.plot_ellipsoids(body_support_0,s_0,color = 'k',body_idx = 0)
        self.plot_ellipsoids(body_support_1,s_1,color = 'peru',body_idx = 1)
            
    def add_trace(self,frame,trace,trace_length=90,trace_clip = None,decay_factor=.9, type_list = ['tip']):
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
        
    def finish_3d_axis(self,view_style = 'top', zoom = False, dump = False):
        # finish the labeling, plot adjustments, dump and show
        ax = self.ax
#         ax.set_xlabel('$x$ (mm)')
#         ax.set_ylabel('\n$y$ (mm)')
#         zlabel = ax.set_zlabel('\n$z$ (mm)')

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
            
        if zoom:
            def mean_point(body_support,impl = False):
                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
                if impl==True:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip,c_impl])
                else:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip])
                mean_0 = np.mean(points_0,0)
                return mean_0
            mean_0 = mean_point(self.body_support_0,impl= True)
            mean_1 = mean_point(self.body_support_1, impl = False)
            
            overall_mean = np.hstack([mean_0,mean_1])
            
            mu_zoom = np.mean(overall_mean,1)
            scaling = 2
            d_zoom = ( mean_0.ravel() - mean_1.ravel() )*scaling
#             print(d_zoom)
#             print(mu_zoom)
#             self.ax.scatter(mu_zoom[0],mu_zoom[1],mu_zoom[2],zdir='z', s=500, alpha = 1 , c='pink',rasterized=False)
            
            pp = np.vstack([mu_zoom-d_zoom,mu_zoom,mu_zoom+d_zoom])
#             print(pp)
#             self.ax.plot(pp[:,0],pp[:,1],pp[:,2],zdir='z',alpha = 1,lw=10 , c='pink',rasterized=False)
            
            X,Y,Z = pp[:,0],pp[:,1],pp[:,2]
            
            # for axis adjustment
            self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            self.mid_x = (X.max()+X.min()) * 0.5
            self.mid_y = (Y.max()+Y.min()) * 0.5
            self.mid_z = (Z.max()+Z.min()) * 0.5
            
            ax.set_xlim(self.mid_x - self.max_range, self.mid_x + self.max_range)
            ax.set_ylim(self.mid_y - self.max_range, self.mid_y + self.max_range)
            ax.set_zlim(0, 2*self.max_range)
            
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            
        ax.view_init(elev=el, azim=az)

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

    def dump_to_disk(self,tag = ""):
        print("saving w/tag: "+tag)
        plt.tight_layout()
        plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/Plotter'+tag+'.pdf',transparent=True)  

    def add_frame_number(self,frame):
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.74, 0.07, "Frame:" ,ha='left',fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_number = self.ax.text2D(0.95, 0.07, str(frame),ha='right', fontsize = 20, transform=self.ax.transAxes)
        pass
    
    def add_frame_time(self,frame,start_frame):
        cam_fps = 60 #Hz
        t_now = (frame-start_frame)/cam_fps
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.71, 0.03, "Time:" ,ha='left', fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_time = self.ax.text2D(0.95, 0.03, "{:0.2f} s".format(t_now) ,ha='right',fontsize = 20, transform=self.ax.transAxes)
        pass    

    def add_ed(self,frame,case='ed'):
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
        c_nose = np.squeeze(c_mid)
        if case == 'ed':
            v_ed = self.v_ed[frame-self.start_frame,:]
            self.h_ed = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c='r',lw = 4)
        if case == 'ed_reject':
            v_ed = self.v_ed_reject[frame-self.start_frame,:]
            ccc = 'blueviolet'
            self.h_ed_reject = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c=ccc,lw = 4)

        pass
           
    def make_me(self,frame=None,cloud=True,trace='raw',skel='raw',ellip='raw',trace_clip=30*60-960 ,savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        # trace is raw or smooth or None
        self.make_3d_axis()
        if frame is not None:
            if cloud:
                self.add_raw_data(frame)
            
            #self.track_or_guess = 'track' #hk
            self.add_skel_fit(frame,fit=skel)
            self.add_ellip_fit(frame,fit=ellip)

            #self.track_or_guess = 'track'
            self.add_trace(frame,trace=trace,trace_length=10*60,trace_clip = trace_clip ,type_list = ['nose'])
            
#             if self.v_ed is not None:
#                 self.add_ed(frame,case='ed')
#                 self.add_ed(frame,case='ed_reject')
            
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)

        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()     
        
    def video_me(self,frame_list=30*60+np.arange(100)*100,cloud=True,trace='raw',skel='raw',ellip='raw',savepath = None,view_override = None,fps=10,dpi=100,time_offset=0):
        
        
        self.trace_length = 10*60
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        # add the fitted values
        self.add_skel_fit(frame_list[0],fit=skel)
        self.add_ellip_fit(frame_list[0],fit=ellip)        
        self.add_trace(frame_list[0],trace=trace,trace_length=10*60,trace_clip = 30*60-960 ,type_list = ['nose'])
        
#         if self.v_ed is not None:
#             self.add_ed(frame_list[0],case='ed')
#             self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        # tighten and add frame # and time
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])

        
        # LOTS OF FUNCTIONS FOR ANIMATION
        
        def update_frame_number(frame):
#             self.h_frame_number.set_text("Frame: {}".format(frame))
            self.h_frame_number.set_text(str(frame))

        def update_frame_time(frame,start_frame = frame_list[0],time_offset=0):
            cam_fps = 60. #Hz
            t_now = (frame-start_frame)/cam_fps - time_offset
            self.h_frame_time.set_text("{:0.2f} s".format(t_now))

        def update_skeleton(body_support,body_idx = 0):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
            if body_idx == 0 :
                p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl] 
                p_line = [c_nose,c_nose,c_mid,c_impl,c_impl]
                q_line = [c_mid,c_tip,c_ass,c_nose,c_tip]
            elif body_idx == 1:
                p_skel = [c_hip,c_mid,c_nose,c_ass,c_tip] 
                p_line = [c_nose,c_nose,c_mid]
                q_line = [c_mid,c_tip,c_ass]

            # update the body points
            for j,p in enumerate(p_skel):
                self.h_bp_list[body_idx][j]._offsets3d = ([p[0]],[p[1]],[p[2]])

            # update the lines between body parts
            for j,(p,q) in enumerate(zip(p_line,q_line)):
                # lines are an extra level deep for some stupid matplotlib reason
                self.h_skel_list[body_idx][j][0].set_xdata([p[0],q[0]])
                self.h_skel_list[body_idx][j][0].set_ydata([p[1],q[1]])
                self.h_skel_list[body_idx][j][0].set_3d_properties([p[2],q[2]])                

        def update_skel_fit(frame,fit=skel):
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
            update_skeleton(body_support_0,body_idx = 0)
            update_skeleton(body_support_1,body_idx = 1)               
                        
        def update_wireframe_lines(h_hip,X,Y,Z):
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
        
        def calculate_wireframe_points(R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip'):
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

        def update_ellipsoids(body_support,s,body_idx = 0):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
            # this is not so elegant, hm hm
            a_hip = a_hip_0.cpu().numpy() + a_hip_delta.cpu().numpy() * s
            b_hip = b_hip_0.cpu().numpy() + b_hip_delta.cpu().numpy() * (1.-s)
            d_hip = .75 * a_hip

            if body_idx == 0:
                RRs,ccs,styles = [R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']
            if body_idx == 1:            
                RRs,ccs,styles = [R_body,R_nose],[c_hip,c_nose],['hip','nose']
            
            for jj, (RR,cc,style) in enumerate(zip(RRs,ccs,styles)):

                X,Y,Z = calculate_wireframe_points(RR,
                                                       cc,
                                                       a_nose.cpu().numpy(),
                                                       b_nose.cpu().numpy(),
                                                       a_hip,
                                                       b_hip,
                                                       r_impl.cpu().numpy(),
                                                       style=style)
                h_hip = self.h_hip_list[body_idx][jj]
                update_wireframe_lines(h_hip,X,Y,Z)             
            
        def update_ellip_fit(frame,fit = ellip):
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
            
            update_ellipsoids(body_support_0,s_0,body_idx = 0)
            update_ellipsoids(body_support_1,s_1,body_idx = 1)
            
        def update_trace_3dlines(body_support,trace_indices,body_idx=0,what_type=['hip']):
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
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_xdata(X)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_ydata(Y)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_3d_properties(Z)
                    
                
        def update_trace_fit(frame,trace='raw',trace_length=None,trace_clip = None,decay_factor=.9, type_list = None):
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

            
            update_trace_3dlines(body_support_0,trace_indices,body_idx=0,what_type = type_list)
            update_trace_3dlines(body_support_1,trace_indices,body_idx=1,what_type = type_list)


        if self.v_ed is not None:
            def update_ed(frame,case='ed'):
                if self.track_or_guess == 'track':
                    part = self.tracking_holder[:-1,frame-self.start_frame]
                if self.track_or_guess == 'guess':
                    part = self.guessing_holder[:-1,frame-self.start_frame]

                part = torch.from_numpy(part).float().unsqueeze(0).cuda()
                body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
                body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
                c_nose = np.squeeze(c_mid)
                if case == 'ed':
                    v_ed = self.v_ed[frame-self.start_frame,:]
                    self.h_ed[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])
                if case == 'ed_reject':
                    v_ed = self.v_ed_reject[frame-self.start_frame,:]
                    self.h_ed_reject[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed_reject[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed_reject[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])

                pass            
        
        def update_pc(frame):
            # get new raw data!
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            # update the pointcloud
            self.h_pc._offsets3d = (X,Y,Z)
            # and update the keypoints
            for body in range(4):
                self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])
            # update the counters
            update_frame_number(frame)

            update_frame_time(frame,time_offset=time_offset)
                            
            # update the fit
            update_skel_fit(frame)
            update_ellip_fit(frame)       
            update_trace_fit(frame,trace=trace,trace_length=10*60,trace_clip = 30*60-960 ,type_list = ['nose'])                

#             # update the fit as well!
            
#             if self.v_ed is not None:
#                 update_ed(frame,case='ed')
#                 update_ed(frame,case='ed_reject')
                
        # trick for updating line3d collection
        # https://mail.python.org/pipermail/matplotlib-users/2015-October/000066.html

        
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, frame_list[1:], interval=10)

        if savepath is None:
            fn = '<"(, ,)~~'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        if savepath[-1] == 'f':
            ani.save(savepath,writer='imagemagick',fps=fps,dpi = dpi)
        else:
            ani.save(savepath,writer='ffmpeg',fps=fps,dpi=dpi)
        plt.rcParams['animation.html'] = 'html5'
        ani

    def rotation(self,frame_list=30*60+np.arange(100)*100,cloud=True,trace='raw',skel='raw',ellip='raw',savepath = None,view_override = None,fps=10,dpi=100):
        
        
        self.trace_length = 10*60
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        # add the fitted values
        self.add_skel_fit(frame_list[0],fit=skel)
        self.add_ellip_fit(frame_list[0],fit=ellip)        
        self.add_trace(frame_list[0],trace=trace,trace_length=10*60,trace_clip = 30*60-960 ,type_list = ['nose'])
        
#         if self.v_ed is not None:
#             self.add_ed(frame_list[0],case='ed')
#             self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        # tighten and add frame # and time
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])
        
        # LOTS OF FUNCTIONS FOR ANIMATION
        def update_pc(i):
            # here, the update rule is simply to change the view
            offset = el*np.sin(i*2*np.pi / 360)
            plt.gca().view_init(elev=el+offset, azim=az+i)
            
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, np.linspace(0,360,80), interval=10)

        fps = 10
        if savepath is None:
            fn = '<"(, ,)~~_rotation'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD_rotation'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        
        if savepath[-1] == 'f':
            ani.save(savepath,writer='imagemagick',fps=fps,dpi=dpi)
        else:
            ani.save(savepath,writer='ffmpeg',fps=fps,dpi=dpi)
        plt.rcParams['animation.html'] = 'html5'
        ani        
        
        
        
    def plot_residuals(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
        
        # and get the corresponding fit: the particle, convert to torch tensor, calculate body supports
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
#         body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
#         body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        pos = torch.from_numpy(pos).float().cuda()
        
        # CALCULATE RESIDUAL
        dist0,_,self.body_support_0 = particles_to_distance_cuda(part[:,:9],pos[:,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(part[:,9:],pos[:,:])
        r = torch.min(dist0[0,:],dist1[0,:])
        r = r.cpu().numpy().squeeze()
        
        # Figure out the weighing stuff
        w2 = pos_weights**2

        rw=np.clip(r,0,.03)
        
        # or weigted mean?
#         rm = np.mean(rw)
        
        # correct
        rm = np.sum(w2*rw)/np.sum(w2)
        # wrong old way:
#         rm = np.mean(w2*rw)/np.median(w2)

        plt.figure()
        plt.plot(r,'.k',label='residual')
        plt.plot(rw,'.r',label='cut')
        plt.plot(w2,'.g',label='w2')
        plt.legend()
        plt.show()
        
#         plt.figure()
#         plt.hist(w2,100)
#         plt.show()
        
        plt.figure(figsize =(10,10))
#         plt.scatter(X,Y,c = 1-rw/np.max(rw))
        plt.scatter(X,Y,c = w2/np.max(w2))

        plt.axis('square')        
        plt.title("clipped: {}, weighted: {}".format(np.mean(rw),rm))
        plt.show()


        pass
# tips https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
# add raw data to plot

# add particle

    def kernel_smoothing(self,width = 3):
        kernel = easy_kernel(width)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')

            
    

class VideoPlotMachine_noimpl(object):
    def __init__(self,tracked_behavior,jagged_lines,what_to_plot = 'guess', has_implant = False):
        self.tracked_behavior = tracked_behavior
        self.jagged_lines = jagged_lines
        # useful functions, get the raw data in plottable format
        # pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[start_frame])
        # useful functions, get the fitted data in plottable format
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Say, "the default sans-serif font is COMIC SANS"
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        # Then, "ALWAYS use sans-serif fonts"
        matplotlib.rcParams['font.family'] = "sans-serif"

        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('text', usetex='false') 
        matplotlib.rcParams.update({'font.size': 13})

        from palettable.cmocean.sequential import Algae_6
        cmpl = Algae_6.mpl_colors
        
        # unpack 
        self.guessing_holder = tracked_behavior['guessing_holder']
        self.tracking_holder = tracked_behavior['tracking_holder']
        self.start_frame = tracked_behavior['start_frame']
        self.has_implant = has_implant
        self.track_or_guess = what_to_plot
        self.n_frames = self.tracking_holder.shape[1]
        
        self.v_ed = None
        self.v_ed_reject = None

    def better_smoothing(self):
        # get the raw tracking data!
        part = self.tracking_holder

        # unpack all the 3D coordinates!
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        # and the spine length
        s_0 = part[:,2].cpu().numpy()
        s_1 = part[:,2+9].cpu().numpy()

        # and smooth the data
#         from utils.analysis_tools import smooth_body_support
        body_support_0_smooth,s_0_smooth = smooth_body_support(body_support_0,s_0)
        body_support_1_smooth,s_1_smooth = smooth_body_support(body_support_1,s_1)

        # add the smoothed coordinates as numpy arrays
        self.body_support_0_raw = [i.cpu().numpy().squeeze() for i in body_support_0]
        self.body_support_0_smooth = body_support_0_smooth
        self.s_0_raw = s_0
        self.s_0_smooth = s_0_smooth
        self.body_support_1_raw = [i.cpu().numpy().squeeze() for i in body_support_1]
        self.body_support_1_smooth = body_support_1_smooth
        self.s_1_raw = s_1
        self.s_1_smooth = s_1_smooth
        # also smooth the body ellipsoid rotations
        self.body_support_0_smooth = smooth_rotation_matrices(body_support_0_smooth)
        self.body_support_1_smooth = smooth_rotation_matrices(body_support_1_smooth)
        # self.body_support_0_smooth = smooth_rotation_matrices_quaternion(body_support_0_smooth)
        # self.body_support_1_smooth = smooth_rotation_matrices_quaternion(body_support_1_smooth)

    def kernel_smoothing(self):
        def easy_kernel(kernel_width = 30):
            from scipy import stats
            kernel = stats.norm.pdf(np.arange(-3*kernel_width,3*kernel_width+1),scale=kernel_width)
            kernel = kernel/np.sum(kernel)
            return kernel
        kernel = easy_kernel(3)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')
    
    def make_3d_axis(self):
        #   3D plot of the 
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(111, projection='3d')
        # add to self for use later
        self.fig = fig
        self.ax = ax
        
    def add_raw_data(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]

        # add to axis 3D plot of Sphere
        self.h_pc = self.ax.scatter(X, Y, Z, zdir='z', s=10, c='k', alpha = .05,rasterized=False)
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        # loop over the types of body, and make emptyscatter plots
        self.h_kp_list = []        
        for body in body_indices:
            h_kp = self.ax.scatter([],[],[], zdir='z', s=100, c=body_colors[body],rasterized=False)
            self.h_kp_list.append(h_kp)
        
        # THEN set the 3d values to be what the shoud be
        for body in body_indices:
            self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])
        
        
        
#         self.h_kp_list = []
#         for i,body in enumerate(ikeyp):
#             h_kp = self.ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[body],rasterized=False)
#             self.h_kp_list.append(h_kp)
        # for axis adjustment
        self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        self.mid_x = (X.max()+X.min()) * 0.5
        self.mid_y = (Y.max()+Y.min()) * 0.5
        self.mid_z = (Z.max()+Z.min()) * 0.5
            
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
        a_hip = a_hip_0.cpu().numpy() + a_hip_delta.cpu().numpy() * s
        b_hip = b_hip_0.cpu().numpy() + b_hip_delta.cpu().numpy() * (1.-s)
        d_hip = .75 * a_hip

        if has_implant:
            RRs,ccs,styles = [R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']
        else:            
            RRs,ccs,styles = [R_body,R_nose],[c_hip,c_nose],['hip','nose']
        
        for RR,cc,style in zip(RRs,ccs,styles):

            h_hip = self.add_wireframe_to_axis(self.ax,RR,
                                                   cc,
                                                   a_nose.cpu().numpy(),
                                                   b_nose.cpu().numpy(),
                                                   a_hip,
                                                   b_hip,
                                                   r_impl.cpu().numpy(),
                                                   style=style,this_color=color)
            self.h_hip_list[body_idx].append(h_hip)
            
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
            
    def add_trace(self,frame,trace,trace_length=90,trace_clip = None,decay_factor=.9, type_list = ['tip']):
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
        
    def finish_3d_axis(self,view_style = 'top', zoom = False, dump = False):
        # finish the labeling, plot adjustments, dump and show
        ax = self.ax
#         ax.set_xlabel('$x$ (mm)')
#         ax.set_ylabel('\n$y$ (mm)')
#         zlabel = ax.set_zlabel('\n$z$ (mm)')

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
            
        if zoom:
            def mean_point(body_support,impl = False):
                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support]
                if impl==True:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip,c_impl])
                else:
                    points_0 = np.concatenate([c_hip,c_ass,c_mid,c_nose,c_tip])
                mean_0 = np.mean(points_0,0)
                return mean_0
            mean_0 = mean_point(self.body_support_0,impl= True)
            mean_1 = mean_point(self.body_support_1, impl = False)
            
            overall_mean = np.hstack([mean_0,mean_1])
            
            mu_zoom = np.mean(overall_mean,1)
            scaling = 2
            d_zoom = ( mean_0.ravel() - mean_1.ravel() )*scaling
#             print(d_zoom)
#             print(mu_zoom)
#             self.ax.scatter(mu_zoom[0],mu_zoom[1],mu_zoom[2],zdir='z', s=500, alpha = 1 , c='pink',rasterized=False)
            
            pp = np.vstack([mu_zoom-d_zoom,mu_zoom,mu_zoom+d_zoom])
#             print(pp)
#             self.ax.plot(pp[:,0],pp[:,1],pp[:,2],zdir='z',alpha = 1,lw=10 , c='pink',rasterized=False)
            
            X,Y,Z = pp[:,0],pp[:,1],pp[:,2]
            
            # for axis adjustment
            self.max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
            self.mid_x = (X.max()+X.min()) * 0.5
            self.mid_y = (Y.max()+Y.min()) * 0.5
            self.mid_z = (Z.max()+Z.min()) * 0.5
            
            ax.set_xlim(self.mid_x - self.max_range, self.mid_x + self.max_range)
            ax.set_ylim(self.mid_y - self.max_range, self.mid_y + self.max_range)
            ax.set_zlim(0, 2*self.max_range)
            
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            
        ax.view_init(elev=el, azim=az)

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

    def dump_to_disk(self,tag = ""):
        print("saving w/tag: "+tag)
        plt.tight_layout()
        plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/Plotter'+tag+'.pdf',transparent=True)  

    def add_frame_number(self,frame):
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.74, 0.07, "Frame:" ,ha='left',fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_number = self.ax.text2D(0.95, 0.07, str(frame),ha='right', fontsize = 20, transform=self.ax.transAxes)
        pass
    
    def add_frame_time(self,frame,start_frame):
        cam_fps = 60 #Hz
        t_now = (frame-start_frame)/cam_fps
        # self.ax.text2D(0.05, 0.95, "Frame {}".format(frame), transform=self.ax.transAxes)
        self.ax.text2D(0.71, 0.03, "Time:" ,ha='left', fontsize = 20, transform=self.ax.transAxes)
        self.h_frame_time = self.ax.text2D(0.95, 0.03, "{:0.2f} s".format(t_now) ,ha='right',fontsize = 20, transform=self.ax.transAxes)
        pass    

    def add_ed(self,frame,case='ed'):
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
        c_nose = np.squeeze(c_mid)
        if case == 'ed':
            v_ed = self.v_ed[frame-self.start_frame,:]
            self.h_ed = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c='r',lw = 4)
        if case == 'ed_reject':
            v_ed = self.v_ed_reject[frame-self.start_frame,:]
            ccc = 'blueviolet'
            self.h_ed_reject = self.ax.plot([c_nose[0],c_nose[0]+v_ed[0]],[c_nose[1],c_nose[1]+v_ed[1]],[c_nose[2],c_nose[2]+v_ed[2]],c=ccc,lw = 4)

        pass
           
    def make_me(self,frame=None,cloud=True,trace='raw',skel='raw',ellip='raw',trace_clip=10*60 ,savepath=None,view_override = None):
        # takes a frame number and plots it, with optional tail
        # trace is raw or smooth or None
        self.make_3d_axis()
        
        self.trace_length = 10*30
        self.trace_clip = 10*30
        
        if frame is not None:
            if cloud:
                self.add_raw_data(frame)
            
            #self.track_or_guess = 'track' #hk
            self.add_skel_fit(frame,fit=skel)
            self.add_ellip_fit(frame,fit=ellip)

            #self.track_or_guess = 'track'
            self.add_trace(frame,trace=trace,trace_length=10*60,trace_clip = trace_clip ,type_list = ['nose'])
            
#             if self.v_ed is not None:
#                 self.add_ed(frame,case='ed')
#                 self.add_ed(frame,case='ed_reject')
            
        self.finish_3d_axis(view_style='ex')
        self.add_frame_number(frame)

        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        if savepath is not None:
            plt.savefig(savepath,transparent=True)  
        plt.show()     
        
    def video_me(self,frame_list=30*60+np.arange(100)*100,cloud=True,trace='raw',skel='raw',ellip='raw',savepath = None,view_override = None,fps=10,dpi=100,time_offset=0):
        
        
        self.trace_length = 10*30
        self.trace_clip = frame_list[0]
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        # add the fitted values
        self.add_skel_fit(frame_list[0],fit=skel)
        self.add_ellip_fit(frame_list[0],fit=ellip)        
        self.add_trace(frame_list[0],trace=trace,trace_length=self.trace_length,trace_clip=self.trace_clip ,type_list = ['nose'])
        
#         if self.v_ed is not None:
#             self.add_ed(frame_list[0],case='ed')
#             self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        # tighten and add frame # and time
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])

        
        # LOTS OF FUNCTIONS FOR ANIMATION
        
        def update_frame_number(frame):
#             self.h_frame_number.set_text("Frame: {}".format(frame))
            self.h_frame_number.set_text(str(frame))

        def update_frame_time(frame,start_frame = frame_list[0],time_offset=0):
            cam_fps = 60. #Hz
            t_now = (frame-start_frame)/cam_fps - time_offset
            self.h_frame_time.set_text("{:0.2f} s".format(t_now))

        def update_skeleton(body_support,body_idx = 0, has_implant = False):
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
                # lines are an extra level deep for some stupid matplotlib reason
                self.h_skel_list[body_idx][j][0].set_xdata([p[0],q[0]])
                self.h_skel_list[body_idx][j][0].set_ydata([p[1],q[1]])
                self.h_skel_list[body_idx][j][0].set_3d_properties([p[2],q[2]])                

        def update_skel_fit(frame,fit=skel):
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
            update_skeleton(body_support_0,body_idx = 0, has_implant = self.has_implant)
            update_skeleton(body_support_1,body_idx = 1, has_implant = False)               
                        
        def update_wireframe_lines(h_hip,X,Y,Z):
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
        
        def calculate_wireframe_points(R_body,c_hip,a_nose,b_nose,a_hip,b_hip,r_impl,style='hip'):
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

        def update_ellipsoids(body_support,s,body_idx = 0, has_implant = False):
            # unpack
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
            # this is not so elegant, hm hm
            a_hip = a_hip_0.cpu().numpy() + a_hip_delta.cpu().numpy() * s
            b_hip = b_hip_0.cpu().numpy() + b_hip_delta.cpu().numpy() * (1.-s)
            d_hip = .75 * a_hip

            if has_implant:
                RRs,ccs,styles = [R_body,R_nose,R_nose],[c_hip,c_nose,c_impl],['hip','nose','impl']
            else:            
                RRs,ccs,styles = [R_body,R_nose],[c_hip,c_nose],['hip','nose']
            
            for jj, (RR,cc,style) in enumerate(zip(RRs,ccs,styles)):

                X,Y,Z = calculate_wireframe_points(RR,
                                                       cc,
                                                       a_nose.cpu().numpy(),
                                                       b_nose.cpu().numpy(),
                                                       a_hip,
                                                       b_hip,
                                                       r_impl.cpu().numpy(),
                                                       style=style)
                h_hip = self.h_hip_list[body_idx][jj]
                update_wireframe_lines(h_hip,X,Y,Z)             
            
        def update_ellip_fit(frame,fit = ellip):
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
            
            update_ellipsoids(body_support_0,s_0,body_idx = 0,has_implant = self.has_implant)
            update_ellipsoids(body_support_1,s_1,body_idx = 1,has_implant = False)
            
        def update_trace_3dlines(body_support,trace_indices,body_idx=0,what_type=['hip']):
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
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_xdata(X)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_ydata(Y)
                self.h_trace_list[body_idx][ii_c_list[type_list == what_type][0]][0].set_3d_properties(Z)
                    
                
        def update_trace_fit(frame,trace='raw',trace_length=None,trace_clip = None,decay_factor=.9, type_list = None):
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

            
            update_trace_3dlines(body_support_0,trace_indices,body_idx=0,what_type = type_list)
            update_trace_3dlines(body_support_1,trace_indices,body_idx=1,what_type = type_list)


        if self.v_ed is not None:
            def update_ed(frame,case='ed'):
                if self.track_or_guess == 'track':
                    part = self.tracking_holder[:-1,frame-self.start_frame]
                if self.track_or_guess == 'guess':
                    part = self.guessing_holder[:-1,frame-self.start_frame]

                part = torch.from_numpy(part).float().unsqueeze(0).cuda()
                body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
                body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)

                c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy() for i in body_support_1]
                c_nose = np.squeeze(c_mid)
                if case == 'ed':
                    v_ed = self.v_ed[frame-self.start_frame,:]
                    self.h_ed[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])
                if case == 'ed_reject':
                    v_ed = self.v_ed_reject[frame-self.start_frame,:]
                    self.h_ed_reject[0].set_xdata([c_nose[0],c_nose[0]+v_ed[0]])
                    self.h_ed_reject[0].set_ydata([c_nose[1],c_nose[1]+v_ed[1]])
                    self.h_ed_reject[0].set_3d_properties([c_nose[2],c_nose[2]+v_ed[2]])

                pass            
        
        def update_pc(frame):
            # get new raw data!
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            # update the pointcloud
            self.h_pc._offsets3d = (X,Y,Z)
            # and update the keypoints
            for body in range(4):
                self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])
            # update the counters
            update_frame_number(frame)

            update_frame_time(frame,time_offset=time_offset)
                            
            # update the fit
            update_skel_fit(frame)
            update_ellip_fit(frame)       
            update_trace_fit(frame,trace=trace,trace_length=10*60,trace_clip = 30*60-960 ,type_list = ['nose'])                

#             # update the fit as well!
            
#             if self.v_ed is not None:
#                 update_ed(frame,case='ed')
#                 update_ed(frame,case='ed_reject')
                
        # trick for updating line3d collection
        # https://mail.python.org/pipermail/matplotlib-users/2015-October/000066.html

        
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, frame_list[1:], interval=10)

        if savepath is None:
            fn = '<"(, ,)~~'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        if savepath[-1] == 'f':
            ani.save(savepath,writer='imagemagick',fps=fps,dpi = dpi)
        else:
            ani.save(savepath,writer='ffmpeg',fps=fps,dpi=dpi)
        plt.rcParams['animation.html'] = 'html5'
        ani

    def rotation(self,frame_list=30*60+np.arange(100)*100,cloud=True,trace='raw',skel='raw',ellip='raw',savepath = None,view_override = None,fps=10,dpi=100):
        
        
        self.trace_length = 10*60
        self.trace_clip = 30*60-960
        self.type_list = ['nose']
        
        self.make_3d_axis()
        self.add_raw_data(frame_list[0])
        # add the fitted values
        self.add_skel_fit(frame_list[0],fit=skel)
        self.add_ellip_fit(frame_list[0],fit=ellip)        
        self.add_trace(frame_list[0],trace=trace,trace_length=10*60,trace_clip = 30*60-960 ,type_list = ['nose'])
        
#         if self.v_ed is not None:
#             self.add_ed(frame_list[0],case='ed')
#             self.add_ed(frame_list[0],case='ed_reject')
                    
        self.finish_3d_axis(view_style='ex')
        # override the view!
        if view_override is not None:
            el, az = view_override
            self.ax.view_init(elev=el, azim=az)
        # tighten and add frame # and time
        plt.tight_layout()
        self.add_frame_number(frame_list[0])
        self.add_frame_time(frame_list[0],frame_list[0])
        
        # LOTS OF FUNCTIONS FOR ANIMATION
        def update_pc(i):
            # here, the update rule is simply to change the view
            offset = el*np.sin(i*2*np.pi / 360)
            plt.gca().view_init(elev=el+offset, azim=az+i)
            
#         for frame in frame_list[1:]:
        ani = animation.FuncAnimation(self.fig, update_pc, np.linspace(0,360,80), interval=10)

        fps = 10
        if savepath is None:
            fn = '<"(, ,)~~_rotation'
            if self.v_ed is not None:
                fn = '<"(, ,)~~__HEAD_rotation'
            savepath = fn+'.gif'
            # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        
        if savepath[-1] == 'f':
            ani.save(savepath,writer='imagemagick',fps=fps,dpi=dpi)
        else:
            ani.save(savepath,writer='ffmpeg',fps=fps,dpi=dpi)
        plt.rcParams['animation.html'] = 'html5'
        ani        
        
        
        
    def plot_residuals(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
        X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
        
        # and get the corresponding fit: the particle, convert to torch tensor, calculate body supports
        if self.track_or_guess == 'track':
            part = self.tracking_holder[:-1,frame-self.start_frame]
        if self.track_or_guess == 'guess':
            part = self.guessing_holder[:-1,frame-self.start_frame]
        
        part = torch.from_numpy(part).float().unsqueeze(0).cuda()
#         body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
#         body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        pos = torch.from_numpy(pos).float().cuda()
        
        # CALCULATE RESIDUAL
        dist0,_,self.body_support_0 = particles_to_distance_cuda(part[:,:9],pos[:,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(part[:,9:],pos[:,:])
        r = torch.min(dist0[0,:],dist1[0,:])
        r = r.cpu().numpy().squeeze()
        
        # Figure out the weighing stuff
        w2 = pos_weights**2

        rw=np.clip(r,0,.03)
        
        # or weigted mean?
#         rm = np.mean(rw)
        
        # correct
        rm = np.sum(w2*rw)/np.sum(w2)
        # wrong old way:
#         rm = np.mean(w2*rw)/np.median(w2)

        plt.figure()
        plt.plot(r,'.k',label='residual')
        plt.plot(rw,'.r',label='cut')
        plt.plot(w2,'.g',label='w2')
        plt.legend()
        plt.show()
        
#         plt.figure()
#         plt.hist(w2,100)
#         plt.show()
        
        plt.figure(figsize =(10,10))
#         plt.scatter(X,Y,c = 1-rw/np.max(rw))
        plt.scatter(X,Y,c = w2/np.max(w2))

        plt.axis('square')        
        plt.title("clipped: {}, weighted: {}".format(np.mean(rw),rm))
        plt.show()


        pass
# tips https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
# add raw data to plot

# add particle

    def kernel_smoothing(self,width = 3):
        kernel = easy_kernel(width)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')

            
                 
            
            
def easy_kernel(kernel_width = 30):
    from scipy import stats
    kernel = stats.norm.pdf(np.arange(-3*kernel_width,3*kernel_width+1),scale=kernel_width)
    kernel = kernel/np.sum(kernel)
    return kernel

class TrackingWrangler(object):
    def __init__(self,tracked_behavior,jagged_lines,what_to_plot = 'guess'):
        self.tracked_behavior = tracked_behavior
        self.jagged_lines = jagged_lines
        # useful functions, get the raw data in plottable format
        # pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[start_frame])
        # useful functions, get the fitted data in plottable format
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Say, "the default sans-serif font is COMIC SANS"
        matplotlib.rcParams['font.sans-serif'] = "Liberation Sans"
        # Then, "ALWAYS use sans-serif fonts"
        matplotlib.rcParams['font.family'] = "sans-serif"

        matplotlib.rc('font', family='sans-serif') 
        matplotlib.rc('text', usetex=False) 
        matplotlib.rcParams.update({'font.size': 13})

        # AHHH, Ok: https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Liberation Sans'
        matplotlib.rcParams['mathtext.it'] = 'Liberation Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Liberation Sans:bold'
        matplotlib.rcParams['mathtext.fallback_to_cm'] = False
        from palettable.cmocean.sequential import Algae_6
        cmpl = Algae_6.mpl_colors
        self.cmpl = cmpl
        # unpack 
        self.guessing_holder = tracked_behavior['guessing_holder']
        self.tracking_holder = tracked_behavior['tracking_holder']
        self.start_frame = tracked_behavior['start_frame']
        
        self.track_or_guess = what_to_plot
        self.n_frames = self.tracking_holder.shape[1]
        
        self.scale_bar = [1,1,.1,1,1,1,.01,.01,.01,1,1,.1,1,1,.1,.1,.1]
        self.units = ["rad","rad","a.u.",'rad','rad','rad',"m","m","m","rad","rad","a.u.",'rad','rad',"m","m","m"]
        self.latex_vars = ['β', 'γ', 's', 'ψ', 'θ', 'φ', 'x', 'y', 'z', 'β', 'γ', 's', 'θ', 'φ', 'x', 'y', 'z']


    def better_smoothing(self):
        # get the raw tracking data!
        part = self.tracking_holder

        # unpack all the 3D coordinates!
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        # and the spine length
        s_0 = part[:,2].cpu().numpy()
        s_1 = part[:,2+9].cpu().numpy()

        # and smooth the data
#         from utils.analysis_tools import smooth_body_support
        body_support_0_smooth,s_0_smooth = smooth_body_support(body_support_0,s_0)
        body_support_1_smooth,s_1_smooth = smooth_body_support(body_support_1,s_1)

        # add the smoothed coordinates as numpy arrays
        self.body_support_0_raw = [i.cpu().numpy().squeeze() for i in body_support_0]
        self.body_support_0_smooth = body_support_0_smooth
        self.s_0_raw = s_0
        self.s_0_smooth = s_0_smooth
        self.body_support_1_raw = [i.cpu().numpy().squeeze() for i in body_support_1]
        self.body_support_1_smooth = body_support_1_smooth
        self.s_1_raw = s_1
        self.s_1_smooth = s_1_smooth
        # also smooth the body ellipsoid rotations
        self.body_support_0_smooth = smooth_rotation_matrices(body_support_0_smooth)
        self.body_support_1_smooth = smooth_rotation_matrices(body_support_1_smooth)
        # self.body_support_0_smooth = smooth_rotation_matrices_quaternion(body_support_0_smooth)
        # self.body_support_1_smooth = smooth_rotation_matrices_quaternion(body_support_1_smooth)

        
    def kernel_smoothing(self):
        kernel = easy_kernel(3)
        for i in range(17):
            self.tracking_holder[i,:] = np.convolve(self.tracking_holder[i,:],kernel,'same')
            self.guessing_holder[i,:] = np.convolve(self.guessing_holder[i,:],kernel,'same')
    def kernel_smoothing_points(self):
        kernel = easy_kernel(3)
        for b_i in range(2):
            for i in range(5):
                dat = self.body_points[b_i][i]
                for j in range(3):
                    dat[:,j] = np.convolve(dat[:,j],kernel,'same')
                self.body_points[b_i][i] = dat

            
    def unpack_all_body_support(self):
        part = self.tracking_holder[:-1,:]
        part = torch.from_numpy(part).float().cuda()
        part = torch.transpose(part,0,1)
        body_support_0 = particles_to_body_supports_cuda(part[:,:9],implant = True)
        body_support_1 = particles_to_body_supports_cuda(part[:,9:],implant = False)
        self.all_support_0 = body_support_0
        self.all_support_1 = body_support_1

    def plot_all_tracking(self,do_wrapping = False,savepath=None):
        
        sc=2
        plt.figure(figsize = (18,8) )
        n_vars = len(self.tracked_behavior['var'])
        start_frame = 0
        frame_window = n_frames

        scale_bar = [1,1,.1,1,1,1,.01,.01,.01,1,1,.1,1,1,.1,.1,.1]
        units = ["rad","rad","a.u.",'rad','rad','rad',"m","m","m","rad","rad","a.u.",'rad','rad',"m","m","m"]

        for i in range(n_vars):
            plt.subplot(n_vars,1,1+i)


            dat0 = self.tracked_behavior['tracking_holder'][i,start_frame:(start_frame+frame_window)]
            dat1 = self.tracked_behavior['guessing_holder'][i,start_frame:(start_frame+frame_window)]

            # wrap:

            def wrap_angles(phases):
                phases = (phases + np.pi) % (2 * np.pi) - np.pi
                return phases
            if units[i] == 'rad' and do_wrapping:
                dat0 = wrap_angles(dat0)
                dat1 = wrap_angles(dat1)


            plt.plot(dat0,c=self.cmpl[1])
            plt.plot(dat1,c=self.cmpl[3])

            ax = plt.gca()
            if i < (n_vars-1):
                adjust_spines(ax,[])
            else:
                adjust_spines(ax,['bottom'])

        #         adjust_spines(ax,['bottom','left'])
            plt.yticks([])
            plt.ylabel(self.tracked_behavior['var'][i])
            plt.xlim([0,frame_window+10])

            ax = plt.gca()
            ylim=ax.get_ylim()
            xlim=ax.get_xlim()
            plt.plot( -2+np.array([1,1])*xlim[1],ylim[0]+np.array([0,1])*scale_bar[i],'-k' )
            plt.text(xlim[1],ylim[0]," "+str(scale_bar[i])+' '+units[i] )

            if i < 9:
                ax.set_facecolor('k')
                ax.patch.set_alpha(0.13)
            else:
                ax.set_facecolor('peru')      
                ax.patch.set_alpha(0.2)
            
            
        plt.xlabel('Frames')    
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)

        plt.tight_layout()
        # plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/tracked_full.pdf',transparent=True)  
        if savepath is not None:
            plt.savefig(savepath,transparent=False)  
            
        plt.show()       
        
    def plot_body_supports(self,savepath=None):

#         import matplotlib
#         matplotlib.rcParams['mathtext.fontset'] = 'custom'
#         matplotlib.rcParams['mathtext.rm'] = 'Liberation Sans'
#         matplotlib.rcParams['mathtext.it'] = 'Liberation Sans:italic'
#         matplotlib.rcParams['mathtext.bf'] = 'Liberation Sans:bold'       
#         matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
#         matplotlib.rcParams['font.family'] = 'sans'
#         matplotlib.rc('text', usetex=True)
#         matplotlib.rc('text', usetex=True)
#         from matplotlib import rc
#         rc('font',**{'family':'sans-serif','sans-serif':['Liberation Sans']})
        ## for Palatino and other serif fonts use:
        #rc('font',**{'family':'serif','serif':['Palatino']})
#         rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble'] = [
               r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
               r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
               r'\usepackage{helvet}',    # set the normal font here
               r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
               r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
        ]          

        self.unpack_all_body_support()
        subplot_counter = 0
        fig = plt.figure(figsize = (20,8) )
        start_frame = 0
        frame_window = self.n_frames
            
        n_subplots_total = 6 + 5 #(6 w implant, 5 w/o implant)
        for body_idx,body_support in enumerate([self.all_support_0,self.all_support_1]):
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy().squeeze() for i in body_support]
            type_list = np.array(['hip','tail','mid','nose','tip','impl'])
            c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
            if body_idx == 1:
                # don't have the implant for the last one
                c_list = c_list[:-1]
                               
            for body_part,c in enumerate(c_list):
                subplot_counter += 1
                plt.subplot(n_subplots_total,1,subplot_counter)
                for i in range(3):
                    plt.plot(c[:,i],c=self.cmpl[i+1])
                plt.ylabel("$c_{"+type_list[body_part]+'}$\n[m]')
                
                plt.xlim([0,frame_window+10])

                ax = plt.gca()
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_bounds(-.1,.1)
                plt.yticks([-.1,.1])

                if subplot_counter < (n_subplots_total):
                    pass
#                     adjust_spines(ax,['left'])
                    plt.gca().spines['bottom'].set_visible(False)
                    plt.xticks([])
                else:
                    pass
#                     adjust_spines(ax,['left','bottom'])

                if body_idx == 0:
                    ax.set_facecolor('k')
                    ax.patch.set_alpha(0.13)
                else:
                    ax.set_facecolor('peru')      
                    ax.patch.set_alpha(0.2)
                plt.xticks(fontname = "Liberation Sans")  # This argument will change the font.
                plt.yticks(fontname = "Liberation Sans")  # This argument will change the font.
                

        plt.xlabel('Frames',fontname = "Liberation Sans")    
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)

        # annoyoing hack!

        
#         plt.tight_layout()
        
#         matplotlib.rc('text', usetex=False)
        
        if savepath is not None:
            plt.savefig(savepath,transparent=False)      
            
            
    def unpack_body_points(self):
        self.unpack_all_body_support()
        self.body_points = []
        for body_idx,body_support in enumerate([self.all_support_0,self.all_support_1]):
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = [i.cpu().numpy().squeeze() for i in body_support]
            type_list = np.array(['hip','ass','mid','nose','tip','impl'])
            c_list = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
            self.body_points.append(c_list)
            self.type_list = np.array(['hip','ass','mid','nose','tip','impl'])
    
    def calculate_social_distances(self,event='n2n',hidden = False):
        def return_part(ttt = 'ass'):
            return int(np.nonzero(self.type_list == ttt)[0])
        self.v_nose2nose = self.body_points[0][ return_part('tip')  ] - self.body_points[1][ return_part('tip')  ]
        self.d_nose2nose = np.linalg.norm(self.v_nose2nose,axis=1)
        
        self.v_nose02ass = self.body_points[0][ return_part('tip')  ] - self.body_points[1][ return_part('ass')  ]
        self.d_nose02ass = np.linalg.norm(self.v_nose02ass,axis=1)
        
        self.v_nose12ass = self.body_points[1][ return_part('tip')  ] - self.body_points[0][ return_part('ass')  ]
        self.d_nose12ass = np.linalg.norm(self.v_nose12ass,axis=1)
        
        labels = ["nose_0 <-> nose_1","nose_0 --> ass_1","nose_1 --> ass_0"]
        d_list = [self.d_nose2nose,self.d_nose02ass,self.d_nose12ass]
        
        n_subplots = 5
        cutoff = .03
        cutoff2 = .005      
        
        plt.figure(figsize=(18,8))
        for i in range(3):
            plt.subplot(n_subplots,1,1+i)
            plt.plot(d_list[i])
            plt.axhline(cutoff,c='g')
            plt.axhline(cutoff2,c='r')

            plt.ylabel(labels[i]+" [m]")
#             plt.xlim([30*60,30*60+4000])
            plt.xlim(0,self.n_frames)
        
        
        if event == 'n2n':
            logic_raw = (self.d_nose2nose < .01) * (self.d_nose02ass > .06) * (self.d_nose12ass > .06) 
#             logic_raw = (self.d_nose2nose > .06) * (self.d_nose02ass > -1) * (self.d_nose12ass < .02) 
#             logic_raw = (self.d_nose2nose > .06) * (self.d_nose12ass > -1) * (self.d_nose02ass < .02) 

        from scipy import ndimage
        
        logic_close=ndimage.binary_closing(logic_raw,structure = np.ones(60))
        # frames is where it's true and the previous one is false
        logic_frames = np.argwhere(logic_close[:-1]*(~logic_close[1:]) ) + 1
        
        plt.subplot(n_subplots,1,4)
        plt.plot(logic_raw,c='r')
        plt.plot(logic_close,c='k')
        for fr in logic_frames:
            plt.axvline(fr,c='g')
        plt.xlim(0,self.n_frames)

        for i in range(n_subplots):
            plt.subplot(n_subplots,1,1+i)            
#             plt.xlim([30*60,30*60+4000])
#             plt.xlim([25000,30000])
    
        # save for analysis, the is an index, not the actual frames. 
        #TODO standardize this somehow?
        self.logic_frames = logic_frames.squeeze()  
        if hidden:
            plt.close()
        else:
            plt.show()
        
#         plt.ylim([0,0.05])
#         nose2ass =
#         nose2neck =
#         hip2hip =
#         pass
    def calculate_2d_running(self,plot=True,zoom=True,savepath=None):
        self.fwd_vel = [None,None]
        self.left_vel = [None,None]
        self.up_vel = [None,None]

        self.fwd = [None,None]
        self.left = [None,None] 
        self.up = [None,None] 
        guessing_holder = self.guessing_holder

        for body_idx in range(2):
            
            c_hip = self.body_points[body_idx][0]
            c_ass = self.body_points[body_idx][1]
            
            # todo, do this w/o splitting
            x_fit = c_hip[:,0]
            y_fit = c_hip[:,1]
            z_fit = c_hip[:,2]
                             
            dx_fit = np.hstack((np.diff(x_fit),0))
            dy_fit = np.hstack((np.diff(y_fit),0))
            dz_fit = np.hstack((np.diff(z_fit),0))

            # convolve with .5 s box
            kernel = np.ones(30)/30
            dx_fit = np.convolve(dx_fit,kernel,'same')
            dy_fit = np.convolve(dy_fit,kernel,'same')
            dz_fit = np.convolve(dz_fit,kernel,'same')

            # calculate the xy-speed
            v_xy = np.vstack((dx_fit,dy_fit)).T
            
            # calculate the vetor from the ass to the hip
            # as a kind of body-direction vector
            
            dir_body = c_hip-c_ass
            # take only 2D  xy-component and normalize the vector!
            dir_body_2d = dir_body[:,:2]/np.sqrt(dir_body[:,0]**2+dir_body[:,1]**2)[:,np.newaxis]
            # make a vector, 90deg rotation, so pointing LEFT!!!, rotate by hand
            dir_left_2d = np.einsum('ij,aj -> ai', np.array([[0,-1],[1,0]]),dir_body_2d )
            
            
            # use einsum for a faster dot product
            fwd_vel = np.einsum('ij,ij->i',v_xy,dir_body_2d)
            left_vel = np.einsum('ij,ij->i',v_xy,dir_left_2d)
            up_vel = dz_fit
            # smooth again
            kernel_length = 30
            kernel = np.ones(kernel_length )/kernel_length 
            # kernel = 1.
            fps = 60
            conversion_factor = fps #from m/frame to m/s
            
            
            fwd = np.convolve(fwd_vel,kernel,'same') * conversion_factor
            left = np.convolve(left_vel,kernel,'same') * conversion_factor
            up = np.convolve(up_vel,kernel,'same') * conversion_factor 
            
            # save the calculations
            self.fwd_vel[body_idx] = fwd_vel
            self.left_vel[body_idx] = left_vel
            self.up_vel[body_idx] = up_vel
            self.fwd[body_idx] = fwd
            self.left[body_idx] = left
            self.up[body_idx] = up
            
            if plot:
                self.plot_2d_running(body_idx=body_idx,zoom= zoom,savepath=savepath)

    def plot_2d_running(self,body_idx = 0,zoom = False,savepath=None):
        c_hip = self.body_points[body_idx][0]
        c_ass = self.body_points[body_idx][1]

        # todo, do this w/o splitting
        x_fit = c_hip[:,0]
        y_fit = c_hip[:,1]
        z_fit = c_hip[:,2]

        dx_fit = np.hstack((np.diff(x_fit),0))
        dy_fit = np.hstack((np.diff(y_fit),0))
        dz_fit = np.hstack((np.diff(z_fit),0))

        # convolve with .5 s box
        kernel = np.ones(30)/30
        dx_fit = np.convolve(dx_fit,kernel,'same')
        dy_fit = np.convolve(dy_fit,kernel,'same')
        dz_fit = np.convolve(dz_fit,kernel,'same')        
        
        axes = []
        plt.figure(figsize=(18,8))
        plt.subplot(4,1,1)
        axes.append(plt.gca())
        al =1
        plt.plot(x_fit,label='x',c=self.cmpl[1],alpha=al)
        plt.plot(y_fit,label='y',c=self.cmpl[2],alpha=al)
        plt.plot(z_fit,label='z',c=self.cmpl[3],alpha=al)
        plt.legend(loc = 'upper right')
        plt.ylabel('Position [m]')

        plt.subplot(4,1,2)
        axes.append(plt.gca())
        al =1
        plt.plot(dx_fit,label='dx',c=self.cmpl[1],alpha =al)
        plt.plot(dy_fit,label='dy',c=self.cmpl[2],alpha=al)
        plt.plot(dz_fit,label='dz',c=self.cmpl[3],alpha=al)
        plt.legend(loc = 'upper right')
        plt.ylabel('ΔPosition\n[m/s]')

        plt.subplot(4,1,3)
        axes.append(plt.gca())
        plt.plot(self.fwd[body_idx],label="fwd",c=self.cmpl[1])
        plt.axhline(0,c='k',ls='--')
#         plt.legend(loc = 'upper right')
        plt.ylabel('Forward speed\n[m/s]')
        
        plt.subplot(4,1,4)
        axes.append(plt.gca())
        plt.plot(self.left[body_idx],label="left",c=self.cmpl[1])
        plt.axhline(0,c='k',ls='--')
#         plt.legend(loc = 'upper right')

        plt.ylabel('Left speed\n[m/s]')

        # fix some stuff
        for i in [2,3]:
            #plt.subplot(4,1,1+i)
            ax = axes[i]

            ax.set_ylim(np.array([-1,1]) *60* 3e-3)

        
        if zoom:
            for i in range(4):
                #plt.subplot(4,1,1+i)
                ax = axes[i]
                ll = 10000
                offset = 0
                ax.set_xlim(np.array([0,ll])+offset)
        else:
            for i in range(4):
#                 plt.subplot(4,1,1+i)
#                 ax = plt.gca()
                ax = axes[i]
                ll = len(x_fit)
                ax.set_xlim(np.array([0,ll]))
                
                
        
        for i in range(4):
#             plt.subplot(4,1,1+i)
#             ax = plt.gca()
            ax = axes[i]
            if i<3:
                adjust_spines(ax,'left')
            else:
                adjust_spines(ax,['left','bottom'])
                plt.xlabel('Frame')

        if savepath is not None:
            plt.savefig(savepath[:-4]+'_'+str(body_idx)+'.pdf',transparent=False)   
                            
        plt.show()  

    def plot_spatial_running(self,body_idx = 0,savepath=None):
        c_hip = self.body_points[body_idx][0]

        # todo, do this w/o splitting
        x_fit = c_hip[:,0]
        y_fit = c_hip[:,1]
        z_fit = c_hip[:,2]
        
        # convolve with .5 s box
        kernel = np.ones(15)/15
        x_sm = np.convolve(x_fit,kernel,'same')
        y_sm = np.convolve(y_fit,kernel,'same')
        z_sm = np.convolve(z_fit,kernel,'same')
        
        plt.figure(figsize = (6,6) )
        for idx in np.array_split(np.arange(len(x_sm)), 1000):
            plt.plot(x_sm[idx],y_sm[idx],c= self.cmpl[2],alpha =.4)
        plt.axis('equal')
        plt.axis('off')

        if savepath is not None:
            plt.savefig(savepath[:-4]+'_'+str(body_idx)+'.pdf',transparent=True)   
                  
        
        plt.show()
        
        fig = plt.figure(figsize = (6,6))
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.gca(projection='3d')
        for idx in np.array_split(np.arange(len(x_sm)), 1000):
            plt.plot(x_sm[idx],y_sm[idx],z_sm[idx],c= self.cmpl[2],alpha =.4)
        

        

        ax.set_xlim(-.10,.20)
        ax.set_ylim(-.30,0.)
        ax.set_zlim(0,.3)
        plt.axis('off')
        
        if savepath is not None:
            plt.savefig(savepath[:-4]+'_'+str(body_idx)+'_3D.pdf',transparent=True)   
                  
        
        plt.show()

        
    def calculate_2d_running_slow(self,plot=True,zoom=True,savepath=None,hidden=False):
        # todo, so slow and stupid, easy to speed up
        guessing_holder = self.guessing_holder

        self.fwd_vel = [None,None]
        self.left_vel = [None,None]
        self.fwd = [None,None]
        self.left = [None,None]
                
        def rotate_body_model(alpha_body,beta_body,gamma_body):
            """
            TODO Very basic, could prob improve A LOOOOT
            Returns R_body, to rotate and transform the mouse body model
            alpha,beta,gamma is rotation around x,y,z axis respectively
            https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
            """
            R_alpha = np.empty((3,3))
            R_alpha[0,:] = [1.,0.,0.]
            R_alpha[1,:] = [0.,np.cos(alpha_body),-np.sin(alpha_body)]
            R_alpha[2,:] = [0.,np.sin(alpha_body),np.cos(alpha_body)]
            R_beta = np.empty((3,3))
            R_beta[0,:] = [np.cos(beta_body),0.,np.sin(beta_body)]
            R_beta[1,:] = [0.,1.,0.]
            R_beta[2,:] = [-np.sin(beta_body),0.,np.cos(beta_body)]
            R_gamma = np.empty((3,3))
            R_gamma[0,:] = [np.cos(gamma_body),-np.sin(gamma_body),0.]
            R_gamma[1,:] = [np.sin(gamma_body),np.cos(gamma_body),0.]
            R_gamma[2,:] = [0.,0.,1.]
            return R_alpha@R_beta@R_gamma
        
        for body_idx in range(2):
            if body_idx == 0:
                beta_fit = guessing_holder[0,:]
                gamma_fit = guessing_holder[1,:]
                x_fit = guessing_holder[6,:]
                y_fit = guessing_holder[7,:]
                z_fit = guessing_holder[8,:]
            elif body_idx == 1:
                beta_fit = guessing_holder[9,:]
                gamma_fit = guessing_holder[10,:]
                x_fit = guessing_holder[14,:]
                y_fit = guessing_holder[15,:]
                z_fit = guessing_holder[16,:]

            dx_fit = np.hstack((np.diff(x_fit),0))
            dy_fit = np.hstack((np.diff(y_fit),0))
            dz_fit = np.hstack((np.diff(z_fit),0))

            # convolve with .5 s box
            kernel = np.ones(30)/30
            dx_fit = np.convolve(dx_fit,kernel,'same')
            dy_fit = np.convolve(dy_fit,kernel,'same')
            dz_fit = np.convolve(dz_fit,kernel,'same')

            # calculate the xy-speed
            v_xy = np.vstack((dx_fit,dy_fit))

            fwd_vel = []
            left_vel = []

            
            #todo vecotrize this, already in the pytorch lib
            for i in tqdm( range(v_xy.shape[1]-2) ):
                v_step = v_xy[:,i]
                dir_body = rotate_body_model(0,beta_fit[i],gamma_fit[i])@[1.,0.,0.]
                dir_body = dir_body[:2]/np.sqrt(dir_body[0]**2+dir_body[1]**2)
                fwd_vel.append(np.dot(v_step,dir_body))
                # fwd_vel.append(v_step @ dir_body)

                # ALONG Y so actually LEFT
                dir_left = rotate_body_model(0,beta_fit[i],gamma_fit[i])@[0.,1.,0.]
                dir_left = dir_right[:2]/np.sqrt(dir_left[0]**2+dir_left[1]**2)
                left_vel.append(np.dot(v_step,dir_left))
                # right_vel.append(v_step @ dir_right)

            # stack the speeds
            fwd_vel = np.hstack(fwd_vel)
            left_vel = np.hstack(left_vel)
            
            
            #%%
            kernel = np.ones(30)/30
            # kernel = 1.
            fwd = np.convolve(fwd_vel,kernel,'same')
            left = np.convolve(left_vel,kernel,'same')

            # save the calculations
            self.fwd_vel[body_idx] = fwd_vel
            self.left_vel[body_idx] = left_vel
            self.fwd[body_idx] = fwd
            self.left[body_idx] = left
            
            
            
            if plot:
                plt.figure(figsize=(12,12))
                plt.subplot(6,1,1)
                plt.plot(x_fit,label='x')
                plt.plot(y_fit,label='y')
                plt.plot(z_fit,label='z')
                plt.legend()

                plt.subplot(6,1,2)
                plt.plot(dx_fit,label='dx')
                plt.plot(dy_fit,label='dy')
                plt.plot(dz_fit,label='dz')
                plt.legend()

                plt.subplot(6,1,3)
                plt.plot(fwd,label="fwd")
                plt.axhline(0)
                plt.legend()

                plt.subplot(6,1,4)
                plt.plot(left,label="left")
                plt.axhline(0)
                plt.legend()
                
                if zoom:
                    for i in range(4):
                        plt.subplot(6,1,1+i)
                        ll = 10000
                        offset = 0
                        plt.xlim(np.array([0,ll])+offset)
                        
                plt.show()

        
        
        pass
    
    def concatenate_runs_calc_four_states(self):
        pass
        
    def calculate_3d_running(self):
        pass
    
    def calculate_head_angle(self):
        # of the non-implated guy
        pass
    
    def calculate_nose2nose(self):
        pass
    
    def show_nose2nose_examples(self):
        pass
    
    def show_running_examples(self):
        pass
    
    def calculate_implant_distance(self,savepath=None):
        type_list = np.array(['hip','tail','mid','nose','tip','impl'])
        c_nose_0 = self.body_points[0][3]
        c_nose_1 = self.body_points[1][3]
        c_impl_0 = self.body_points[0][5]
        
        # get out the 3d positions of all the implant keypoints
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        n_frames = self.tracking_holder.shape[1]
        
        # unpack all the keypoints
        
        self.keyp_implant_holder = []
        self.keyp_frame_holder = []
        
        self.d_c_nose_0_holder = []
        self.d_c_nose_1_holder = []
        self.d_c_impl_0_holder = []
        
        for frame in tqdm(range(n_frames)):
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            
            # only if there are some keypoints
            if (np.sum(ikeyp == 0) > 0)*((frame-self.start_frame) > 0 ) :
                keyp_implant = keyp[ikeyp == 0]
                keyp_frame = frame * np.ones(np.sum(ikeyp == 0) )

                self.keyp_implant_holder.append(keyp_implant)
                self.keyp_frame_holder.append(keyp_frame)

                # gete the nose coordinate of the FIT, remember the offset
                d_c_nose_0 = np.linalg.norm( keyp_implant- c_nose_0[frame-self.start_frame,:], axis = 1 )
                d_c_nose_1 = np.linalg.norm( keyp_implant- c_nose_1[frame-self.start_frame,:], axis = 1 )
                d_c_impl_0 = np.linalg.norm( keyp_implant- c_impl_0[frame-self.start_frame,:], axis = 1 )

                self.d_c_nose_0_holder.append(d_c_nose_0)
                self.d_c_nose_1_holder.append(d_c_nose_1)
                self.d_c_impl_0_holder.append(d_c_impl_0)

        self.keyp_frame_holder = np.hstack(self.keyp_frame_holder)
        self.d_c_nose_0_holder = np.hstack(self.d_c_nose_0_holder)
        self.d_c_nose_1_holder = np.hstack(self.d_c_nose_1_holder)
        self.d_c_impl_0_holder = np.hstack(self.d_c_impl_0_holder)
        
        # get c_nose of both mice
        
        # plot the implant-to-nose distance
        self.plot_implant_distance(savepath=savepath)
        
        # plot the nose-to-nose distance 
    def plot_implant_distance(self,savepath=None):
        
        plt.figure(figsize = (18,8) )

        n_subplots = 4

        plt.subplot(n_subplots,1,1)
        c_nose_0 = self.body_points[0][3]
        c_nose_1 = self.body_points[1][3]
        head_distance = np.linalg.norm(c_nose_0 -  c_nose_1,axis =1)

        plt.plot(head_distance,c = self.cmpl[2])
        plt.ylabel("Head-to-head \n distamce [m]")
        plt.xlim([0,c_nose_0.shape[0]])

        plt.subplot(n_subplots,1,2)
        a = 1
        mz = .3
        cc = 'DodgerBlue'
#         cc = self.cmpl[2]
        plt.plot(self.keyp_frame_holder,self.d_c_nose_0_holder,'.',c = cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse0\n $c_{nose}$-to-keyp$_{impl}$ \n distamce [m]")
        plt.ylim([0,.3])

        plt.subplot(n_subplots,1,3)
        plt.plot(self.keyp_frame_holder,self.d_c_nose_1_holder,'.',c =cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse1\n $c_{nose}$-to-keyp$_{impl}$ \n distamce [m]")
        plt.ylim([0,.3])

        plt.subplot(n_subplots,1,4)
        plt.plot(self.keyp_frame_holder,self.d_c_impl_0_holder,'.',c = cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse1\n $c_{impl}$-to-keyp$_{impl}$ \n distamce [m]")
        plt.ylim([0,.3])
        plt.xlabel('Frame')
        
        for subplot_counter in range(4):
            plt.subplot(n_subplots,1,1+subplot_counter)
            ax = plt.gca()
            if subplot_counter < (n_subplots-1):
                adjust_spines(ax,['left'])
            else:
                adjust_spines(ax,['left','bottom'])

        if savepath is not None:
            plt.savefig(savepath,transparent=False)   

        plt.show()    

        
    def calculate_ear_distance(self,savepath=None,zoom=False):
        type_list = np.array(['hip','tail','mid','nose','tip','impl'])
        c_nose_0 = self.body_points[0][3]
        c_nose_1 = self.body_points[1][3]

        c_tip_0 = self.body_points[0][4]
        c_tip_0 = self.body_points[0][4]
        c_tip_1 = self.body_points[1][4]

        # the direction of the nose
        v_nose_0 = c_tip_0-c_nose_0
        v_nose_1 = c_tip_1-c_nose_1

        
        # get out the 3d positions of all the implant keypoints
        body_colors = ['dodgerblue','red','lime','orange']
        body_indices = [0,1,2,3]
        n_frames = self.tracking_holder.shape[1]
        
        # unpack all the keypoints
        
        self.keyp_ear_holder = []
        self.keyp_ear_rotated_holder = []

        self.keyp_frame_holder = []
        
        self.d_c_ear_0_holder = []
        self.d_c_ear_1_holder = []
        
        self.angle_c_ear_0_holder = []
        self.angle_c_ear_1_holder = []
        
        self.rot_holder = []
        
        for frame in tqdm(range(n_frames)):
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(self.jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            
            # only if there are some keypoints
            if (np.sum(ikeyp == 1) > 0)*((frame-self.start_frame) > 0 ) :
                keyp_ear = keyp[ikeyp == 1]
                keyp_frame = frame * np.ones(np.sum(ikeyp == 1) )

                self.keyp_ear_holder.append(keyp_ear)
                self.keyp_frame_holder.append(keyp_frame)

                # gete the nose coordinate of the FIT, remember the offset
                d_c_nose_0 = np.linalg.norm( keyp_ear - c_nose_0[frame-self.start_frame,:], axis = 1 )
                d_c_nose_1 = np.linalg.norm( keyp_ear - c_nose_1[frame-self.start_frame,:], axis = 1 )

                self.d_c_ear_0_holder.append(d_c_nose_0)
                self.d_c_ear_1_holder.append(d_c_nose_1)
                
                # also calculate the angle with the nose direction vector
                # the vecors from the nose to the ear keypoints
                v_nose2ears_0 = keyp_ear - c_nose_0[frame-self.start_frame,:]
                v_nose2ears_1 = keyp_ear - c_nose_1[frame-self.start_frame,:]
                
                # now get the projection onto a plane which has the nose vector as the normal
                plane_point = c_nose_1[frame-self.start_frame,:]
                plane_normal = v_nose_1[frame-self.start_frame,:]
                v_up = np.array([0,0,1]) # pointing to z
                
                # now, we rotate 
                def rotate_f_onto_t(f,t):
                    # make unit vectors
                    f = f/np.linalg.norm(f)
                    t = t/np.linalg.norm(t)
                    
                    # trick from here: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
                    # from this paper: http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf
                    # to rotate vector f onto vector t (around f x t!), we can do:
                    v = np.cross(f, t)
                    u = v/np.linalg.norm(v)
                    c = np.dot(f, t)
                    h = (1 - c)/(1 - c**2)

                    vx, vy, vz = v
                    rot = np.array([[c + h*vx**2, h*vx*vy - vz, h*vx*vz + vy],
                          [h*vx*vy+vz, c+h*vy**2, h*vy*vz-vx],
                          [h*vx*vz - vy, h*vy*vz + vx, c+h*vz**2]])
                    return rot
                # now, find a vector, which would rotate the nose direction vector 
                # into the xy plane. 
                rotA = rotate_f_onto_t(plane_normal,plane_normal*np.array([1,1,0]))
                # and now, we rotate in the xy plane to point in x-direction
                rotB = rotate_f_onto_t(plane_normal*np.array([1,1,0]),np.array([1,0,0]))
                # Now, we also rotate the keypoints by the same vector 
                rot = rotB @ rotA 
                
                # let's rotate the ear points into the same space
                keyp_ear_rotated = np.einsum('ij,ai->aj',rot,keyp_ear)
                self.keyp_ear_rotated_holder.append(keyp_ear_rotated)
                
                # and save their coordinates
                self.rot_holder.append(rot)
                
        # stack and concatenate, actually ear, so rename??
        self.keyp_ear_holder = np.concatenate(self.keyp_ear_holder,axis=0)
        self.keyp_ear_rotated_holder = np.concatenate(self.keyp_ear_rotated_holder,axis=0)

        self.keyp_frame_holder = np.hstack(self.keyp_frame_holder)
        self.d_c_ear_0_holder = np.hstack(self.d_c_ear_0_holder)
        self.d_c_ear_1_holder = np.hstack(self.d_c_ear_1_holder)
        print('pik')
        
                
        # get c_nose of both mice
        
        # plot the implant-to-nose distance
        self.plot_ear_distance(savepath=savepath,zoom=zoom)        

    def plot_ear_distance(self,savepath=None,zoom=False,xlim = None):
        # wrangling the ear keypoints, 2nd version
        plt.figure(figsize = (14,16) )

        n_subplots = 8

        plt.subplot(n_subplots,1,1)
        c_nose_0 = self.body_points[0][3]
        c_nose_1 = self.body_points[1][3]

        c_mid_0 = self.body_points[0][2]
        c_mid_1 = self.body_points[1][2]
     
        c_tip_0 = self.body_points[0][4]
        c_tip_1 = self.body_points[1][4]

        # the direction of the nose
        v_nose_0 = c_tip_0-c_nose_0
        v_nose_1 = c_tip_1-c_nose_1

        # the vecors from the nose to the ear keypoints

        cutoff = .025
        cutoff = .03
        close_to_0 = (self.d_c_ear_0_holder < cutoff )*(self.d_c_ear_1_holder > cutoff )
        close_to_1 = (self.d_c_ear_1_holder < cutoff )*(self.d_c_ear_0_holder > cutoff )

        # INSTEAD,just assign to the closest one?
        cutoff = .03
        close_to_0 = (self.d_c_ear_0_holder < self.d_c_ear_1_holder) * (self.d_c_ear_0_holder < cutoff )
        close_to_1 = (~close_to_0) * (self.d_c_ear_1_holder < cutoff )

        plt.subplot(n_subplots,1,1)
        a = 1
        mz = .3
        cc = 'Red'
        #         cc = self.cmpl[2]
        plt.plot(self.keyp_frame_holder,self.d_c_ear_0_holder,'.',c = cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse0\n $c_{nose}$-to-keyp$_{ear}$ \n distamce [m]")
        plt.ylim([0,.3])
        # plt.ylim([0,.025])

        plt.subplot(n_subplots,1,2)
        plt.plot(self.keyp_frame_holder,self.d_c_ear_1_holder,'.',c =cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse1\n $c_{nose}$-to-keyp$_{ear}$ \n distamce [m]")
        plt.ylim([0,.3])
        # plt.ylim([0,.025])

        plt.subplot(n_subplots,1,3)
        #         cc = self.cmpl[2]
        plt.plot(self.keyp_frame_holder[close_to_0],self.d_c_ear_0_holder[close_to_0],'.',c = cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse0\n $c_{nose}$-to-keyp$_{ear}$ \n distamce [m]")
        plt.ylim([0,.3])
        # plt.ylim([0,.025])

        plt.subplot(n_subplots,1,4)
        plt.plot(self.keyp_frame_holder[close_to_1],self.d_c_ear_1_holder[close_to_1],'.',c =cc,alpha = a,markersize = mz)
        plt.xlim([0,c_nose_0.shape[0]])
        plt.ylabel("Mouse1\n $c_{nose}$-to-keyp$_{ear}$ \n distamce [m]")
        plt.ylim([0,.3])
        # plt.ylim([0,.1])

        plt.subplot(n_subplots,1,5)
        # plot the y
        # smoothing, dirty fast version
        def easy_kernel(kernel_width = 30):
            from scipy import stats
            kernel = stats.norm.pdf(np.arange(-3*kernel_width,3*kernel_width+1),scale=kernel_width)
            kernel = kernel/np.sum(kernel)
            return kernel


        dat = self.keyp_ear_holder

        ears_x = dat[:,0]
        ears_y = dat[:,1]
        ears_z = dat[:,2]

        kernel = easy_kernel(1)
        ears_x_smooth = np.convolve(ears_x[close_to_1],kernel,'same')
        ears_y_smooth = np.convolve(ears_y[close_to_1],kernel,'same')
        ears_z_smooth = np.convolve(ears_z[close_to_1],kernel,'same')

        plt.plot(self.keyp_frame_holder[close_to_1],ears_x[close_to_1],'.',c =self.cmpl[1],alpha = a,markersize = 6*mz,label='y')
        plt.plot(self.keyp_frame_holder[close_to_1],ears_y[close_to_1],'.',c =self.cmpl[2],alpha = a,markersize = 6*mz,label='y')
        plt.plot(self.keyp_frame_holder[close_to_1],ears_z[close_to_1],'.',c =self.cmpl[3],alpha = a,markersize = 6*mz,label='z')

        # plt.plot(self.keyp_frame_holder[close_to_1],ears_x_smooth,'.',c =cmpl[-1],alpha = a,markersize = mz,label='y smooth')
        # plt.plot(self.keyp_frame_holder[close_to_1],ears_y_smooth,'.',c =cmpl[-1],alpha = a,markersize = mz,label='z smooth')
        # plt.plot(self.keyp_frame_holder[close_to_1],ears_z_smooth,'.',c =cmpl[-1],alpha = a,markersize = mz,label='z smooth')

        # plt.legend()
        plt.xlim([0,c_nose_0.shape[0]])

        plt.ylabel("Mouse1 Ear\ncoordinates [m]")

        frames_tracked = np.arange(self.start_frame,self.start_frame+self.n_frames)

        
        plt.subplot(n_subplots,1,7)


        c_nose_1_smooth = np.zeros((self.n_frames,3))
        for i in range(3):
            plt.plot(frames_tracked,c_nose_1[:,i],'.',c =self.cmpl[1+i],markersize=10*mz)
            kernel = easy_kernel(3)
            c_nose_1_smooth[:,i] = np.convolve(c_nose_1[:,i],kernel,'same')
            plt.plot(frames_tracked, c_nose_1_smooth[:,i] ,'-k')


        c_tip_1_smooth = np.zeros((self.n_frames,3))
        for i in range(3):
            kernel = easy_kernel(3)
            c_tip_1_smooth[:,i] = np.convolve(c_tip_1[:,i],kernel,'same')

        c_mid_1_smooth = np.zeros((self.n_frames,3))
        for i in range(3):
            kernel = easy_kernel(3)
            c_mid_1_smooth[:,i] = np.convolve(c_mid_1[:,i],kernel,'same')

        # use pandas to group the vectors!


        df = pd.DataFrame(np.vstack([self.keyp_frame_holder[close_to_1], ears_x_smooth,ears_y_smooth,ears_z_smooth]).T, columns = ['frame', 'x','y','z']) 

        grouped_data = df.groupby(['frame']).mean()
        # these are all the frames, where we have an estimate
        grouped_frame = np.array(grouped_data.index)

        # now, interpolate
        from scipy import interpolate

        c_ears = np.zeros((self.n_frames,3))

        for i,name in enumerate(['x','y','z']):
            f_interp = interpolate.interp1d(grouped_frame, grouped_data[name],bounds_error = False,fill_value=0,assume_sorted=False)
            print(name)
            c_ears[:,i] = f_interp(frames_tracked)    

        plt.ylabel("Mouse1 Nose\ncoordinates [m]")

        c_ears_smooth = np.zeros((self.n_frames,3))
        for i in range(3):
            kernel = easy_kernel(3)
            c_ears_smooth[:,i] = np.convolve(c_ears[:,i],kernel,'same')

        
        
        # ear direction vector!
        v_ed = c_ears-c_mid_1_smooth
        v_ed_smooth = np.zeros((self.n_frames,3))
        for i in range(3):
            kernel = easy_kernel(10)
            v_ed_smooth[:,i] = np.convolve(v_ed[:,i],kernel,'same')
        v_ed = v_ed/np.linalg.norm(v_ed,axis =1)[:,np.newaxis]
        v_ed_smooth = v_ed_smooth/np.linalg.norm(v_ed_smooth,axis =1)[:,np.newaxis]
        
        # also calculate the orthogonal vector! aka vector rejection
        v_nose_1 = c_tip_1_smooth-c_nose_1_smooth
        v_nose_norm = v_nose_1/np.linalg.norm(v_nose_1,axis =1)[:,np.newaxis]

        # projection is row wise
        dotp = np.einsum('ai,ai->a',v_ed,v_nose_norm)
        v_project = dotp[:,np.newaxis]*v_nose_norm
        v_reject = v_ed - v_project

        # ear direction
        v_ed_reject = v_reject/np.linalg.norm(v_reject,axis =1)[:,np.newaxis]
        v_ed_reject_smooth = np.zeros((self.n_frames,3))
        for i in range(3):
            kernel = easy_kernel(10)
            v_ed_reject_smooth[:,i] = np.convolve(v_ed_reject[:,i],kernel,'same')


        plt.subplot(n_subplots,1,8)
        for i in [2]:#range(3):
            
            aa = .05
            if zoom:
                aa = 1.
            plt.plot(frames_tracked,v_ed[:,i],'.',c='r',alpha = .1)
            plt.plot(frames_tracked,v_ed_smooth[:,i],'-',c='r',alpha = 1.)
            plt.plot(frames_tracked,v_ed_reject[:,i],'.',c='blueviolet',alpha =.1)
            plt.plot(frames_tracked,v_ed_reject_smooth[:,i],'-',c='blueviolet',alpha =1.)

        plt.ylim([-1,1])
        plt.ylabel("Mouse1 Ear\ndirection vector\nz-component")
        plt.xlabel("Frame")

        plt.subplot(n_subplots,1,6)
        for i in range(3):
            plt.plot(frames_tracked,c_ears[:,i],'-',c=self.cmpl[1+i])

        plt.ylabel("Mouse1 Ear\ncoordinates\nsmooth & interp [m]")

        
        for subplot_counter in range(n_subplots):
            plt.subplot(n_subplots,1,1+subplot_counter)
            ax = plt.gca()
            if subplot_counter < (n_subplots-1):
                adjust_spines(ax,['left'])
            else:
                adjust_spines(ax,['left','bottom'])
        #     plt.xlim(np.array([3000,4000])+1000+2000)

        #     # good example!
        #     plt.xlim(np.array([3000,4000])+1000+2000)

            # good example zoomed
            if zoom:
                plt.xlim(np.array([6400,7000+400]))
            else:
                plt.xlim([frames_tracked[0],frames_tracked[-1]])
                plt.xlim([0,len(frames_tracked)])
#                 plt.xlim([frames_tracked[0],frames_tracked[-1]])
            if xlim is not None:
                plt.xlim(xlim)

        # savepath = 'figure_raw_pics/figure_5_S/Ear-vector-calculation.png'
        if savepath is not None:
            plt.savefig(savepath,transparent=False)   

        plt.show()          

        # add to the plotter
        self.v_ed = .06*v_ed
        self.v_ed_reject = .06*v_ed_reject        

        # add to the plotter
        self.v_ed_smooth = .06*v_ed_smooth
        self.v_ed_reject_smooth = .06*v_ed_reject_smooth      
        
        # add to self for use later !
        self.c_ears_1 = c_ears
        self.c_ears_smooth_1 = c_ears_smooth

    def make_figure5(self):
        # plot XYZ for both mice
        
        # plot the running speed for both mice
        
        # plot the z speed for both mice
        
        # Fit arma model for running and rearing
        
        # calculate the social distances
        
        # Plot the 'Ethogram' for Running
        
        # Plot the 'Ethogram' for Z-running (rearing)
        
        # Plot the social ethogram (n2n,n2ass,n2ass)
        # follwing is nose-ass, while running, for a while, for example. Mounting would be center above centers
        # Select a few good examples for identified behaviors
        
        pass

    

from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    # from here: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def plot_t_ellipse(nu, cov, pos, fraction = .5, ax=None, **kwargs):
    # from here: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    nstd = 1
    # This gives the width and height, of 1 standard deviation
    # of a Gaussian, with the same covariance 
    width, height = 2 * nstd * np.sqrt(vals)
    
    # 
    
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


from matplotlib.colors import LinearSegmentedColormap

def gradient_cmap(colors, nsteps=256, bounds=None):
    # from https://github.com/slinderman/ssm/
    # Make a colormap that interpolates between a set of colors
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = np.linspace(0,1,ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b,c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap

def plot_2dhist_all(y_data):
    # Plot the raw data (whitened!
    import matplotlib


    #profile with a lower number of states
    # y_data = y_data[80000:85000,:]

    bins = (np.linspace(-4,6,10*4),np.linspace(-10,10,6*4))
    plt.figure(figsize = (4,2.7))
    plt.hist2d(y_data[:,0],y_data[:,1],bins = bins,cmap = 'Greys',norm=matplotlib.colors.LogNorm())
    plt.xlim([-4,6])
    plt.ylim([-10,10])
    # plt.title('all data')
    plt.xlabel('Fwd speed [z]')
    plt.ylabel('Left speed [z]')
    plt.tight_layout()

    ax = plt.gca()
#     adjust_spines2(ax,['Bottom','Left'])
    ax.set_yticks([-8,0,8])
    ax.set_xticks([-4,0,6])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_bounds(-8,8)

    plt.savefig('figs/all_histo.png',transparent = True,dpi=600)

    plt.show()
    #profile with a lower number of states


def fit_kmeans(data,hidden_dim,colors,show_plots = True):
    data_dim = data.shape[1]
    np.random.seed(1987)
    # for initializing, we use k-means, should we use the most likely state as first?
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=hidden_dim, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(data.numpy())
    mu_prior = torch.tensor(kmeans.cluster_centers_,dtype=torch.float).squeeze()
    std_prior = torch.zeros(hidden_dim,data_dim)
    for i in range(hidden_dim):
        for j in range(data_dim):
            std_prior[i,j] = torch.std(data[pred_y == i,j])
    print("prior mu: {} and std: {}".format(mu_prior,std_prior))

    


    if show_plots:
        plt.figure(figsize = (4,2.7))
    
        for i in range(hidden_dim):

            dat = data[pred_y==i]
            plt.scatter(dat[::25,0].numpy(),dat[::25,1].numpy(),
                        c=colors[i],s=3,alpha = .3)

    #         plt.plot(mu_prior[i,0],mu_prior[i,1],'o', c='k')
            plt.plot(mu_prior[i,0] * np.ones(2),mu_prior[i,1]+np.array([-1,1])*std_prior[i,1].numpy(),'-', c='k')
            plt.plot(mu_prior[i,0]+np.array([-1,1])*std_prior[i,0].numpy(),mu_prior[i,1]*np.ones(2),'-', c='k')
            plt.xlabel('Fwd speed [z]')
            plt.ylabel('Left speed [z]')
            plt.tight_layout()
            ax = plt.gca()
#             adjust_spines(ax,['Bottom','Left'])
            plt.xlim([-4,6])
            plt.ylim([-10,10])
            ax.set_yticks([-8,0,8])
            ax.set_xticks([-4,0,6])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_bounds(-8,8)
        plt.savefig( 'figs/kmeans.png',transparent = True,dpi=600)
        plt.show()
    return mu_prior,std_prior

def plot_latent_ellisoids(mus,Sigmas,data_np,colors,savepath = None):
    plt.figure(figsize=(4,2.7))
    ax = plt.gca()
    plt.hist2d(data_np[:,0],data_np[:,1],40, cmap='Greys',norm=matplotlib.colors.LogNorm())
    plt.xlabel('Fwd speed [z]')
    plt.ylabel('Left speed [z]')
    hidden_dim = mus.shape[0]
    for state in range(hidden_dim):
        mean = mus[state]
        cov = Sigmas[state]

        plt.plot(mean[0],mean[1],'o',c=colors[state])
        for std in range(4):
            ellip = plot_cov_ellipse(cov, mean, nstd=std, ax=ax,lw=2)
            ellip.set_edgecolor(colors[state])    
            ellip.set_facecolor('None')  

    plt.xlim([-4,6])
    plt.ylim([-10,10])
    ax.set_yticks([-8,0,8])
    ax.set_xticks([-4,0,6])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_bounds(-8,8)            
            
    if savepath is not None:
        plt.tight_layout()
        ax = plt.gca()
#         adjust_spines(ax,['Bottom','Left'])
        plt.xlim([-4,6])
        plt.ylim([-10,10])
        plt.savefig(savepath,transparent=True,dpi=600)
    plt.show()

def plot_map_estimate(map_esti_holder,colors,data_np,i=-1,savepath=None):
    # unpack
    transition,locs,scales,lkj = map_esti_holder[i]
    hidden_dim,data_dim = locs.shape
    
    # Calculate the covariance
    L_om = np.zeros([hidden_dim,data_dim,data_dim])
    Sigma = np.zeros([hidden_dim,data_dim,data_dim])
#     print(scales)
    # print(lkj)
    for k in range(hidden_dim):
        # make a matrix of the scales
        s_matrix = np.diag( np.sqrt(scales[k,...]) )
        lkj_matrix=lkj[k,...]
        L_om[k,...] = s_matrix @ (lkj_matrix @ s_matrix)
        # mirror the lower to the upper!
        X = L_om[k,...]
        Sigma[k,...] = X + X.T - np.diag(np.diag(X))    
    
    plot_latent_ellisoids(locs,Sigma,data_np,colors,savepath=savepath) 
    
    
from scipy.stats import norm
def plot_map_estimate_z(map_esti_holder,colors,data_np,i=-1,savepath=None):
    # unpack
    transition,locs,scales = map_esti_holder[i]
    hidden_dim = locs.shape[0]  
    plt.figure(figsize=(3.5,2.4))
#     plt.hist(data_np,100,color = 'lightgrey')
    
    color_offset = 5
    
    for state in range(hidden_dim):
        xxx= np.linspace(-6,6,1000)
        yyy = norm.pdf(xxx,loc = locs[state],scale = scales[state])
        plt.fill_between(xxx,yyy,color = colors[state+color_offset],alpha = .5)
        
    for state in range(hidden_dim):
        xxx= np.linspace(-6,6,1000)
        yyy = norm.pdf(xxx,loc = locs[state],scale = scales[state])
        plt.plot(xxx,yyy,color = colors[state+color_offset],lw = 2)        
        
    plt.xlim([-6,6])
    plt.xlabel('Up speed [z]')
    plt.yticks([])
    ax = plt.gca()
#     adjust_spines(ax,['Bottom','Left'])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_bounds(-5,5)
    plt.gca().spines['left'].set_bounds(-0,1.4)

    plt.ylabel('Density')
    plt.yticks([0,1.4])
    plt.xticks([-5,0,5])
    plt.gca().set_yticklabels(' ')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,transparent=True,dpi=600)
    plt.show()
    
def plot_losses(losses,savepath = None,z=False):
    if z:
        plt.figure(figsize=(3,2))
    else:
        plt.figure(figsize=(4,2.3))
    
    plt.plot(np.log(losses),':ok',markersize = 4)
    plt.xlabel('Iterations')
    plt.ylabel('log(Loss)')
    # pyplot.yscale('log')
    ax = plt.gca()
    # adjust_spines(ax,['Bottom','Left'])

    buf = 10
    ax.set_xlim([-buf,len(losses)+buf])
    plt.xticks([0,100])
    # ax.set_xticklabels([])

    ybuf=20
    plt.ylim(np.log([losses[-1]-ybuf,losses[0]+ybuf]))
    plt.yticks(np.log([losses[-1]-ybuf,losses[0]+ybuf]))
    ax.set_yticklabels([])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_bounds(-8,8)

    plt.gca().spines['bottom'].set_bounds(0,len(losses))

    ax.xaxis.labelpad = -10
    
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,
                transparent = True,dpi=600)
    plt.show()
    