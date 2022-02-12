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

        
        
# %debug
# MAKE a function that takes a Frame of raw data and plots a particle
# Maybe make this an object, even though I hate those...

# from utils.cuda_tracking_utils_for_figures import *
import matplotlib.animation as animation

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

    # the nose-pointing vector, make more
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    # a unit vector pointing to the nose
    nose_pointer = torch.stack([cos_theta,   sin_theta*cos_phi,     sin_theta*sin_phi], dim=1)
    # a unit vector along the x-axis
    x_pointer = torch.stack([one, zero,zero], dim=1)
    # use the nose-pointing vector to calculate the nose rotation matrix
    R_head = rotation_matrix_vec2vec(x_pointer,nose_pointer)
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
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[frame])
        
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
        self.track_or_guess = 'guess'
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
            pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[frame])
            X, Y, Z = pos[:,0],pos[:,1],pos[:,2]
            # update 
            self.h_pc._offsets3d = (X,Y,Z)
            
            # THEN set the 3d values to be what the shoud be
            for body in range(4):
                self.h_kp_list[body]._offsets3d = (keyp[ikeyp==body,0], keyp[ikeyp==body,1], keyp[ikeyp==body,2])

            # update the fit as well!
            # get the particle, convert to torch tensor, calculate body supports
            self.track_or_guess = 'guess'

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

    def plot_residuals(self,frame):
        # unpack the raw data in a plottable format
        pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[frame])
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
