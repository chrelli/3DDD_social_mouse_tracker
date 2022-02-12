
from utils.fitting_utils import *


import time, os, sys, shutil

# for math and plotting
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#%matplotlib notebook
#%matplotlib widget

from itertools import compress # for list selection with logical
from tqdm import tqdm

from multiprocessing import Process

# ALLSO JIT STUFF
from numba import jit, njit

# and pytorch
import torch



#TODO make the voxel etc load automatically!!<<
def read_processed_frame(top_folder,frame,p_time_subset = 0.1,voxel = 0.001, n_padding_digits = 8):
    #TODO make the voxel etc load automatically!!<<
    # should also be commmon among several
    raw = np.load(top_folder+'/npy_extrafine/frame_'+str(frame).rjust(n_padding_digits,'0')+'.npy')
    #todo make column order and split

    # get a random subset from the loaded frame
    subset_filter = np.random.uniform(size = raw.shape[0]) <= p_time_subset

    positions = raw[subset_filter,0:3]*voxel
    weights = raw[subset_filter,3]

    # Do a fractional weight filtering
    # weight_filter = weights >= np.percentile(weights,self.weight_percentile)
    # positions,weights = positions[weight_filter,:],weights[weight_filter]

    # do the position filtering down step! #todo can be moved to pre-processing
    # positions,weights = clean_positions_by_weights(positions,weights,cutoff = 2)


    return positions,weights




def color3d(positions,colors=None):
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    #   3D plot of the
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    correctX,correctY,correctZ = positions[:,0],positions[:,1],positions[:,2]

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        if len(colors.shape)> 1:
            ax.scatter(correctX, correctY, correctZ, zdir='z', s=1, c=colors/255.,rasterized=True)
        else:
            ax.scatter(correctX, correctY, correctZ, zdir='z', s=1, c=colors/np.max(colors),rasterized=True)
    else:
        ax.scatter(correctX, correctY, correctZ, zdir='z', s=1, c='b',rasterized=True)
    ax.set_aspect('equal')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35,35)
    #ax.set_zlim3d(-70,0)
    ax.set_xlabel('$x$ (mm)',fontsize=16)
    ax.set_ylabel('\n$y$ (mm)',fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
    ax.set_title(str(positions.shape[0])+' points',fontsize=16)

    X,Y,Z = correctX,correctY,correctZ
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()
    w,h = 570,800
    #plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)


def click_one_mouse(positions):

    ###############
    # Show a 2D plot and ask for two clicks
    ###############
    plt.figure()
    plt.scatter(positions[:,0],positions[:,1],c=positions[:,2]/np.max(positions[:,2]),s=5)
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('click center of hip, then mid, then head of mouse!')
    w,h = 570,800
    # plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)

    click_points = np.asanyarray(plt.ginput(3))
    hip_click = click_points[0]
    mid_click = click_points[1]
    nose_click = click_points[2]

    # plt.show()

    ###############
    # Now calculate a reference direction
    ###############
    v_click = nose_click-hip_click
    # and add to the plot
    def add_vec_from_point(c_mid_est,v_ref_est):
        data = np.vstack((c_mid_est,c_mid_est+v_ref_est))
        plt.plot(data[:,0],data[:,1],c='red')
        plt.plot(data[0,0],data[0,1],c='red',marker='o')


    plt.figure()
    plt.scatter(positions[:,0],positions[:,1],c=positions[:,2]/np.max(positions[:,2]),s=5)
    ax = plt.gca
    plt.axes().set_aspect('equal', 'datalim')
    add_vec_from_point(hip_click,v_click)
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('estimated hip and heading direction')
    w,h = 570,800
    # plt.get_current_fig_manager().window.setGeometry(1920-w-10,60,w,h)

    plt.show()

    return hip_click,mid_click,nose_click



#%% make a handy function to click the first frame_
def good_guess(hip_click,mid_click,nose_click):
    # translation vector, which moves the center of the mouse body
    z_guess = 0.022 # guess the z as around 2 cm
    t_body = np.append(hip_click,z_guess)

    # set the scaling
    s = 0.8

    # guess the rotation as the body model:
    # - alpha is around x axis, none, I guess
    alpha = 0

    # - beta is around y axis, so elevation
    beta = 0
    # - gamma is around z, so in the xy plane, the left-right angle of the mouse wrt. e^hat_x
        # get the vector
    v_click = mid_click-hip_click
    # make it a 3d vector to use with the handy function
    target = np.append(v_click,0)
    angle_with_x = angle_between(np.array([1.,0,0]),target)
    gamma = angle_with_x

    # same with the head
    theta = 0
    v_click = nose_click-mid_click
    # make it a 3d vector to use with the handy function
    target = np.append(v_click,0)
    angle_with_x = angle_between(np.array([1.,0,0]),target)
    phi = angle_with_x - gamma # NB since phi is with respect to x', not x

    return alpha,beta,gamma,s,theta,phi,t_body

def click_mouse_body(top_folder,start_frame,p_time_subset = .5):
    positions,weights = read_processed_frame(top_folder,start_frame,p_time_subset=p_time_subset)
    #plot the frame
    #color3d(positions)
    # click the mouse to generate starting positions
    hip_click,mid_click,nose_click = click_one_mouse(positions)
    #convert the clicks to a guess
    alpha,beta,gamma,s,theta,phi,t_body = good_guess(hip_click,mid_click,nose_click)
    # and save the best guess
    x0_guess = np.hstack((alpha,beta,gamma,s,theta,phi,t_body))
    return x0_guess

def initialize_x0(top_folder,start_frame=0,click_start=True,p_time_subset = 0.2):
    if click_start:
        # open a start frame, plot, and accept clicks
        x0_mouse0 = click_mouse_body(top_folder,start_frame)
        x0_mouse1 = click_mouse_body(top_folder,start_frame)
        x0_start = np.hstack((x0_mouse0,x0_mouse1))
    else:
        # the starting guess, in numpy on cpu
        x0_start = np.array([ 0.        ,  2.90604767,  0.5       ,  0.        , -0.18267935,
        -0.00996607, -0.00510009,  0.022     ,  0.        ,  2.53930531,
        0.5       ,  0.        ,  0.28053679,  0.09732097,  0.05387669,
        0.022     ])
    return x0_start



# r = residual(part,pos)

# r,J = jacobian_approx(part,pos)

#%% to make a figure, plotting the initial mouse
def plot_particles(ax,particles,i_frame,body_constants,alpha = 0.1):
    if len(particles.shape) == 1:
        particles = np.tile(particles,(1,1))

    positions,weights = read_processed_frame(top_folder,i_frame)
    n_particles = particles.shape[0]
    # plot the particle mice!

    # adjust the bottom!
    scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c=weights/np.max(weights),alpha=.2,marker='o',s=1)
    for i in range(n_particles):
        x_fit = particles[i,:]
        add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],body_constants,color = 'r',alpha = alpha)
        if len(x_fit) > 8:
            x_fit = x_fit[8:]
            add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],body_constants,color = 'orange',alpha = alpha)
    close_3d(ax,positions)
    fz = 10
    #  ax.set_xlabel('x (mm)',fontsize=6)
    #   ax.set_ylabel('y (mm)',fontsize=6)
    #    zlabel = ax.set_zlabel('z (mm)',fontsize=6)
    ax.xaxis.label.set_size(fz)
    ax.yaxis.label.set_size(fz)
    ax.zaxis.label.set_size(fz)

def plot_fitted_mouse():
    # the winning mouse is the one, with the lowest final loss
    #end_loss = [np.mean(ll[-1:]) for ll in ll_holder]

    #best_idx = np.argmin(end_loss)
    #best_mouse = best_holder[best_idx]

    which_opt = 0
    opt_names = ['geoLM']
    i_frame = 0
    #fig,ax = open_3d()
    fig = plt.figure(figsize=(10,15))

    ax = fig.add_subplot(3, 2, 1, projection='3d')
    plot_particles(ax,x0_start,i_frame,alpha = .5)
    ax.set_title("Initial clicked mouse")

    ax = fig.add_subplot(3, 2, 2, projection='3d')
    plot_particles(ax,best_mouse,i_frame,alpha = .5)
    ax.set_title("After fitting w. "+opt_names[which_opt])


    ax = fig.add_subplot(3, 2, 3, projection='3d')
    plot_particles(ax,x0_start,i_frame,alpha = .5)
    ax.view_init(90, 0)
    ax.set_title("Initial clicked mouse")

    ax = fig.add_subplot(3, 2, 4, projection='3d')
    plot_particles(ax,best_mouse,i_frame,alpha = .5)
    ax.view_init(90, 0)
    ax.set_title("After fitting w. "+opt_names[which_opt])

    ax = fig.add_subplot(3, 2, 5, projection='3d')
    plot_particles(ax,x0_start,i_frame,alpha = .5)
    ax.view_init(0, -90)
    ax.set_title("Initial clicked mouse")

    ax = fig.add_subplot(3, 2, 6, projection='3d')
    plot_particles(ax,best_mouse,i_frame,alpha = .5)
    ax.view_init(0, -90)
    ax.set_title("After fitting w. "+opt_names[which_opt])

    plt.show(fig)

#%% normal lm routine -- hmmmmmmmmmmmm
