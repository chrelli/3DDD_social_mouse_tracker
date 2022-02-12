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






#%% A few geometry functions
# @njit
def unit_vector(v):
    if np.sum(v) != 0:
        v = v/np.sqrt(v[0]**2+v[1]**2+v[2]**2 )
    return v


#@njit
def angle_between(v1, v2):
    """ Returns the SIGNED!! angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    angle = np.arccos(np.dot(v1_u, v2_u))
    cross_product = np.cross(v1_u,v2_u)
    ez = np.array([0,0,1.])

    # check for the sign!
    if np.dot(cross_product,ez)< 0:
        return -angle
    else:
        return angle


def click_one_mouse(positions):
    import cmocean
    ###############
    # Show a 2D plot and ask for two clicks
    ###############
    plt.figure(figsize = (5,5))
    plt.scatter(positions[:,0],positions[:,1],c=positions[:,2]/np.max(positions[:,2]),s=5,cmap=cmocean.cm.algae_r)
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
    psi = 0

    return alpha,beta,gamma,s,psi,theta,phi,t_body

def click_mouse_body(positions):
    hip_click,mid_click,nose_click = click_one_mouse(positions)
    #convert the clicks to a guess
    alpha,beta,gamma,s,psi,theta,phi,t_body = good_guess(hip_click,mid_click,nose_click)
    # and save the best guess
    x0_guess = np.hstack((alpha,beta,gamma,s,psi,theta,phi,t_body))
    return x0_guess,hip_click,mid_click,nose_click

def initialize_x0(positions,click_start=True):
    if click_start:
        # open a start frame, plot, and accept clicks
        x0_mouse0,hip_click0,mid_click0,nose_click0 = click_mouse_body(positions)
        x0_mouse1,hip_click1,mid_click1,nose_click1 = click_mouse_body(positions)
        x0_start = np.hstack((x0_mouse0,x0_mouse1))
        click_holder = [hip_click0,mid_click0,nose_click0,hip_click1,mid_click1,nose_click1]
    else:
        # the starting guess, in numpy on cpu
        x0_start = np.array([ 0.        ,  2.90604767,  0.5       ,  0.        , -0.18267935,
        -0.00996607, -0.00510009,  0.022     ,  0.        ,  2.53930531,
        0.5       ,  0.        ,  0.28053679,  0.09732097,  0.05387669,
        0.022     ])
    return x0_start,click_holder

