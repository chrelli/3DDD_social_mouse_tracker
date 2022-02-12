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


#%% ####################################
# Init the variables
#####################################

# where to put the model?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# put the constants for the mouse bodies onto the gpu!
body_scale =1.
## HIP is a prolate ellipsoid, centered along the x axis
a_hip_min = 0.03/2 #.01m
a_hip_max = 0.06/2 #.055m
b_hip_min = 0.04/2 #.03m
b_hip_max = 0.05/2 #.035m, was 0.046, which was too much

# converting it to the new terminology
a_hip_0     = torch.Tensor([body_scale*a_hip_min ]).to(device)#m
a_hip_delta = torch.Tensor([body_scale*(a_hip_max - a_hip_min)] ).to(device)#m
b_hip_0     = torch.Tensor([body_scale*b_hip_min ]).to(device)#m
b_hip_delta = torch.Tensor([body_scale*(b_hip_max - b_hip_min)] ).to(device)#m


## NOSE is prolate ellipsoid, also along the head direction vector
# here, there is no re-scaling
a_nose = torch.Tensor([body_scale*0.045/2]).to(device)#m was .04
b_nose = torch.Tensor([body_scale*0.025/2]).to(device) #m

a_nose = torch.Tensor([body_scale*0.028/2]).to(device)#m was .04
b_nose = torch.Tensor([body_scale*0.018/2]).to(device) #m

a_nose = torch.Tensor([body_scale*0.04/2]).to(device)#m long axis was .04
b_nose = torch.Tensor([body_scale*0.03/2]).to(device) #m was.3

d_nose = torch.Tensor([body_scale*0.01]).to(device) #m

r_impl = 1.1*b_nose
x_impl = 1.* d_nose+.7*a_nose
z_impl = 1.5* r_impl# .0+0*1.5*r_impl

r_impl = 0.9*b_nose
x_impl = 1.* d_nose+.5*a_nose
z_impl = 1.5* r_impl# .0+0*1.5*r_impl

# make a list of the body constants to pass and save!
body_constants = np.asanyarray([body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose.numpy(),b_nose.numpy(),d_nose.numpy(),x_impl.numpy(),z_impl.numpy(),r_impl.numpy()]).astype('float32')

# DIRTY IMPORT
from utils.fitting_utils import *
from utils.trkr_utils import *
from utils.torch_utils import *

def particles_to_distance(part,pos,implant = False):
    # calculates the distance to one mouse, given positions and particles
    alpha = part[:,0]
    beta = part[:,1]
    gamma = part[:,2]

    s = part[:,3]
    #todo naming here is off
    psi = part[:,4]
    theta = part[:,5]
    phi = part[:,6]
    t_body = part[:,7:10]

    # calculate vectors holding the hip values!
    # the values are n_particles long
    a_hip = a_hip_0 + a_hip_delta * s
    b_hip = b_hip_0 + b_hip_delta * (1.-s)
    d_hip = .75 * a_hip

    # we need to do cos an sin on the angles!
    # and other places too over and over, let's just keep them in mem?
    one  = torch.ones_like(s)
    zero = torch.zeros_like(s)


    R_body = make_xyz_rotation(alpha,beta,gamma,one,zero)
    R_head = make_xyz_rotation(psi,theta,phi,one,zero)
    # also batch-wise here!
    R_nose = torch.einsum('aij,ajk->aik',[R_body,R_head])


    # use einsum to multiply Q with R.T, to get R @ Q @ R.T
    #Q_hip = R_body @  torch.einsum('aij,akj->aik', [make_inner_Q(a_hip,b_hip,n_particles),R_body])

    # again, but here the head is always the same
    aa = 1./a_nose**2
    bb = 1./b_nose**2
    Q_inner = torch.diagflat(torch.stack([aa,bb,bb]))

    Q_nose = torch.einsum('aij,akj->aik', [torch.einsum('aij,jk->aik', [R_nose ,Q_inner] ),R_nose ] )

    aa = 1./a_hip**2
    bb = 1./b_hip**2
    # stupid way of dpoing it:
    # Q_inner =  torch.stack([torch.stack([aa,  zero, zero],      dim=1),
    #               torch.stack([zero,  bb,  zero],      dim=1),
    #               torch.stack([zero,       zero,       bb],       dim=1)], dim=1)
    # probably a faster way: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560/2
    # this uses ztriding to make the batch
    Q_inner = batch_diagonal(torch.stack([aa,bb,bb],dim=1))
    # now, we go over the hips, remember to batch
    Q_hip = torch.einsum('aij,akj->aik', [torch.einsum('aij,ajk->aik', [R_body ,Q_inner] ),R_body ] )

    # THESE COORDINATES ARE WIHTIN THE REF FRAME OF THE MOUSE
    # c_nose is just d_nose, rotated by R_nose:
    # so for ever 2D matrix, it's c_nose =  R_nose @ np.array([d_nose,0,0])
    # and this only selects the first column of R_head and sums it
    c_mid = torch.stack([d_hip,zero,zero],dim=1)
    c_nose = c_mid + d_nose * R_head[:,:,0]
    # c_nose = c_mid + R_head @ (torch.tensor([d_nose,0,0],dtype=torch.float32))

    # c_impl =  torch.einsum('aij,j->ai',[R_head,torch.tensor([x_impl,0,z_impl],dtype=torch.float32)])
    # new!

    c_impl = c_mid + R_head @ (torch.tensor([x_impl,0,z_impl],dtype=torch.float32))


    # NOW HERE IS THE CONVERTION

    # Now, calculate the distance vectors from the origin of the hip, mid and head, in the real world
    c_hip = t_body
    c_nose = torch.einsum('aij,aj->ai',[R_body,c_nose]) + t_body
    c_impl = torch.einsum('aij,aj->ai',[R_body,c_impl]) + t_body

    # now, we get the distance to the hip
    # we use extended dimentions to make use of broadcasting
    # keep the first dimension the particles
    # in order for broadcasting in numpy to work, we transpose th last two dimensions here

    c_hip = torch.unsqueeze(c_hip,1).transpose(-2,-1)
    c_nose = torch.unsqueeze(c_nose,1).transpose(-2,-1)
    c_impl = torch.unsqueeze(c_impl,1).transpose(-2,-1)

    pos = torch.unsqueeze(pos,0).transpose(-2,-1)

    # now we can just subtract, and tf will broadcast automatically
    # now the points are n_particles x n_points x 3 spatial dimensions
    # TODO optimize this from the beginning to avoid the transposition!
    p_hip = pos-c_hip
    p_nose = pos-c_nose

    # inner prduct between the position and Q
    # pQp_hip = torch.sum( p_hip *( Q_hip @ p_hip ) , dim =1)
    # pQp_nose = torch.sum( p_nose *( Q_nose @ p_nose ) ,dim=1)
    # signed_d_hip = ( 1. - 1./torch.sqrt( torch.sum( p_hip *( Q_hip @ p_hip ) , dim =1) ) ) * torch.norm(p_hip,dim = 1)

    delta_hip_signed = ( 1. - 1./torch.sqrt( torch.sum( p_hip *( Q_hip @ p_hip ) , dim =1) ) ) * torch.norm(p_hip,dim = 1)
    delta_nose_signed = ( 1. - 1./torch.sqrt( torch.sum( p_nose *( Q_nose @ p_nose ) , dim =1) ) ) * torch.norm(p_nose,dim = 1)

    # plt.figure();plt.plot(delta_hip_signed.numpy()[0]);plt.plot(delta_nose_signed.numpy()[0]);plt.show()

    dist = torch.min(torch.abs(delta_hip_signed),torch.abs(delta_nose_signed))
    unsigned_dist = torch.clone(dist)
    if implant:
        # collected
        p_impl = pos-c_impl
        delta_impl = torch.norm(p_impl,dim = 1) - r_impl
        dist = torch.min(dist,np.abs(delta_impl) )

    # plt.figure();plt.plot(dist.numpy()[0]);plt.show()

    # update with BIG loss for the ones inside!
    # indices = torch.arange(dist.shape[1])

    # inside_nose = torch.masked_select(indices,delta_hip_signed <0.)
    # inside_hip = torch.masked_select(indices,delta_hip_signed <0.)

    # dist[0,indices[] ] = 3*torch.abs(delta_hip_signed)

    # IF THE POINT WAS INSIDE THE NOSE, TRIPLE THE RESIDUAL!!
    # dist[delta_nose_signed < 0.] = 3.*torch.abs(delta_nose_signed)[delta_nose_signed < 0.]
    # dist[delta_hip_signed < 0.] = 3.*torch.abs(delta_hip_signed)[delta_hip_signed < 0.]
    #
    # dist[(delta_nose_signed < 0. )|( delta_hip_signed < 0.)] = unsigned_dist[(delta_nose_signed < 0. )|( delta_hip_signed < 0.)]


    # plt.figure();plt.plot(dist.numpy()[0]);plt.show()

    # delta_hip = torch.abs( 1. - 1./torch.sqrt( torch.sum( p_hip *( Q_hip @ p_hip ) , dim =1) ) ) * torch.norm(p_hip,dim = 1)
    # delta_nose = torch.abs( 1. - 1./torch.sqrt( torch.sum( p_nose *( Q_nose @ p_nose ) , dim =1) ) ) * torch.norm(p_nose,dim = 1)

    # TODO try if reduce min is faster than minimum twice?
    # dist = torch.min(delta_hip,delta_nose)

    # for keypoints: also return a lsit of support points for the body!

    c_ass = torch.stack([-a_hip,zero,zero],dim=1)
    c_tip = c_mid + (d_nose + a_nose) * R_head[:,:,0]
    # c_nose = c_mid + R_head @ (torch.tensor([d_nose,0,0],dtype=torch.float32))
    c_ass = torch.einsum('aij,aj->ai',[R_body,c_ass]) + t_body
    c_tip = torch.einsum('aij,aj->ai',[R_body,c_tip]) + t_body
    c_mid = torch.einsum('aij,aj->ai',[R_body,c_mid]) + t_body


    c_ass = torch.unsqueeze(c_ass,1).transpose(-2,-1)
    c_tip = torch.unsqueeze(c_tip,1).transpose(-2,-1)
    c_mid = torch.unsqueeze(c_mid,1).transpose(-2,-1)


    body_support = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    return dist,unsigned_dist,body_support

#%% Helpers to CALCULATE THE CENTER FROM THE particles_to_distance

def unsigned_residual(part,pos,overlap_penalty = False):
    _, dist0, _ = particles_to_distance(part[:,:10],pos,implant = True)
    _, dist1,_ = particles_to_distance(part[:,10:],pos)
    r = torch.min(dist0,dist1)
    if overlap_penalty:
        r = r + ball_cost(part)
    return r


def add_implant_residual(r,keyp,ikeyp,body_support_0, setpoint = 0.0135, scaling = 10.):
    #     # stack on first dim?
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    c_impl = body_support_0[5][...,0]

    keyp_implant = (ikeyp == 0)

    if sum(keyp_implant) > 0:
        # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)
        #         print(r_implant)
        r = torch.cat([r,r_implant],dim=1)
    return r


def add_ass_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 3, setpoint = 0.0, scaling = 10.):
    #     # stack on first dim?
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    c_impl = torch.cat(( body_support_0[1][...,0], body_support_1[1][...,0]))

    keyp_implant = (ikeyp == which_keyp)

    if sum(keyp_implant) > 0:
        # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance
        keypoint_distance = torch.min(keypoint_distance[:r.shape[0],:], keypoint_distance[r.shape[0]:,:])

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)

        r = torch.cat([r,r_implant],dim=1)
    return r

def add_ear_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 1, setpoint = 0.015, scaling = 10.):
    #     # stack on first dim?
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    c_impl = torch.cat(( body_support_0[2][...,0], body_support_1[2][...,0]))

    keyp_implant = (ikeyp == which_keyp)

    if sum(keyp_implant) > 0:
        # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance
        keypoint_distance = torch.min(keypoint_distance[:r.shape[0],:], keypoint_distance[r.shape[0]:,:])

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)

        r = torch.cat([r,r_implant],dim=1)
    return r


def add_nose_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 2, setpoint = 0.0, scaling = 10.):
    #     # stack on first dim?
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    #  [0,    1,    2 ,   3,     4 ,   5     ]
    c_impl = torch.cat(( body_support_0[4][...,0], body_support_1[4][...,0]))

    keyp_implant = (ikeyp == which_keyp)

    if sum(keyp_implant) > 0:
        # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance
        keypoint_distance = torch.min(keypoint_distance[:r.shape[0],:], keypoint_distance[r.shape[0]:,:])

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)

        r = torch.cat([r,r_implant],dim=1)
    return r


def residual(part,pos,keyp,ikeyp,overlap_penalty = False,clip=True):
    dist0,_,body_support_0 = particles_to_distance(part[:,:10],pos,implant = True)
    dist1,_,body_support_1 = particles_to_distance(part[:,10:],pos)
    r = torch.min(dist0,dist1)
    if overlap_penalty:
        r = r + ball_cost(part)
    if clip:
        r = torch.clamp(r,0,.025)

    # for the scaling
    big_n = r.shape[0]

    r = add_implant_residual(r,keyp,ikeyp,body_support_0, setpoint = 0.0135, scaling = big_n/50)
    r = add_ass_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 3, setpoint = 0.0, scaling = big_n/500)
    r = add_ear_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 1, setpoint = 0.015, scaling = big_n/500)
    r = add_nose_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 2, setpoint = 0.0, scaling = big_n/500)

    return r

def jacobian_approx(part,pos,keyp,ikeyp,overlap_penalty = False):
    # TODO some rescaling maybe. Changes in xyz are sort of on the order of
    # 10 times smaller than changes in beta gamma s theta phi

    # takes the point, and calculates the jacobian around it
    # first, we make a tensor, no gradients here
    #part = torch.tensor(x0_start,dtype = torch.float32,requires_grad = False).to(device).unsqueeze(0)
    # we do forward approximation, so we need to add epsilon to all parameters
    epsilon = torch.tensor(1e-4)
    # now we can use broadcasting to generate the test points:
    part_test = epsilon * torch.eye(part.shape[1]) + part

    # concatenate the real parameters onto here:
    # part: each row is a particle, row 0 is the real residual
    part = torch.cat([part,part_test])

    # now, calcuate the residuals for these parlicles:
    r_all = residual(part,pos,keyp,ikeyp)
    if overlap_penalty:
        #TODO this is not so pretty, will clean up later
        r_all = r_all + ball_cost(part).unsqueeze(0).transpose(0,1)

    r = r_all[0,:]
    #We can use broadcasting to calculate the jacobian:
    # the residuals are batch
    Jac = torch.transpose( ( r_all[1:,:] - r ) / epsilon ,-2,-1)

    # and we can transpose the jacobian, if we want

    #return r.detach().cpu().numpy(),torch.transpose(Jac,-1,-2).detach().cpu().numpy()
    return r,Jac










#%% more complicated routine also has geodesic etc
from utils.ellipsoid_utils import check_mice_for_overlap,plot_mouse_ellipsoids

def apply_weights_to_r_and_J(r,J):
    # this sets the cutoff|
    k_weights = torch.clamp( 4.6*torch.median(r) ,min= 0.03) # or simply 2?
    # calculate which residuals are small enough
    weights_logic = r<k_weights
    # cut down the r!
    r = r[weights_logic]
    # r is already positive, so no need to calculate magnitude of the bisquare loss
    w = torch.pow(1. - torch.pow(r/k_weights,2.),2.)
    # w = torch.ones_like(r)
    # update r using weights
    r = w*r
    # and also update the Jacobian
    J = J[weights_logic,:]*w.unsqueeze(1)
    return r,J

def apply_weights_to_r(r):
    # this sets the cutoff|
    k_weights = torch.clamp( 4.6*torch.median(r) ,min= 0.03) # or simply 2?True
    # calculate which residuals are small enough
    weights_logic = r<k_weights
    # cut down the r!
    r = r[weights_logic]
    # r is already positive, so no need to calculate magnitude of the bisquare loss
    w = torch.pow(1. - torch.pow(r/k_weights,2.),2.)
    # w = torch.ones_like(r)
    # update r using weights
    r = w*r
    return r

#%% TRY a whole run!
# this is the x0 guess! plt.close('all')


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
    psi = 0

    return alpha,beta,gamma,s,psi,theta,phi,t_body

def click_mouse_body(positions):
    hip_click,mid_click,nose_click = click_one_mouse(positions)
    #convert the clicks to a guess
    alpha,beta,gamma,s,psi,theta,phi,t_body = good_guess(hip_click,mid_click,nose_click)
    # and save the best guess
    x0_guess = np.hstack((alpha,beta,gamma,s,psi,theta,phi,t_body))
    return x0_guess

def initialize_x0(positions,click_start=True):
    if click_start:
        # open a start frame, plot, and accept clicks
        x0_mouse0 = click_mouse_body(positions)
        x0_mouse1 = click_mouse_body(positions)
        x0_start = np.hstack((x0_mouse0,x0_mouse1))
    else:
        # the starting guess, in numpy on cpu
        x0_start = np.array([ 0.        ,  2.90604767,  0.5       ,  0.        , -0.18267935,
        -0.00996607, -0.00510009,  0.022     ,  0.        ,  2.53930531,
        0.5       ,  0.        ,  0.28053679,  0.09732097,  0.05387669,
        0.022     ])
    return x0_start


search_cone = torch.Tensor(np.tile([.2,.2,.2,.1,.2,.2,.2,.01,.01,.01],2)).to(device).unsqueeze(0)


def klm_routine(part,pos,keyp,ikeyp,max_iters = 100,verbose=False,save_history = True,geodesic= False,ftol = 1e-4, search_cone = search_cone):

    # for limits
    upper_limit = part + search_cone
    lower_limit = part - search_cone

    # lambda for dampening is kind of an unfortiunate name in python, but w/e
    lamb = torch.tensor(np.exp(2.)).to(device) #TODO set the start value of lambda, based on the step size, perhaps? T&S say 1, from inspection e10
    lamb_up = torch.tensor(2.).to(device) # Trasnstrum & Sethna 2012 suggest 2 up,3 down for medium, 1.5 up and 5 down for large problems
    lamb_down = torch.tensor(3.).to(device)

    # Calculate the r and jacobian opf the staring position
    r,J = jacobian_approx(part,pos,keyp,ikeyp)

    # and the starting loss!
    C = 0.5 * torch.dot(r,r)

    MAXSTEP = 100
    i = 0

    # why this safety factor??
    SAFE = 0.5

    # make holding lists for the cost and position
    h_C = []
    h_C_new = []
    h_r = []
    h_lamb = []
    h_part = []
    h_step = []

    for _ in range(max_iters):
        if verbose:
            i += 1
            print(i)

        # calc g and nablaC
        # g = j.T.dot(j) + l*I
        g = torch.einsum('ji,jk->ik',[J,J]) + torch.diagflat(lamb * torch.ones_like(part))
        # gradC = j.T.dot(r)
        nablaC = torch.einsum('ji,j->i',[J,r])

        # solve gg x = gradC, d1 = -x
        # -d1 = solve(gg,gradC)
        d1, LU1 = torch.gesv(nablaC.unsqueeze(1),g) # wierdly it wants
        d1 = - d1[:,0]

        # print(d1)

        # xnew = x - SAFE*solve(g,gradC)
        # x_new = part + SAFE * d1
        d1 = torch.clamp(d1,min=-.2, max = .2)

        x_new = part + d1

        # CLIP the s
        # alpha, beta, gamma, s,
        # x_new = torch.max(torch.min(x_new,upper_limit),lower_limit)
        x_new[0,[3,3+10]] = torch.clamp(x_new[0,[3,3+10]],min=0, max = 1)




        r_new = residual(x_new,pos,keyp,ikeyp)[0,:]
        C_new = 0.5 * torch.dot(r_new,r_new)

        # what will the change be?
        deltaC = C_new - C

        if verbose:
            # print()
            print("deltaC: {:.6f}, C_old: {:.6f}, C_new: {:.6f}".format(deltaC.numpy(),C.numpy(),C_new.numpy()) )
            print(deltaC)
            print(C_new)
            print(C)

        if deltaC < 0:
            if save_history:
                h_C.append(C.numpy())
                h_C_new.append(C_new.numpy())
                h_r.append(torch.median(r).numpy())
                h_lamb.append(lamb.numpy())
                h_part.append(x_new.numpy())
                h_step.append(1.)

            # update part, r and J
            part = x_new
            # Calculate the r and jacobian opf the staring position
            r,J = jacobian_approx(part,pos,keyp,ikeyp)
            # and the starting loss!
            C = 0.5 * torch.dot(r,r)
            # and update lamb
            lamb = lamb/lamb_down

        else:
            lamb = lamb*lamb_up
            if save_history:
                h_C.append(C.numpy())
                h_C_new.append(C_new.numpy())
                h_r.append(torch.median(r).numpy())
                h_lamb.append(lamb.numpy())
                h_part.append(x_new.numpy())
                h_step.append(0.)

        # check for stopping criteria!
        # if lambda is too big,
        # if there was a step own in the median errors
        # if there was a step down in the cost function, which was smaller than ftol = 0.0001

        if np.log(lamb) > 10 or (deltaC < 0 and np.abs(deltaC) < ftol):
            break

    return part,[h_C, h_C_new, h_r, h_lamb, h_part, h_step]



def lm_routine(part,pos,max_iters = 100,verbose=False,save_history = True,geodesic= False,ftol = 1e-4):
    # lambda for dampening is kind of an unfortiunate name in python, but w/e
    lamb = torch.tensor(np.exp(2.)).to(device) #TODO set the start value of lambda, based on the step size, perhaps? T&S say 1, from inspection e10
    lamb_up = torch.tensor(2.).to(device) # Trasnstrum & Sethna 2012 suggest 2 up,3 down for medium, 1.5 up and 5 down for large problems
    lamb_down = torch.tensor(3.).to(device)

    # Calculate the r and jacobian opf the staring position
    r,J = jacobian_approx(part,pos)

    # and the starting loss!
    C = 0.5 * torch.dot(r,r)

    MAXSTEP = 100
    i = 0

    # why this safety factor??
    SAFE = 0.5

    # make holding lists for the cost and position
    h_C = []
    h_C_new = []
    h_r = []
    h_lamb = []
    h_part = []
    h_step = []

    for _ in range(max_iters):
        if verbose:
            i += 1
            print(i)

        # calc g and nablaC
        # g = j.T.dot(j) + l*I
        g = torch.einsum('ji,jk->ik',[J,J]) + torch.diagflat(lamb * torch.ones_like(part))
        # gradC = j.T.dot(r)
        nablaC = torch.einsum('ji,j->i',[J,r])

        # solve gg x = gradC, d1 = -x
        # -d1 = solve(gg,gradC)
        d1, LU1 = torch.gesv(nablaC.unsqueeze(1),g) # wierdly it wants
        d1 = - d1[:,0]

        # print(d1)

        # xnew = x - SAFE*solve(g,gradC)
        # x_new = part + SAFE * d1
        x_new = part + d1

        # CLIP the s
        # alpha, beta, gamma, s,
        # x_new[0,[2,2+8]] = torch.clamp(x_new[0,[3,3+10]],min=0, max = 1)

        r_new = residual(x_new,pos)[0,:]
        C_new = 0.5 * torch.dot(r_new,r_new)

        # what will the change be?
        deltaC = C_new - C

        if verbose:
            # print()
            print("deltaC: {:.6f}, C_old: {:.6f}, C_new: {:.6f}".format(deltaC.numpy(),C.numpy(),C_new.numpy()) )
            print(deltaC)
            print(C_new)
            print(C)

        if deltaC < 0:
            if save_history:
                h_C.append(C.numpy())
                h_C_new.append(C_new.numpy())
                h_r.append(torch.median(r).numpy())
                h_lamb.append(lamb.numpy())
                h_part.append(x_new.numpy())
                h_step.append(1.)

            # update part, r and J
            part = x_new
            # Calculate the r and jacobian opf the staring position
            r,J = jacobian_approx(part,pos)
            # and the starting loss!
            C = 0.5 * torch.dot(r,r)
            # and update lamb
            lamb = lamb/lamb_down

        else:
            lamb = lamb*lamb_up
            if save_history:
                h_C.append(C.numpy())
                h_C_new.append(C_new.numpy())
                h_r.append(torch.median(r).numpy())
                h_lamb.append(lamb.numpy())
                h_part.append(x_new.numpy())
                h_step.append(0.)

        # check for stopping criteria!
        # if lambda is too big,
        # if there was a step own in the median errors
        # if there was a step down in the cost function, which was smaller than ftol = 0.0001

        if np.log(lamb) > 10 or (deltaC < 0 and np.abs(deltaC) < ftol):
            break

    return part,[h_C, h_C_new, h_r, h_lamb, h_part, h_step]


#%% TRY to generate a version for only one mouse!

def geolm_routine_single_implanted(part,pos,max_iters = 100,verbose=False,save_history = True,geodesic= False,uphill=False,overlap_penalty=False,robust_weights=False,random_learning=False,ftol = 1e-4,upper_limit=None,lower_limit = None):
    '''               _     __  __
      __ _  ___  ___ | |   |  \/  |
     / _` |/ _ \/ _ \| |   | |\/| |
    | (_| |  __/ (_) | |___| |  | |
     \__, |\___|\___/|_____|_|  |_|
     |___/

    '''
    # lambda for dampening is kind of an unfortiunate name in python, but w/e
    lamb = torch.tensor(np.exp(2.)).to(device) #TODO set the start value of lambda, based on the step size, perhaps? T&S say 1, from inspection e10
    lamb_up = torch.tensor(2.).to(device) # Trasnstrum & Sethna 2012 suggest 2 up,3 down for medium, 1.5 up and 5 down for large problems
    lamb_down = torch.tensor(3.).to(device)

    lamb_up = torch.tensor(1.5).to(device) # Trasnstrum & Sethna 2012 suggest 2 up,3 down for medium, 1.5 up and 5 down for large problems
    lamb_down = torch.tensor(5.).to(device)
    # cut it down to one mouse only!
    part = part[0,:10].unsqueeze(0)

    # Calculate the r and jacobian opf the staring position
    if random_learning:
        n_pos = pos.shape[0]
        n_to_fit = round(0.4*n_pos)
        # get a random subset from the loaded frame
        random_subidx=torch.randperm(n_pos)[:n_to_fit]
        # [subset_filter,0:3]
        r,J = jacobian_approx_single(part,pos[random_subidx,:],overlap_penalty=overlap_penalty)
    else:
        r,J = jacobian_approx_single(part,pos,overlap_penalty=overlap_penalty)

    # apply weights
    if robust_weights:
        r,J = apply_weights_to_r_and_J(r,J)

    # plt.figure()
    # # plt.plot(r.numpy())
    # plt.plot(w[weights_logic].numpy())
    # # plt.plot(weights_logic.numpy())
    # # plt.imshow(J,aspect='auto')
    # # plt.imshow(J[weights_logic ==1,:],aspect='auto')
    # plt.show()

    # and the starting loss!
    C = 0.5 * torch.dot(r,r)

    MAXSTEP = 100
    i = 0

    # why this safety factor??
    SAFE = 0.5

    # make holding lists for the cost and position
    h_C = []
    h_C_new = []
    h_r = []
    h_lamb = []
    h_part = []
    h_step = []

    # set up the previous change for the upfill steps
    # and set a boolean to let us know that
    d_previous = torch.zeros_like(part)
    has_accepted_one_step = False

    # set up the overlap overlap_penalty
    C_overlap = torch.tensor(0.)

    for _ in range(max_iters):
        if verbose:
            i += 1
            print(i)

        # calc g and nablaC
        # g = j.T.dot(j) + l*I
        g = torch.einsum('ji,jk->ik',[J,J]) + torch.diagflat(lamb * torch.ones_like(part))
        # gradC = j.T.dot(r)
        nablaC = torch.einsum('ji,j->i',[J,r])

        # solve gg x = gradC, d1 = -x
        # -d1 = solve(gg,gradC)
        d1, LU1 = torch.gesv(nablaC.unsqueeze(1),g) # wierdly it wants
        d1 = - d1[:,0]

        # calculate the 2nd order correction!

        if geodesic:
            h = 0.01 # calculate the 2nd deriv using 10% of the linear step
            # that paper calls the Hessian K:, it's the directs 2nd deriv
            if random_learning:
                random_subidx=torch.randperm(n_pos)[:n_to_fit]
                r_direct_second_deriv = (2./h) * ( (residual_single(part + h * d1,pos[random_subidx,:],overlap_penalty=overlap_penalty) - r )/h - torch.einsum('ij,j->i',[J,d1]) )
            else:
                r_direct_second_deriv = (2./h) * ( (residual_single(part + h * d1,pos,overlap_penalty=overlap_penalty) - r )/h - torch.einsum('ij,j->i',[J,d1]) )
            # find the 2nd order step by solving!
            # dx2 = - 0.5*solve(g, j.T.dot(k))
            d2, LU2 = torch.gesv( torch.einsum('ji,j->i',[J,r_direct_second_deriv[0] ]) , g ) # wierdly it wants
            d2 = - 0.5 * d2[:,0]

            # from T & S 2012, they suggest an alpha criteria of .75, defined like this:
            alpha = torch.tensor(0.75)

            # calculate the test alpha fraction (eq 15 in T & S 2012)
            taylor_error = 2*torch.norm(d2)/torch.norm(d1)

            # only add the 2nd order correction, if the taylor error is low enough
        else:
            # if no geodesig, just set d2 to zero
            d2 = torch.tensor(0.)

        if geodesic and (taylor_error > alpha):
            # set back to zero if the taylor error is too big
            # todo kind of confusing
            d2 = torch.tensor(0.)

        # xnew = x - SAFE*solve(g,gradC)
        # x_new = part + SAFE * d1

        # clip the gradient!
        # max_angle_step=3.14/40.
        # d1 = torch.clamp(d1,min = -max_angle_step,max=max_angle_step)

        x_new = part + d1 + d2
        # print(d1[4])

        # First we clamp the particle within the limits around the proposed guess
        if upper_limit is not None:
            #
            x_new = torch.max(torch.min(x_new,upper_limit),lower_limit)

        # THEN we clip the particle with some hard limits #TODO make this one line?
        single_mouse = True
        if single_mouse:
            x_new[0,[3]] = torch.clamp(x_new[0,[3]],min=0.2, max = 1)
            # x_new = enforce_hard_limits(x_new,hard_lo,hard_hi)
            x_new[0,[5,6]] = torch.clamp(x_new[0,[5,6]],min=-1.3, max = 1.3)
            # x_new[0,[4]] = torch.clamp(x_new[0,[4]],min=-2*3.14, max = 2*3.14)

            x_new[0,[0]] = 0*x_new[0,[0]]

            x_new[0,[4]] = 0*x_new[0,[4]]

            # AND the Z a well!
            x_new[0,[9]] = torch.clamp(x_new[0,[9]],min=.015, max = .1)
        else:
            x_new[0,[2,2+8]] = torch.clamp(x_new[0,[2,2+8]],min=0.2, max = 1)
            # x_new = enforce_hard_limits(x_new,hard_lo,hard_hi)
            x_new[0,[3,4,11,12]] = torch.clamp(x_new[0,[3,4,11,12]],min=2*-1.5, max = 2*1.5)
            # AND the Z a well!
            x_new[0,[7,15]] = torch.clamp(x_new[0,[7,15]],min=.015, max = .1)

        if random_learning:
            random_subidx=torch.randperm(n_pos)[:n_to_fit]
            r_new = residual_single(x_new,pos[random_subidx,:],overlap_penalty=overlap_penalty)[0,:]
        else:
            r_new = residual_single(x_new,pos,overlap_penalty=overlap_penalty)[0,:]
        # apply weights???
        if robust_weights:
            r_new = apply_weights_to_r(r_new)

        # in the new proposed step, we need to add a penalty, if the mice are overlapping!
        C_new = 0.5 * torch.dot(r_new,r_new)

        # what will the change be?
        deltaC = C_new - C

        # check for uphill steps!
        bool_uphill = False
        if uphill and has_accepted_one_step:
            # calculate the angle between the proposed and old step
            beta_uphill = torch.dot(d1+d2,d_previous)/ (torch.norm(d1+d2) * torch.norm(d_previous) + 1e-6 )
            bool_uphill = (1 - beta_uphill) * C_new < C

        if (deltaC < 0) or bool_uphill:
            if save_history:
                h_C.append(C.numpy())
                h_C_new.append(C_new.numpy())
                h_r.append(torch.median(r).numpy())
                h_lamb.append(lamb.numpy())
                h_part.append(x_new.numpy())
                h_step.append(1.)

            # update part, r and J
            part = x_new
            # Calculate the r and jacobian opf the staring position
            if random_learning:
                random_subidx=torch.randperm(n_pos)[:n_to_fit]
                r,J = jacobian_approx_single(part,pos[random_subidx,:],overlap_penalty=overlap_penalty)
            else:
                r,J = jacobian_approx_single(part,pos,overlap_penalty=overlap_penalty)

            # apply weights?
            if robust_weights:
                r,J = apply_weights_to_r_and_J(r,J)

            # and the starting loss!
            C = 0.5 * torch.dot(r,r)
            # and update lamb
            lamb = lamb/lamb_down
            # and save the previous step for uphill steps
            d_previous = d1+d2
            has_accepted_one_step = True



        else:
            lamb = lamb*lamb_up
            if save_history:
                h_C.append(C.numpy())
                h_C_new.append(C_new.numpy())
                h_r.append(torch.median(r).numpy())
                h_lamb.append(lamb.numpy())
                h_part.append(x_new.numpy())
                h_step.append(0.)

        # check for stopping criteria!
        # if lambda is too big,
        # if there was a step own in the median errors
        # if there was a step down in the cost function, which was smaller than ftol = 0.0001

        if np.log(lamb) > 10 or (deltaC < 0 and np.abs(deltaC) < ftol): #cutoff was 10
            # these are the stopping criteria
            break

    return torch.cat((part,part),dim=1),[h_C, h_C_new, h_r, h_lamb, h_part, h_step]

#%%

def geolm_convergence_single(history, second = False):
    h_C, h_C_new, h_r, h_lamb, h_part,h_step = history[:]

    plt.figure()
    iters = np.arange(len(h_C))
    plt.subplot(6,1,1)
    plt.plot(iters,np.log(h_C),':')
    #plt.plot(iters,np.log(h_C_new),'o:')

    step_logic = np.asarray(h_step) == 1
    plt.plot(iters[step_logic],np.log(h_C_new)[step_logic],'og')
    plt.plot(iters[~step_logic],np.log(h_C_new)[~step_logic],'or')

    plt.legend(("Loss","proposed Loss"))
    plt.ylim([None,np.max(np.log(h_C)) + np.std(np.log(h_C))])

    plt.ylabel('$\log Loss$')

    plt.subplot(6,1,2)
    plt.plot(iters,np.log(h_lamb),'o:')
    plt.ylabel('$\log\lambda$')

    plt.subplot(6,1,3)
    plt.plot(np.multiply(h_r,1e3),':o')
    plt.ylabel("median residual [mm]")

    plt.plot(np.diff(np.multiply(h_r,1e3)),':o')

    plt.legend(("absolute","difference"))
    # conver the part list to a matrix_diag
    part_history = np.asarray(h_part).squeeze(1)
    if second:
        part_history = part_history[:,10:]
    part_history.shape

    plt.subplot(6,1,4)
    plt.ylabel("xyz history")
    plt.plot(part_history[:,[7,8,9]],'o:')

    plt.subplot(6,1,5)
    plt.ylabel("spine history")
    angle_history = part_history[:,[3]]
    plt.plot(angle_history,':o')


    plt.subplot(6,1,6)
    plt.ylabel("angle history")
    angle_history = part_history[:,[0,1,2,4,5,6]]
    plt.plot(angle_history,':o')


    for i in range(6):
        plt.subplot(6,1,i+1)
        plt.xlim((0,len(h_C)))

    plt.show()




# @njit
def rotate_body_model(alpha_body,beta_body,gamma_body):
    """
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


def torch_mouse_body_geometry(part,body_constants):
    """
    This function calculates the configuration of the mouse body
    In this configureation, it has four free parameters: azimuth and elevation of the nose/hip
    Returns the points, which define the model: center-points and radii
    theta el is elevation of the head (in xz plane)
    phi lr is head rotation in xy plane

    beta,gamma,s is hip pitch,yaw and spine scaling
    theta,phi is nose pitch,yaw (i.e. around y and z, respectively since major body is along x axis)

    """
    # get the constants for the body model
    # a_hip_0,a_hip_delta,b_hip_0,b_hip_delta,d_hip,a_nose,b_nose,d_nose = mouse_body_size_constants()

    body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl = body_constants

    alpha = part[0]
    beta = part[1]
    gamma = part[2]
    s = part[3]
    #todo naming here is off
    psi = part[4]
    theta = part[5]
    phi = part[6]
    t_body = part[7:10]


    a_hip_0     = body_scale*a_hip_min
    a_hip_delta = body_scale*(a_hip_max - a_hip_min)
    b_hip_0     = body_scale*b_hip_min
    b_hip_delta = body_scale*(b_hip_max - b_hip_min)

    # calculate the spine
    a_hip = a_hip_0 + s * a_hip_delta
    b_hip = b_hip_0 + (1-s)**1 * b_hip_delta

    # scale the hip position
    d_hip = .75*a_hip # tried this, no good
    # d_hip = a_hip - a_nose

    R_body = rotate_body_model(alpha,beta,gamma)
    R_head = rotate_body_model(psi,theta,phi)
    R_nose = R_body @ R_head

    # use einsum to multiply Q with R.T, to get R @ Q @ R.T
    #Q_hip = R_body @  torch.einsum('aij,akj->aik', [make_inner_Q(a_hip,b_hip,n_particles),R_body])

    # and the Q matrices
    Q_inner =np.zeros((3,3))
    Q_inner[[0,1,2],[0,1,2]] = np.asanyarray([1/a_hip**2,1/b_hip**2,1/b_hip**2]).ravel()
    Q_hip = R_body @ Q_inner @ R_body.T
    Q_inner[[0,1,2],[0,1,2]] = np.asanyarray([1/a_nose**2,1/b_nose**2,1/b_nose**2]).ravel()
    Q_nose = R_nose @ np.diag(np.array([1/a_nose**2,1/b_nose**2,1/b_nose**2])) @ R_nose.T

    # And now we get the spine coordinates
    c_hip = np.array([0,0,0])
    c_mid = np.array([d_hip,0,0])
    c_nose = c_mid + R_head @ np.array([d_nose,0,0])
    c_impl = c_mid + R_head @ np.array([x_impl,0,z_impl])


    # Now, calculate the distance vectors from the origin of the hip, mid and head, in the real world
    c_hip = c_hip + t_body
    c_nose = R_body @ c_nose + t_body
    c_impl = R_body @ c_impl + t_body


    # now, just return the coordinates and the radii
    return R_body,R_nose,c_mid,c_hip,c_nose,c_impl,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose




def add_torch_mouse_for_video(ax,part,body_constants,color = 'r',plot_alpha = .7, implant=False):
    # this also need a vector
    # get the geometry of the mouse body # not really the preferred way

    body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl = body_constants
    R_body,R_nose,c_mid,c_hip,c_nose,c_impl,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose = torch_mouse_body_geometry(part,body_constants)


    alpha = part[0]
    beta = part[1]
    gamma = part[2]

    s = part[3]
    #todo naming here is off
    psi = part[4]
    theta = part[5]
    phi = part[6]
    t_body = part[7:10]

    # print("plotting thinks that c_nose is {}".format(c_nose))
    # print("plotting thinks that c_mid is {}".format(c_mid))

    h_hip,h_nose,h_impl = None,None,None
    # We have to plot two ellipses

    # FIRST PLOT THE ELLIPSE, which is the hip
    # generate points on a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    # get the mesh, by using the equation of an ellipsoid
    x=np.cos(u)*a_hip
    y=np.sin(u)*np.sin(v)*b_hip
    z=np.sin(u)*np.cos(v)*b_hip

    # pack to matrix of positions
    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # apply the rotatation and unpack
    # posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
    posi_rotated = (R_body @ posi + c_hip[:,np.newaxis])

    # posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T

    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    # reshape for wireframe
    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

    h_hip = ax.plot_wireframe(x, y, z, color=color,alpha = plot_alpha)

    # THEN PLOT THE ELLIPSE, which is the nose
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    x=np.cos(u)*a_nose
    y=np.sin(u)*np.sin(v)*b_nose
    z=np.sin(u)*np.cos(v)*b_nose

    posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

    # kind of old, but w/e
    # R_head = rotate_body_model(psi,theta,phi)

    # posi_rotated = ((R_body @ ( (R_head @ posi).T + c_nose).T ).T + t_body).T
    # posi_rotated = R_body @ ((R_head @ (posi.T + c_nose).T ).T + t_body).T
    # posi_rotated = ((R_nose @ (posi.T ).T ).T + t_body)

    posi_rotated = (R_nose @ posi + c_nose[:,np.newaxis])

    x = posi_rotated[0,:]
    y = posi_rotated[1,:]
    z = posi_rotated[2,:]

    x = np.reshape(x, (u.shape) )
    y = np.reshape(y, (u.shape) )
    z = np.reshape(z, (u.shape) )

#    h_nose = ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)
    h_nose = ax.plot_wireframe(x, y, z, color='green',alpha = plot_alpha)
    if implant:

        # THEN PLOT THE ELLIPSE, which is the nose
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

        x=np.cos(u)*r_impl
        y=np.sin(u)*np.sin(v) * r_impl
        z=np.sin(u)*np.cos(v) *r_impl

        posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

        posi_rotated = (R_nose @ posi + c_impl[:,np.newaxis])

        # kind of old, but w/e

        # posi_rotated = ((R_body @ ( (R_head @ posi).T + c_impl).T ).T + t_body).T
        #    posi_rotated = ((R_nose @ (posi.T + c_nose).T ).T + t_body).T

        x = posi_rotated[0,:]
        y = posi_rotated[1,:]
        z = posi_rotated[2,:]

        x = np.reshape(x, (u.shape) )
        y = np.reshape(y, (u.shape) )
        z = np.reshape(z, (u.shape) )

    #    h_nose = ax.plot_wireframe(x, y, z, color=color,alpha = 0.7)
        h_impl = ax.plot_wireframe(x, y, z, color='blue',alpha = plot_alpha)

#%%


def close_4d(ax,positions):
    """
    The positions keyword is a bit silly, but this is just used to estimate
    the min and max of the axes, so that all are visible
    """
    # ax.set_aspect('equal')

    X,Y,Z = positions[:,0], positions[:,1], positions[:,2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5


    max_range = .3
    mid_x = 0.
    mid_y = 0.
    mid_z = 0.

    ax.set_xlim(-.3,.3)
    ax.set_ylim(-.3,.3)
    ax.set_zlim(0,.6)


    ax.set_xlabel('x (mm)',fontsize=16)
    ax.set_ylabel('y (mm)',fontsize=16)
    zlabel = ax.set_zlabel('z (mm)',fontsize=16)




def plot_particles(ax,particles,positions,body_constants,alpha = 0.1,keyp = None,ikeyp = None,body_supports=None,single = False):
    if len(particles.shape) == 1:
        particles = np.tile(particles,(1,1))
    if single:
        particles = particles[:,:10]
    # print(particles.shape)

    n_particles = particles.shape[0]
    # plot the particle mice!

    # adjust the bottom!
    scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='k',alpha=.2,marker='o',s=6)
    for i in range(n_particles):
        x_fit = particles[i,:]
        add_torch_mouse_for_video(ax,x_fit,body_constants,color = 'r',plot_alpha = alpha,implant=True)
        if len(x_fit) > 10:
            x_fit = x_fit[10:]
            add_torch_mouse_for_video(ax,x_fit,body_constants,color = 'r',plot_alpha = alpha,implant=False)

    # close_4d(ax,positions)
    fz = 10
    #  ax.set_xlabel('x (mm)',fontsize=6)
    #   ax.set_ylabel('y (mm)',fontsize=6)
    #    zlabel = ax.set_zlabel('z (mm)',fontsize=6)

    if keyp is not None:
        body_colors = ['dodgerblue','red','lime','orange']
        for i,body in enumerate(ikeyp.numpy()):
            ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[int(body)],rasterized=True)

    if body_supports is not None:
        for pp in body_supports[0][:]:
            pp = pp[0,:,0].numpy()

            ax.scatter(pp[0], pp[1], pp[2], zdir='z', s=100, c='k',rasterized=True)
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl = body_supports[0]

        ax.plot([c_mid[0,0,0],c_ass[0,0,0]],[c_mid[0,1,0],c_ass[0,1,0]],[c_mid[0,2,0],c_ass[0,2,0]],c='k')
        ax.plot([c_mid[0,0,0],c_nose[0,0,0]],[c_mid[0,1,0],c_nose[0,1,0]],[c_mid[0,2,0],c_nose[0,2,0]],c='k')
        ax.plot([c_mid[0,0,0],c_tip[0,0,0]],[c_mid[0,1,0],c_tip[0,1,0]],[c_mid[0,2,0],c_tip[0,2,0]],c='k')
        ax.plot([c_impl[0,0,0],c_nose[0,0,0]],[c_impl[0,1,0],c_nose[0,1,0]],[c_impl[0,2,0],c_nose[0,2,0]],c='k')
        ax.plot([c_impl[0,0,0],c_tip[0,0,0]],[c_impl[0,1,0],c_tip[0,1,0]],[c_impl[0,2,0],c_tip[0,2,0]],c='k')

        for pp in body_supports[1][:5]:
            pp = pp[0,:,0].numpy()
            ax.scatter(pp[0], pp[1], pp[2], zdir='z', s=100, c='k',rasterized=True)
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl = body_supports[1]

        ax.plot([c_mid[0,0,0],c_ass[0,0,0]],[c_mid[0,1,0],c_ass[0,1,0]],[c_mid[0,2,0],c_ass[0,2,0]],c='k')
        ax.plot([c_mid[0,0,0],c_nose[0,0,0]],[c_mid[0,1,0],c_nose[0,1,0]],[c_mid[0,2,0],c_nose[0,2,0]],c='k')
        ax.plot([c_mid[0,0,0],c_tip[0,0,0]],[c_mid[0,1,0],c_tip[0,1,0]],[c_mid[0,2,0],c_tip[0,2,0]],c='k')


    ax.xaxis.label.set_size(fz)
    ax.yaxis.label.set_size(fz)
    ax.zaxis.label.set_size(fz)



def plot_fitted_mouse(positions,x0_start,best_mouse, keyp = None, ikeyp = None,body_supports=None):
    # the winning mouse is the one, with the lowest final loss
    #end_loss = [np.mean(ll[-1:]) for ll in ll_holder]

    #best_idx = np.argmin(end_loss)
    #best_mouse = best_holder[best_idx]

    which_opt = 0
    opt_names = ['geoLM']
    # i_frame = 0
    #fig,ax = open_3d()
    fig = plt.figure(figsize=(15,7.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plot_particles(ax,x0_start,positions,body_constants,alpha = .5,keyp = keyp, ikeyp = ikeyp,body_supports=body_supports)
    ax.set_title("Initial clicked mouse")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_particles(ax,best_mouse,positions,body_constants,alpha = .5,keyp = keyp, ikeyp = ikeyp,body_supports=body_supports)
    ax.set_title("After fitting w. "+opt_names[which_opt])


    plt.show()








class rls_bank:
    def __init__(self,embedding = 9):
        # try to make everything [batch x embedding], i.e. [n_vars X embedding X ...]
        self.embedding = embedding
        self.mu = 0.99
        self.eps = 0.1
        self.n_vars = 20

        self.w = torch.zeros((self.n_vars,self.embedding))
        # by convention (I think?) the most recent is on the left
        # self.w[:,0] += 1.

        single_R = 1/self.eps * torch.eye(self.embedding)
        single_R = single_R.reshape((1, self.embedding, self.embedding))
        self.R = single_R.repeat(self.n_vars, 1, 1)
        # and make a stanck
        self.Rnp = 1/self.eps * np.eye(self.embedding)


    def adapt(self,d,x):
        """
        Adapt weights according one desired value and its input.
        **Args:**
        * `d` : desired value (float)
        * `x` : input array (1-dimensional array)
        """
        # start by calculating the
        # wnp = np.zeros(len(xnp))
        # wnp[0] =1.
        # ynp = np.dot(wnp, xnp)
        # y = torch.dot(self.w[0,:], x[0,:])
        y = torch.einsum('ij,ij->i',(self.w,x))
        # calculate the error
        # enp = dnp - ynp
        e = d - y
        # calculate the R
        # R1 = np.dot(np.dot(np.dot(self.Rnp,xnp),xnp.T),self.Rnp)
        # iiner
        # np.dot(self.Rnp,xnp)
        #innermost
        R1coeff = torch.einsum('ij,ij->i' , (torch.einsum('ijk,ik->ik',(self.R,x)), x) )
        # use broadcasting to multiply each of the batched vectors with the coefficien
        R1 = R1coeff.unsqueeze(1).unsqueeze(2) * self.R
        # R2 = self.mu + np.dot(np.dot(xnp,self.Rnp),xnp.T)
        R2 = self.mu + torch.einsum('ai,ai->a' , ( torch.einsum('ai,aij->ai' , (x,self.R)), x ) )
        # now, we can update R, again use the unsqueezing trikc
        self.R = 1/self.mu * (self.R - R1/R2.unsqueeze(1).unsqueeze(2) )
        # and calculate the change in w
        # dw = np.dot(self.Rnp, xnp.T) * e
        dw = torch.einsum('aij,ai->ai' , (self.R,x) ) * e.unsqueeze(1)
        self.w += dw

    def predict(self, x):
        """
        This function calculates the new output value `y` from input array `x`.
        **Args:**
        * `x` : input vector (1 dimension array) in length of filter.
        **Returns:**
        * `y` : output value (float) calculated from input array.
        """
        # y = np.dot(self.w, x)
        y = torch.einsum('ij,ij->i',(self.w,x))
        return y
