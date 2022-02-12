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
a_hip_min = 0.01/2 #.01m
a_hip_max = 0.05/2 #.055m
b_hip_min = 0.035/2 #.03m
b_hip_max = 0.04/2 #.035m, was 0.046, which was too much

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
b_nose = torch.Tensor([body_scale*0.035/2]).to(device) #m was.3

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


def make_xyz_rotation(alpha,beta,gamma,one,zero):
    # helper function
    # makes a rotation matrix, only around y and z angles
    # first, calcul,ate
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # makes a rotation matrix, only around y and z angles
    rot_alpha = torch.stack([torch.stack([one,   zero,      zero], dim=1),
                          torch.stack([zero,       cos_alpha,       -sin_alpha],      dim=1),
                          torch.stack([zero,  sin_alpha,      cos_alpha], dim=1)], dim=1)
    rot_beta = torch.stack([torch.stack([cos_beta,   zero,      sin_beta], dim=1),
                      torch.stack([zero,       one,       zero],      dim=1),
                      torch.stack([-sin_beta,  zero,      cos_beta], dim=1)], dim=1)
    rot_gamma = torch.stack([torch.stack([cos_gamma,  -sin_gamma, zero],      dim=1),
                      torch.stack([sin_gamma,  cos_gamma,  zero],      dim=1),
                      torch.stack([zero,       zero,       one],       dim=1)], dim=1)

    # now, these are also n-particles x 3 x 3, or batchzie x 3 x 3 in tf lingo
    # do batch-wise matrix multiplication with einsum
    rot_xy = torch.einsum('aij,ajk->aik',[rot_beta,rot_gamma])
    rot_xyz = torch.einsum('aij,ajk->aik',[rot_alpha,rot_xy])
    return rot_xyz


def rotation_matrix_vec2vec(f,t):
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

    # rotate f onto t
    # very fast, but slightly numerically unstable, so we add epsilon!
    epsilon = 1e-6
    # f = x_pointer
    # t = nose_pointer
    # cross product
    v = torch.cross(f,t)
    u = v/(torch.norm(v,dim=1).unsqueeze(1) + epsilon)
    # dot product
    c = torch.einsum('ai,ai->a', [f,t])
    # the factor h
    h = (1 - c)/(1 - c**2 + epsilon)

    vx, vy, vz = v[:,0],v[:,1],v[:,2]

    R = torch.stack([torch.stack([c + h*vx**2, h*vx*vy - vz, h*vx*vz + vy], dim=1),
                         torch.stack([h*vx*vy+vz, c+h*vy**2, h*vy*vz-vx],      dim=1),
                         torch.stack([h*vx*vz - vy, h*vy*vz + vx, c+h*vz**2], dim=1)], dim=1)

    return R


def particles_to_distance(part,pos,implant = False):
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

    pos = torch.unsqueeze(pos,0).transpose(-2,-1)

    # now we can just subtract, and torch will broadcast automatically
    # now the points are n_particles x n_points x 3 spatial dimensions
    # TODO optimize this from the beginning to avoid the transposition!
    p_hip = pos-c_hip
    p_nose = pos-c_nose

    # Make the matrices for the ellipsoids, nose is always the same
    aa = 1./a_nose**2
    bb = 1./b_nose**2
    Q_inner = torch.diagflat(torch.stack([aa,bb,bb]))
    R_nose = torch.einsum('aij,ajk->aik',[R_body,R_head])
    Q_nose = torch.einsum('aij,akj->aik', [torch.einsum('aij,jk->aik', [R_nose ,Q_inner] ),R_nose ] )

    # probably a faster way: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560/2
    # this uses ztriding to make the batch
    aa = 1./a_hip**2
    bb = 1./b_hip**2
    # my own custom bactching
    #     Q_inner = batch_diagonal(torch.stack([aa,bb,bb],dim=1))
    # they added batching now:
    Q_inner = torch.diag_embed(torch.stack([aa,bb,bb],dim=1))
    
    # now, we go over the hips, remember to batch
    Q_hip = torch.einsum('aij,akj->aik', [torch.einsum('aij,ajk->aik', [R_body ,Q_inner] ),R_body ] )

    # inner prduct between the position and Q
    delta_hip_signed = ( 1. - 1./torch.sqrt( torch.sum( p_hip *( Q_hip @ p_hip ) , dim =1) ) ) * torch.norm(p_hip,dim = 1)
    delta_nose_signed = ( 1. - 1./torch.sqrt( torch.sum( p_nose *( Q_nose @ p_nose ) , dim =1) ) ) * torch.norm(p_nose,dim = 1)

    # we're done!

    dist = torch.min(torch.abs(delta_hip_signed),torch.abs(delta_nose_signed))
    unsigned_dist = torch.clone(dist)
    if implant:
        # collected
        p_impl = pos-c_impl
        delta_impl = torch.norm(p_impl,dim = 1) - r_impl
        dist = torch.min(dist,torch.abs(delta_impl) )

    body_support = [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
    return dist,unsigned_dist,body_support




#%% Helpers to CALCULATE THE CENTER FROM THE particles_to_distance

def unsigned_residual(part,pos,overlap_penalty = False):
    _, dist0, _ = particles_to_distance(part[:,:9],pos,implant = True)
    _, dist1,_ = particles_to_distance(part[:,9:],pos)
    r = torch.min(dist0,dist1)
    if overlap_penalty:
        r = r + ball_cost(part)
    return r


def add_implant_residual(r,keyp,ikeyp,body_support_0, setpoint = 0.0135, scaling = 1.):
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    c_impl = body_support_0[5][...,0]
    keyp_implant = (ikeyp == 0)

    n_keyp = sum(keyp_implant)
    if n_keyp > 0:
        # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)
        #         print(r_implant)
        # r = torch.cat([r,r_implant],dim=1)
        r = r+torch.mean(r_implant,dim=1).unsqueeze(1)
    return r / (1.+scaling)


def add_body_residual(r,keyp,ikeyp,body_support_0,body_support_1,bpart = 'ass', setpoint = 0.0, scaling = 10.):
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    if bpart == 'ear':
        which_keyp = 1
        which_support = 3
    elif bpart == 'nose':
        which_keyp = 2
        which_support = 4
    elif bpart == 'ass':
        which_keyp = 3
        which_support = 1

    c_impl = torch.cat(( body_support_0[which_support][...,0], body_support_1[which_support][...,0]))

    keyp_implant = (ikeyp == which_keyp)
    n_keyp = sum(keyp_implant)
    if n_keyp > 0:
    # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance
        keypoint_distance = torch.min(keypoint_distance[:r.shape[0],:], keypoint_distance[r.shape[0]:,:])

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)

        # r = torch.cat([r,r_implant],dim=1)
        r = r+torch.mean(r_implant,dim=1).unsqueeze(1)
    return r / (1.+scaling)

def add_ass_residual(r,keyp,ikeyp,body_support_0,body_support_1,which_keyp = 3, setpoint = 0.0, scaling = 10.):
    #     # stack on first dim?
    #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
    c_impl = torch.cat(( body_support_0[1][...,0], body_support_1[1][...,0]))

    keyp_implant = (ikeyp == which_keyp)
    n_keyp = sum(keyp_implant)
    if n_keyp > 0:
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
    c_impl = torch.cat(( body_support_0[3][...,0], body_support_1[3][...,0]))

    keyp_implant = (ikeyp == which_keyp)
    n_keyp = sum(keyp_implant)
    if n_keyp > 0:
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

    n_keyp = sum(keyp_implant)
    if n_keyp > 0:

        # these are n x 3, i.e. n x xyz
        target_keyp = keyp[keyp_implant,:]
        keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
        # get the smallest distance
        keypoint_distance = torch.min(keypoint_distance[:r.shape[0],:], keypoint_distance[r.shape[0]:,:])

        r_implant = scaling * torch.abs(keypoint_distance - setpoint)

        r = torch.cat([r,r_implant],dim=1)
    return r

def ball_cost(part,body_support_0,body_support_1):
    '''
    A function which takes particles and returns an L2 loss on the amount of overlap of the balls
    '''
    c_hip_0,c_ass_0,c_mid_0,c_nose_0,c_tip_0,c_impl_0,R_body,R_head,R_nose = body_support_0
    c_hip_1,c_ass_1,c_mid_1,c_nose_1,c_tip_1,c_impl_1,R_body,R_head,R_nose = body_support_1

    s = part[:,2]
    a_hip_00 = a_hip_0 + a_hip_delta * s
    b_hip_00 = b_hip_0 + b_hip_delta * (1.-s)

    s = part[:,11]
    a_hip_01 = a_hip_0 + a_hip_delta * s
    b_hip_01 = b_hip_0 + b_hip_delta * (1.-s)

    # first, we calculate the distances betjupween the centers of the ellipsoids
    # nose2nose
    d_n0n1 = torch.norm(c_nose_0-c_nose_1,dim=1)
    # nose2hip
    d_n0h1 = torch.norm(c_nose_0-c_hip_1,dim=1)
    d_n1h0 = torch.norm(c_nose_1-c_hip_0,dim=1)
    # hip2hip
    d_h0h1 = torch.norm(c_hip_0-c_hip_1,dim=1)
    # implant to other's nose
    d_imp0n1 = torch.norm(c_impl_0-c_nose_1,dim=1)
    # implant to other's hip
    d_imp0h1 = torch.norm(c_impl_0-c_hip_1,dim=1)

    # make a list of the actual distance between the centers
    d_actual = torch.stack([d_n0n1,d_n0h1,d_n1h0,d_h0h1,d_imp0n1,d_imp0h1]).squeeze(2)

    # make a list of the minimum allowed distance between
    cutoff_barrier = 0.8*torch.stack([(b_nose + b_nose)*torch.ones_like(b_hip_01), b_nose+b_hip_01, b_nose+b_hip_00, b_hip_00+b_hip_01, (r_impl + b_nose)*torch.ones_like(b_hip_01), r_impl+b_hip_01 ])

    # clip the overlap
    overlap = torch.clamp(cutoff_barrier-d_actual,0.,None)
    # do a kind of L2 loss, which we add everywhere
    barrier_loss = torch.mean(overlap,dim=0)

    return barrier_loss

def residual(part,pos,keyp,ikeyp,overlap_penalty = False,clip=True):
    dist0,_,body_support_0 = particles_to_distance(part[:,:9],pos,implant = True)
    dist1,_,body_support_1 = particles_to_distance(part[:,9:],pos)
    r = torch.min(dist0,dist1)

    if clip:
        r = torch.clamp(r,0,.04)

    r = add_implant_residual(r,keyp,ikeyp,body_support_0, setpoint = 0.0135, scaling = 0.2)
    r = add_body_residual(r,keyp,ikeyp,body_support_0,body_support_1,bpart = 'ass', setpoint = 0.0, scaling = 0.1)
    r = add_body_residual(r,keyp,ikeyp,body_support_0,body_support_1,bpart = 'nose', setpoint = 0.0, scaling = 0.05)
    r = add_body_residual(r,keyp,ikeyp,body_support_0,body_support_1,bpart = 'ear', setpoint = 0.01, scaling = 0.1)

    if overlap_penalty:
        overlap_scaling = 1
        bc = ball_cost(part,body_support_0,body_support_1) #.unsqueeze(0).transpose(0,1)
        if bc.shape[0] == 1:
            r = ( r + bc ) / overlap_scaling
        else:
            r = ( r + bc.unsqueeze(0).transpose(0,1) ) / overlap_scaling

    return r

def jacobian_approx(part,pos,keyp,ikeyp):
    # TODO some rescaling maybe. Changes in xyz are sort of on the order of
    # 10 times smaller than changes in beta gamma s theta phi

    # takes the point, and calculates the jacobian around it
    # first, we make a tensor, no gradients here
    #part = torch.tensor(x0_start,dtype = torch.float32,requires_grad = False).to(device).unsqueeze(0)
    # we do forward approximation, so we need to add epsilon to all parameters
    epsilon = torch.tensor(1e-5)
    # now we can use broadcasting to generate the test points:
    part_test = epsilon * torch.eye(part.shape[1]) + part

    # concatenate the real parameters onto here:
    # part: each row is a particle, row 0 is the real residual
    part = torch.cat([part,part_test])

    # now, calcuate the residuals for these parlicles:
    r_all = residual(part,pos,keyp,ikeyp)

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


# search_cone = torch.Tensor(np.tile([.2,.2,.2,.1,.2,.2,.2,.01,.01,.01],2)).to(device).unsqueeze(0)
# search_cone = search_cone[:,[1,2,3,4,5,6,7,8,9,11,12,13,15,16,17,18,19]]
search_cone = torch.Tensor([.2,.2,.1,.2,.2,6.0,.01,.01,.01,.2,.2,.1,.2,6.0,.01,.01,.01]).unsqueeze(0)


def klm_routine(part,pos,keyp,ikeyp,max_iters = 100,verbose=False,save_history = True,geodesic= False,ftol = 1e-4, search_cone = search_cone):

    # for limits
    upper_limit = part + search_cone
    lower_limit = part - search_cone
    # global_upper_limit =
    # global_lower_limit =

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
        i += 1

        if verbose:
            print(i)

        # calc g and nablaC
        # g = j.T.dot(j) + l*I
        g = torch.einsum('ji,jk->ik',[J,J]) + torch.diagflat(lamb * torch.ones_like(part))
        # gradC = j.T.dot(r)
        nablaC = torch.einsum('ji,j->i',[J,r])

        # solve gg x = gradC, d1 = -x
        # -d1 = solve(gg,gradC)

        # SAFE = 0.5
        d1, LU1 = torch.solve(nablaC.unsqueeze(1),g) # wierdly it wants
        d1 = - d1[:,0]

        # print(d1)

        # xnew = x - SAFE*solve(g,gradC)
        # x_new = part + SAFE * d1
        # clamp the gradient

        # d1 = torch.clamp(d1,min=-.2, max = .2)

        x_new = part + d1

        # CLIP the s
        #
        # x_new = torch.max(torch.min(x_new,upper_limit),lower_limit)

        # we have global limits on the head angle and s
        x_new[0,[4,12]] = torch.clamp(x_new[0,[4,12]],min=-3.14/4., max = 3.14/4)
        x_new[0,[2,11]] = torch.clamp(x_new[0,[2,11]],min=0, max = 1)



        r_new = residual(x_new,pos,keyp,ikeyp)[0,:]
        C_new = 0.5 * torch.dot(r_new,r_new)

        # what will the change be?
        deltaC = C_new - C

        if verbose:
            # print()
            print("deltaC: {:.12f}, C_old: {:.12f}, C_new: {:.12f}".format(deltaC.numpy(),C.numpy(),C_new.numpy()) )


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

        if np.log(lamb) > 100 or (deltaC < 0 and np.abs(deltaC) < ftol and i > 5):
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




def add_torch_mouse_for_video(ax,part,body_constants,body_support,color = 'r',plot_alpha = .7, implant=False):
    # this also need a vector
    # get the geometry of the mouse body # not really the preferred way

    body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose,b_nose,d_nose,x_impl,z_impl,r_impl = body_constants


    c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_support
    c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = c_hip[0,...].numpy(),c_ass[0,...].numpy(),c_mid[0,...].numpy(),c_nose[0,...].numpy(),c_tip[0,...].numpy(),c_impl[0,...].numpy(),R_body[0,...].numpy(),R_head[0,...].numpy(),R_nose[0,...].numpy()


    if implant:
        beta = part[0]
        gamma = part[1]
        s = part[2]
        #todo naming here is off
        psi = part[3]
        theta = part[4]
        phi = part[5]
        t_body = part[6:9]
    else:
        beta = part[0]
        gamma = part[1]
        s = part[2]
        #todo naming here is off
        theta = part[3]
        phi = part[4]
        t_body = part[5:8]

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
    posi_rotated = np.einsum('ij,ja->ia',R_body,posi) + c_hip

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

    posi_rotated =   np.einsum('ij,ja->ia',R_nose,posi)  + c_nose

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

        posi_rotated = np.einsum('ij,ja->ia',R_nose,posi)  + c_impl

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
    mid_x = 0.05
    mid_y = -0.15
    mid_z = 0.

    ax.set_xlim(-.10,.20)
    ax.set_ylim(-.3,0)
    ax.set_zlim(0,.3)


    ax.set_xlabel('x (mm)',fontsize=16)
    ax.set_ylabel('y (mm)',fontsize=16)
    zlabel = ax.set_zlabel('z (mm)',fontsize=16)



def plot_particles_new_nose(ax,particles,positions,body_constants,body_supports,alpha = 0.1,keyp = None,ikeyp = None,single = False):
    if len(particles.shape) == 1:
        particles = np.tile(particles,(1,1))
    if single:
        particles = particles[:,:10]
    # print(particles.shape)

    n_particles = particles.shape[0]
    # plot the particle mice!

    # adjust the bottom!
    scat = ax.scatter(positions[:,0],positions[:,1],positions[:,2],c='k',alpha=.1,marker='o',s=3)
    for i in range(n_particles):
        x_fit = particles[i,:]
        add_torch_mouse_for_video(ax,x_fit,body_constants,body_supports[0],color = 'r',plot_alpha = alpha,implant=True)
        if len(x_fit) > 10:
            x_fit = x_fit[9:]
            add_torch_mouse_for_video(ax,x_fit,body_constants,body_supports[1],color = 'r',plot_alpha = alpha,implant=False)

    close_4d(ax,positions)
    fz = 10
    #  ax.set_xlabel('x (mm)',fontsize=6)
    #   ax.set_ylabel('y (mm)',fontsize=6)
    #    zlabel = ax.set_zlabel('z (mm)',fontsize=6)

    if keyp is not None:
        body_colors = ['dodgerblue','red','lime','orange']
        for i,body in enumerate(ikeyp.numpy()):
            ax.scatter(keyp[i,0], keyp[i,1], keyp[i,2], zdir='z', s=100, c=body_colors[int(body)],rasterized=True)

    if body_supports is not None:
        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_supports[0]

        for p in [c_hip,c_mid,c_nose,c_ass,c_tip,c_impl]:
            ax.scatter(p[:,0],p[:,1],p[:,2],zdir='z', s=100, c='k',rasterized=True)
        for p,q in zip([c_nose,c_nose,c_mid,c_impl,c_impl],[c_mid,c_tip,c_ass,c_nose,c_tip]):
            ax.plot([p[0,0],q[0,0]],[p[0,1],q[0,1]],[p[0,2],q[0,2]],'k')

        c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = body_supports[1]


        for p in [c_hip,c_mid,c_nose,c_ass,c_tip]:
            ax.scatter(p[0,0],p[0,1],p[0,2],zdir='z', s=100, c='k',rasterized=True)
        for p,q in zip([c_nose,c_nose,c_mid],[c_mid,c_tip,c_ass]):
            ax.plot([p[0,0],q[0,0]],[p[0,1],q[0,1]],[p[0,2],q[0,2]],'k')


    ax.xaxis.label.set_size(fz)
    ax.yaxis.label.set_size(fz)
    ax.zaxis.label.set_size(fz)



def plot_fitted_mouse_new_nose(positions,x0_start,best_mouse, keyp = None, ikeyp = None,body_supports=None):
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
    plot_particles_new_nose(ax,x0_start,positions,body_constants,body_supports,alpha = .5,keyp = keyp, ikeyp = ikeyp)
    ax.set_title("Initial clicked mouse")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_particles_new_nose(ax,best_mouse,positions,body_constants,body_supports,alpha = .5,keyp = keyp, ikeyp = ikeyp)
    ax.set_title("After fitting w. "+opt_names[which_opt])


    plt.show()



def geolm_convergence_single(history):
    h_C, h_C_new, h_r, h_lamb, h_part,h_step = history[:]
    # plt.close('all')
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

    part_history.shape

    plt.subplot(6,1,4)
    plt.ylabel("xyz history")
    plt.plot(part_history[:,[6,7,8]],'o:')

    plt.subplot(6,1,5)
    plt.ylabel("spine history")
    angle_history = part_history[:,[2]]
    plt.plot(angle_history,':o')


    plt.subplot(6,1,6)
    plt.ylabel("angle history")
    angle_history = part_history[:,[0,1,3,4,5]]
    plt.plot(angle_history,':o')


    for i in range(6):
        plt.subplot(6,1,i+1)
        plt.xlim((0,len(h_C)))

    plt.show()






class rls_bank:
    def __init__(self, n_vars = 17, embedding = 9):
        # try to make everything [batch x embedding], i.e. [n_vars X embedding X ...]
        self.embedding = embedding
        self.mu = 0.99
        self.eps = 0.1
        self.n_vars = n_vars

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
