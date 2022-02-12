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




def unpack_from_jagged(jagged_line):
    ''' THE REVESER SO HERE IT UNPACKS AGAIN SO THE DATA CAN BE SAVED
    AS A JAGGED H5PY DATASET 
    FROM OTHER: Takes the NX3, N, Mx3, M, M shapes and packs to a single float16
    We ravel the position, ravel the keyp, stack everything and 
    - importantly - we also save M, the number of keypoints'''
    n_keyp = int(jagged_line[-1])
    keyp_idx2 = jagged_line[-(1+n_keyp):-1].astype('int')
    pkeyp2 = jagged_line[-(1+2*n_keyp):-(1+n_keyp)]
    keyp2 = jagged_line[-(1+5*n_keyp):-(1+2*n_keyp)].reshape((n_keyp,3))
    block2 = jagged_line[:-(1+5*n_keyp)].reshape((-1,4))
    pos2,pos_weights2 = block2[:,:3], block2[:,3]
    # HACK to cut the floor
    floor_logic = pos2[:,2] > .012
    pos2 = pos2[floor_logic,:]
    pos_weights2 = pos_weights2[floor_logic]
    
    return pos2, pos_weights2, keyp2, pkeyp2, keyp_idx2





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
plt.close('all')

#############
# SOME GEOMETRY 
#############


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





#%% ####################################
# Init the variables
#####################################

# where to put the model?
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# put the constants for the mouse bodies onto the gpu!
body_scale =1.
## HIP is a prolate ellipsoid, centered along the x axis
a_hip_min = 0.01/2 #.01m
a_hip_max = 0.05/2 #.055m
b_hip_min = 0.035/2 #.03m
b_hip_max = 0.04/2 #.035m, was 0.046, which was too much

# converting it to the new terminology
a_hip_0     = torch.Tensor([body_scale*a_hip_min ]).to(torch_device)#m
a_hip_delta = torch.Tensor([body_scale*(a_hip_max - a_hip_min)] ).to(torch_device)#m
b_hip_0     = torch.Tensor([body_scale*b_hip_min ]).to(torch_device)#m
b_hip_delta = torch.Tensor([body_scale*(b_hip_max - b_hip_min)] ).to(torch_device)#m


## NOSE is prolate ellipsoid, also along the head direction vector
# here, there is no re-scaling
a_nose = torch.Tensor([body_scale*0.045/2]).to(torch_device)#m was .04
b_nose = torch.Tensor([body_scale*0.025/2]).to(torch_device) #m

a_nose = torch.Tensor([body_scale*0.028/2]).to(torch_device)#m was .04
b_nose = torch.Tensor([body_scale*0.018/2]).to(torch_device) #m

a_nose = torch.Tensor([body_scale*0.04/2]).to(torch_device)#m long axis was .04
b_nose = torch.Tensor([body_scale*0.035/2]).to(torch_device) #m was.3

d_nose = torch.Tensor([body_scale*0.01]).to(torch_device) #m

r_impl = 1.1*b_nose
x_impl = 1.* d_nose+.7*a_nose
z_impl = 1.5* r_impl# .0+0*1.5*r_impl

r_impl = 0.9*b_nose
x_impl = 1.* d_nose+.5*a_nose
z_impl = 1.5* r_impl# .0+0*1.5*r_impl

# make a list of the body constants to pass and save!
body_constants = np.asanyarray([body_scale,a_hip_min,a_hip_max,b_hip_min,b_hip_max,a_nose.cpu().numpy(),b_nose.cpu().numpy(),d_nose.cpu().numpy(),x_impl.cpu().numpy(),z_impl.cpu().numpy(),r_impl.cpu().numpy()]).astype('float32')


def particles_to_distance_cuda(part,pos,implant = False):

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
    Q_inner = torch.diag_embed(torch.stack([aa,bb,bb],dim=1))# now, we go over the hips, remember to batch
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

# dist0,_,body_support_0 = particles_to_distance_cuda(particles, positions)




def loading_wrapper(frame,jagged_lines):
    '''RETURNS THE CLOUD OF A FRAME as torch tensors
    NO DOWNSAMPLING OF POSITIONS
    '''
    pos, pos_weights, keyp, pkeyp, ikeyp = unpack_from_jagged(jagged_lines[frame])
    
    # downsample?
    pos = pos
    pos_weights = pos_weights

    # and convert to torch
    keyp  = torch.tensor(keyp).float().to(torch_device)
    ikeyp  = torch.tensor(ikeyp).float().to(torch_device)
    pos = torch.Tensor(pos).float().to(torch_device)
    pos_weights = torch.Tensor(pos_weights).float().to(torch_device)
    return pos,pos_weights,keyp,ikeyp

def unsigned_residual_cuda(part,pos,overlap_penalty = False):
    '''CALCULATE DISTANCE UNSIGNED'''
    _, dist0, _ = particles_to_distance_cuda(part[:,:9],pos,implant = True)
    _, dist1,_ = particles_to_distance_cuda(part[:,9:],pos)
    r = torch.min(dist0,dist1)
    if overlap_penalty:
        r = r + ball_cost(part)
    return r.squeeze()

def clean_keyp_by_r(part,keyp,ikeyp):
    '''Cleans the keypoints, if they are too distant
    might be unnescessary...
    USE 6 cm cutoff'''
    # also cut any keypoints which are par away!!
    r = unsigned_residual_cuda(part,keyp)
    ikeyp = ikeyp[r<0.06]
    keyp = keyp[r<0.06,:]
    return keyp,ikeyp



# local limits
abc_lim = .5
psi_lim = 1.
theta_lim = 3.14/4
phi_lim = 2.4
xy_lim = .05
z_lim = .05
s_lim = .3
search_cone = torch.Tensor([.2,.2,.1,.2,.2,1.6,.01,.01,.01,.2,.2,.1,.2,1.6,.01,.01,.01]).unsqueeze(0)
search_cone = torch.Tensor([abc_lim, abc_lim, s_lim, psi_lim, theta_lim, phi_lim, xy_lim,xy_lim,z_lim,       
                            abc_lim, abc_lim, s_lim, theta_lim, phi_lim, xy_lim,xy_lim,z_lim]).unsqueeze(0).to(torch_device)

# global limits
abc_max = float('inf') #1000
psi_max = float('inf') #1000
theta_max = 3.14 / 3
phi_max = float('inf') #1000
xy_max = float('inf') #1000
z_max = .07 #1000
s_max = 1.
global_max = torch.Tensor([abc_max, abc_max, s_max, psi_max, theta_max, phi_max, xy_max,xy_max,z_max,       
                           abc_max, abc_max, s_max, theta_max, phi_max, xy_max,xy_max,z_max]).unsqueeze(0).to(torch_device)

abc_min = -float('inf') #-1000
psi_min = -float('inf')#-1000
theta_min = -3.14 / 3
phi_min = -float('inf')#-1000
xy_min = -float('inf') #-1000
z_min = 0. #-1000
s_min = 0.
global_min = torch.Tensor([abc_min, abc_min, s_min, psi_min, theta_min, phi_min, xy_min,xy_min,z_min,       
                           abc_min, abc_min, s_min, theta_min, phi_min, xy_min,xy_min,z_min]).unsqueeze(0).to(torch_device)



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


def make_some_bounds(part,search_cone,global_max,global_min):
    upper_bound = torch.min(global_max,part+search_cone)
    lower_bound = torch.max(global_min,part-search_cone)
    return upper_bound[0,:],lower_bound[0,:]


class MousePFilt(object):
    def __init__(self,swarm_size=100,options=None):
        if (options == None):
            options = [2.,2.,0.,.01,100]
        self.swarm_size = swarm_size
        self.c_cognitive = options[0]
        self.c_social = options[1]
        self.inertia_weight = options[2]
        self.velocity_limit = options[3]
        self.max_iterations = options[4]
        self.loss_winner = 100000. # kind of hacky
        self.sorted_loss = None
        self.histo_mu = []
        self.histo_var = []
        self.histo_loss = []
        self.save_history = False
        self.state_history_pre_resample = []
        self.state_history_post_resample = []
        print("HELLO I'm a slow figure making version")
        self.figure_counter = None
        
        # sampling_cone
        abc_lim = .4
        psi_lim = .4
        theta_lim = 3.14/.25 # was /3
        phi_lim = .6 # wa .6
        xyz_lim = .02
        s_lim = .2
        self.sampling_cone_big = torch.Tensor([abc_lim, abc_lim, s_lim, psi_lim, theta_lim, phi_lim, xyz_lim,xyz_lim,xyz_lim,       
                                    abc_lim, abc_lim, s_lim, theta_lim, phi_lim, xyz_lim,xyz_lim,xyz_lim]).unsqueeze(0).to(torch_device)

        abc_lim = .2
        psi_lim = .2
        theta_lim = 3.14/5 
        phi_lim = .3
        xyz_lim = .01
        s_lim = .1
        self.sampling_cone_small = torch.Tensor([abc_lim, abc_lim, s_lim, psi_lim, theta_lim, phi_lim, xyz_lim,xyz_lim,xyz_lim,       
                                    abc_lim, abc_lim, s_lim, theta_lim, phi_lim, xyz_lim,xyz_lim,xyz_lim]).unsqueeze(0).to(torch_device)

        
    def search_space(self,upper_bound,lower_bound):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.dimensionality = upper_bound.size()[0]  
        
        self.velocity_limit = (self.upper_bound - self.lower_bound)*.2 # 5 pct of max?

    def populate(self,sobol = True):
        # populate some indices, which we will need again and again
        self.idx0,self.idx1 = torch.meshgrid(torch.arange(self.swarm_size),torch.arange(self.swarm_size))
        self.idx0_flat = self.idx0.contiguous().view(-1)
        self.idx1_flat = self.idx1.contiguous().view(-1)
        # now populate some random particles
        if sobol:
            # initialize a sobol engine to do this
            self.soboleng = torch.quasirandom.SobolEngine(dimension=self.dimensionality)
            self.position = ((self.upper_bound - self.lower_bound)*self.soboleng.draw(self.swarm_size).to(torch_device) ) + self.lower_bound
            self.velocity = (2*self.velocity_limit*torch.rand(self.swarm_size,self.dimensionality).to(torch_device) ) - self.velocity_limit        
            self.velocity = 0. * self.velocity
        else:
            self.position = ((self.upper_bound - self.lower_bound)*torch.rand(self.swarm_size,self.dimensionality).to(torch_device)  ) + self.lower_bound
            self.velocity = (2*self.velocity_limit*torch.rand(self.swarm_size,self.dimensionality).to(torch_device) ) - self.velocity_limit        


    def calc_loss_2d(self,use_weights = False):
        # def spread_parts(part,pos):
        dist0,_,self.body_support_0 = particles_to_distance_cuda(self.position[:,:9],self.pos[::5,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(self.position[:,9:],self.pos[::5,:])
        r = torch.min(dist0[self.idx0,:],dist1[self.idx1,:])
        if use_weights:
            w2 = self.pos_weights[::5].pow(2)
            r = ( w2 * r) / torch.median(w2)
            self.loss_2d = torch.mean(r,dim=2)  
        else:
            r = torch.clamp(r,0,.03)  
            self.loss_2d = torch.mean(r,dim=2)  
        
        if use_weights and False:
            print(self.loss_2d[:200])
        
    def calc_loss_2d_separately(self):
        '''HERE we just clip the distances at .03 first, individually '''
        # def spread_parts(part,pos):
        dist0,_,self.body_support_0 = particles_to_distance_cuda(self.position[:,:9],self.pos[::5,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(self.position[:,9:],self.pos[::5,:])
        r0 = torch.clamp(dist0,0,.03)
        r1 = torch.clamp(dist0,0,.03)
        
        self.loss_2d = torch.mean(r,dim=2)                    
        

    def calc_r_impl(self):
        # calculate the 2d loss sheet for the implant
        # get the c_impl out _ it's the 6th, so 5    
        #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
        c_impl = self.body_support_0[5][...,0]
        keyp_implant = (self.ikeyp == 0)
        setpoint = 0.015
        n_keyp = sum(keyp_implant)

        if n_keyp > 0:
            # these are n x 3, i.e. n x xyz
            target_keyp = self.keyp[keyp_implant,:]
            keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
            # get the distance from the 
            r_implant = torch.abs(keypoint_distance - setpoint)
            # do the average 
            r_implant = torch.mean(r_implant,dim=1)
        else:
            r_implant = torch.zeros(c_impl.shape[0]).to(torch_device)

        self.r_impl_2d = r_implant[self.idx0]

    def calc_r_body(self,bpart):
        # def add_body_residual(r,keyp,ikeyp,body_support_0,body_support_1,bpart = 'ass', setpoint = 0.0, scaling = 10.):
        #  [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl]
        if bpart == 'ear':
            which_keyp = 1
            which_support = 3
            setpoint = 0.0135
        elif bpart == 'nose':
            which_keyp = 2
            which_support = 4
            setpoint = 0.
        elif bpart == 'ass':
            which_keyp = 3
            which_support = 1
            setpoint = 0.

        c_impl = torch.cat(( self.body_support_0[which_support][...,0], self.body_support_1[which_support][...,0]))

        keyp_implant = (self.ikeyp == which_keyp)
        n_keyp = sum(keyp_implant)
        if n_keyp > 0:
        # these are n x 3, i.e. n x xyz
            target_keyp = self.keyp[keyp_implant,:]
            keypoint_distance = torch.norm( c_impl[:,np.newaxis,:] - target_keyp[np.newaxis,:,:] ,dim=2)
            # get the smallest distance
            keypoint_to_0 = keypoint_distance[:self.swarm_size,:]
            keypoint_to_1 = keypoint_distance[self.swarm_size:,:]
            keypoint_distance = torch.min( keypoint_to_0[self.idx0,...], keypoint_to_1[self.idx1,...])

            r_body = torch.abs(keypoint_distance - setpoint)
            r_body = torch.mean(r_body,dim=2)
        else:
            r_body = torch.zeros(self.swarm_size,self.swarm_size).to(torch_device)
            
        if bpart == 'ear':
            self.r_ear_2d = r_body
        elif bpart == 'nose':
            self.r_nose_2d = r_body
        elif bpart == 'ass':
            self.r_ass_2d = r_body
            
    def calc_barrier(self):
        '''
        A function which takes particles and returns an L2 loss on the amount of overlap of the balls
        '''
        c_hip_0,c_ass_0,c_mid_0,c_nose_0,c_tip_0,c_impl_0,R_body,R_head,R_nose = self.body_support_0
        c_hip_1,c_ass_1,c_mid_1,c_nose_1,c_tip_1,c_impl_1,R_body,R_head,R_nose = self.body_support_1

        s = self.position[:,2]
        a_hip_00 = a_hip_0 + a_hip_delta * s
        b_hip_00 = b_hip_0 + b_hip_delta * (1.-s)

        s = self.position[:,11]
        a_hip_01 = a_hip_0 + a_hip_delta * s
        b_hip_01 = b_hip_0 + b_hip_delta * (1.-s)

        # first, we calculate the distances betjupween the centers of the ellipsoids
        # nose2nose
        d_n0n1 = torch.norm(c_nose_0[self.idx0,...]-c_nose_1[self.idx1,...],dim=2)
        # nose2hip
        d_n0h1 = torch.norm(c_nose_0[self.idx0,...]-c_hip_1[self.idx1,...],dim=2)
        d_n1h0 = torch.norm(c_nose_1[self.idx1,...]-c_hip_0[self.idx0,...],dim=2)
        # hip2hip
        d_h0h1 = torch.norm(c_hip_0[self.idx0,...]-c_hip_1[self.idx1,...],dim=2)
        # implant to other's nose
        d_imp0n1 = torch.norm(c_impl_0[self.idx0,...]-c_nose_1[self.idx1,...],dim=2)
        # implant to other's hip
        d_imp0h1 = torch.norm(c_impl_0[self.idx0,...]-c_hip_1[self.idx1,...],dim=2)

        # make a list of the actual distance between the centers
        d_actual = torch.stack([d_n0n1,d_n0h1,d_n1h0,d_h0h1,d_imp0n1,d_imp0h1]).squeeze(3)

        # make a list of the minimum allowed distance between
        cutoff_barrier = 0.8*torch.stack([(b_nose + b_nose)*torch.ones_like(d_n0n1).squeeze(2), 
                                          b_nose+b_hip_01[self.idx1], 
                                          b_nose+b_hip_00[self.idx0], 
                                          b_hip_00[self.idx0]+b_hip_01[self.idx1], 
                                          (r_impl + b_nose)*torch.ones_like(d_n0n1).squeeze(2), 
                                          r_impl+b_hip_01[self.idx1]])

        # clip the overlap
        overlap = torch.clamp(cutoff_barrier-d_actual,0.,None)
        # do a kind of L2 loss, which we add everywhere
        self.barrier_2d = torch.mean(overlap,dim=0)

    def distance_between_winner(self):
        '''
        A function which takes particles and returns an L2 loss on the amount of overlap of the balls
        '''
        c_hip_0,c_ass_0,c_mid_0,c_nose_0,c_tip_0,c_impl_0,R_body,R_head,R_nose = self.body_support_0
        c_hip_1,c_ass_1,c_mid_1,c_nose_1,c_tip_1,c_impl_1,R_body,R_head,R_nose = self.body_support_1

        s = self.position[:,2]
        a_hip_00 = a_hip_0 + a_hip_delta * s
        b_hip_00 = b_hip_0 + b_hip_delta * (1.-s)

        s = self.position[:,11]
        a_hip_01 = a_hip_0 + a_hip_delta * s
        b_hip_01 = b_hip_0 + b_hip_delta * (1.-s)

        # first, we calculate the distances betjupween the centers of the ellipsoids
        # nose2nose
        d_n0n1 = torch.norm(c_nose_0[0,...]-c_nose_1[0,...],dim=0)
        print(d_n0n1.shape)
        # nose2hip
        d_n0h1 = torch.norm(c_nose_0[0,...]-c_hip_1[0,...],dim=0)
        d_n1h0 = torch.norm(c_nose_1[0,...]-c_hip_0[0,...],dim=0)
        # hip2hip
        d_h0h1 = torch.norm(c_hip_0[0,...]-c_hip_1[0,...],dim=0)
        # implant to other's nose
        d_imp0n1 = torch.norm(c_impl_0[0,...]-c_nose_1[0,...],dim=0)
        # implant to other's hip
        d_imp0h1 = torch.norm(c_impl_0[0,...]-c_hip_1[0,...],dim=0)

        # make a list of the actual distance between the centers
        d_actual = torch.stack([d_n0n1,d_n0h1,d_n1h0,d_h0h1,d_imp0n1,d_imp0h1]).squeeze(1)

        return d_actual
    
    def min_distance_between_mice(self):
        '''returns the minimim distance between the centers'''
        if self.meanwinner is not None:
            return torch.min(self.distance_between_winner())
        else:
            return torch.tensor(0.).to(torch_device)
        

    def update_loss_flat(self):
        self.calc_loss_2d()
        self.calc_r_impl()
        self.calc_r_body('nose')
        self.calc_r_body('ear')
        self.calc_r_body('ass')
#         self.calc_barrier()
        w_impl = .2
        w_nose = .1
        w_ear = .1
        w_ass = .1
        w_barrier = .1
        self.loss_flat = (self.loss_2d+w_impl*self.r_impl_2d+w_nose*self.r_nose_2d+
                    w_ear*self.r_ear_2d+w_ass*self.r_ass_2d).view(-1)
#         self.loss_flat = (self.loss_2d+w_impl*self.r_impl_2d+w_nose*self.r_nose_2d+
#                     w_ear*self.r_ear_2d+w_ass*self.r_ass_2d+w_barrier*self.barrier_2d).view(-1)
        
    def resample_max(self):
        # only cloud
        # loss_flat = self.loss_2d.view(-1)
        # todo, only sort the bottom x pct, that's all we care about
        self.sorted_loss, self.idx_sorted = torch.sort(self.loss_flat)
               
        good_loss = self.idx_sorted[:self.swarm_size]
        # resample mouse0
        keep_alive_0 = self.idx0_flat[good_loss]
        self.position[:,:9] = self.position[keep_alive_0,:9]
        # resample mouse1
        keep_alive_1 = self.idx1_flat[good_loss]
        self.position[:,9:] = self.position[keep_alive_1,9:]

        # update the body supports?
        self.body_support_0 = [pik[keep_alive_0,...] for pik in self.body_support_0]
        self.body_support_1 = [pik[keep_alive_1,...] for pik in self.body_support_1]
        
    def resample_wheel(self):
        # update the particles

        # idea from https://github.com/rlabbe/filterpy/
        weights = self.loss_flat/self.loss_flat.sum()
        
        split_positions = (torch.rand(1) + torch.arange(self.swarm_size).float()) / self.swarm_size

        indices = torch.zeros(self.swarm_size,dtype=torch.long)

        cumulative_sum = torch.cumsum(weights,dim=0)
        cumulative_sum[-1] = 1.

        i, j = 0, 0
        # faster than sorting
        while i < N:
            if split_positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        
        # resample mouse0
        keep_alive_0 = self.idx0_flat[indices]
        self.position[:,:9] = self.position[keep_alive_0,:9]
        # resample mouse1
        keep_alive_1 = self.idx1_flat[indices]
        self.position[:,9:] = self.position[keep_alive_1,9:]

        # update the body supports?
        self.body_support_0 = [pik[keep_alive_0,...] for pik in self.body_support_0]
        self.body_support_1 = [pik[keep_alive_1,...] for pik in self.body_support_1]
    
    def blow_up(self,style = 'big', multiplier = 1.):
        if style == 'big':
            self.position.add_( torch.randn(self.swarm_size,self.dimensionality).to(torch_device) * self.sampling_cone_big)
#             self.position.add_( (self.soboleng.draw(self.swarm_size).to(torch_device) - .5)*4*self.sampling_cone_big)        
            self.enforce_bounds()
        else:
            self.position.add_( torch.randn(self.swarm_size,self.dimensionality).to(torch_device) * self.sampling_cone_small* multiplier)
#             self.position.add_( (self.soboleng.draw(self.swarm_size).to(torch_device) - .5)*4*self.sampling_cone_small)
            self.enforce_bounds()
            
        # TODO this blow up is only here for debugging, waste of calculations
        dist0,_,self.body_support_0 = particles_to_distance_cuda(self.position[:,:9],self.pos[::5,:],implant = True)
        dist1,_,self.body_support_1 = particles_to_distance_cuda(self.position[:,9:],self.pos[::5,:])
        
    def flip_around(self):
        # flip around axis! add pi to 
        self.position[1::3,1].add_(3.14159)
        self.position[2::3,10].add_(3.14159)
        # TODO this blow up is only here for debugging, waste of calculations
#         dist0,_,self.body_support_0 = particles_to_distance(self.position[:,:9],self.pos[::5,:],implant = True)
#         dist1,_,self.body_support_1 = particles_to_distance(self.position[:,9:],self.pos[::5,:])
        
        
    def plot_status(self,reduce_mean=False,plot_ellipsoids=False,example_plot = True,starting=False,final=False):
        # update all, can be removed
        self.calc_loss_2d()
        self.calc_r_impl()
        self.calc_r_body('nose')
        self.calc_r_body('ear')
        self.calc_r_body('ass')

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        n_particles = self.position.shape[0]
        # plot the particle mice!

        # add the plots
        scat = ax.scatter(self.pos.cpu()[:,0].numpy(),self.pos.cpu()[:,1].numpy(),self.pos.cpu()[:,2].numpy(),c='k',alpha=.1,marker='o',s=3)

        # and keypoints
        if self.keyp is not None:
            body_colors = ['dodgerblue','red','lime','orange']
            for i,body in enumerate(self.ikeyp.cpu().numpy()):
                ax.scatter(self.keyp.cpu()[i,0], self.keyp.cpu()[i,1], self.keyp.cpu()[i,2], zdir='z', s=100, c=body_colors[int(body)],rasterized=True)

        if True:
            # plot the body supports
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = self.body_support_0
            if reduce_mean:
                c_hip = torch.mean(c_hip,dim=0).unsqueeze(0)
                c_ass = torch.mean(c_ass,dim=0).unsqueeze(0)
                c_mid = torch.mean(c_mid,dim=0).unsqueeze(0)
                c_nose = torch.mean(c_nose,dim=0).unsqueeze(0)
                c_tip = torch.mean(c_tip,dim=0).unsqueeze(0)
                c_impl = torch.mean(c_impl,dim=0).unsqueeze(0)
                    
            for p in [c_hip.cpu(),c_mid.cpu(),c_nose.cpu(),c_ass.cpu(),c_tip.cpu(),c_impl.cpu()]:
                ax.scatter(p[:,0,0],p[:,1,0],p[:,2,0],zdir='z', s=100, alpha = 0.1 , c='k',rasterized=True)
            for p,q in zip([c_nose.cpu(),c_nose.cpu(),c_mid.cpu(),c_impl.cpu(),c_impl.cpu()],[c_mid.cpu(),c_tip.cpu(),c_ass.cpu(),c_nose.cpu(),c_tip.cpu()]):
                p = p.numpy()
                q = q.numpy()
                for ii in range(p.shape[0]):
                    if reduce_mean:
                        ax.plot([p[ii,0,0],q[ii,0,0]],[p[ii,1,0],q[ii,1,0]],[p[ii,2,0],q[ii,2,0]],c='k',lw = 4)
                    else:
                        ax.plot([p[ii,0,0],q[ii,0,0]],[p[ii,1,0],q[ii,1,0]],[p[ii,2,0],q[ii,2,0]],c='k',alpha = 0.4)

            if plot_ellipsoids:
                # plot the ellipsoids as well!
                # we need a_hip and b_hip
                s = self.position[:,2]
                a_hip = a_hip_0 + a_hip_delta * s
                b_hip = b_hip_0 + b_hip_delta * (1.-s)
                d_hip = .75 * a_hip
                if reduce_mean:
                    a_hip = torch.mean(a_hip,dim=0).unsqueeze(0).cpu().numpy()
                    b_hip = torch.mean(b_hip,dim=0).unsqueeze(0).cpu().numpy()

                    # not really a proper way to average rotation
                    R_body = torch.mean(R_body,dim=0).unsqueeze(0).cpu().numpy()
                    R_head = torch.mean(R_head,dim=0).unsqueeze(0).cpu().numpy()
                    R_nose = torch.mean(R_nose,dim=0).unsqueeze(0).cpu().numpy()
                    
                    def add_wireframe_to_plot(ax,R_body,c_hip,style='hip',this_color='k',this_alpha=.4):
                        # FIRST PLOT THE ELLIPSE, which is the hip
                        # generate points on a sphere
                        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

                        # get the mesh, by using the equation of an ellipsoid
                        if style == 'hip':
                            x=np.cos(u)*a_hip
                            y=np.sin(u)*np.sin(v)*b_hip
                            z=np.sin(u)*np.cos(v)*b_hip
                        if style == 'nose':
                            x=np.cos(u)*a_nose.cpu().numpy()
                            y=np.sin(u)*np.sin(v)*b_nose.cpu().numpy()
                            z=np.sin(u)*np.cos(v)*b_nose.cpu().numpy()
                        if style == 'impl':
                            x=np.cos(u)*r_impl.cpu().numpy()
                            y=np.sin(u)*np.sin(v)*r_impl.cpu().numpy()
                            z=np.sin(u)*np.cos(v)*r_impl.cpu().numpy()
                            
                                

                        # pack to matrix of positions
                        posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

                        # apply the rotatation and unpack
                        # posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
                        # REMEBRE BODY SUPPORTS ARE [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
                        posi_rotated = np.einsum('ij,ja->ia',R_body[0,...],posi) + c_hip[0,...].cpu().numpy()

                        x = posi_rotated[0,:]
                        y = posi_rotated[1,:]
                        z = posi_rotated[2,:]

                        # reshape for wireframe
                        x = np.reshape(x, (u.shape) )
                        y = np.reshape(y, (u.shape) )
                        z = np.reshape(z, (u.shape) )

                        h_hip = ax.plot_wireframe(x, y, z, color=this_color,alpha = this_alpha)
                        return h_hip
                    h_hip = add_wireframe_to_plot(ax,R_body,c_hip,style='hip')
                    h_hip = add_wireframe_to_plot(ax,R_nose,c_nose,style='nose')
                    h_hip = add_wireframe_to_plot(ax,R_nose,c_impl,style='impl')
                       
            # plot the body supports
            c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose = self.body_support_1
            if reduce_mean:
                c_hip = torch.mean(c_hip,dim=0).unsqueeze(0)
                c_ass = torch.mean(c_ass,dim=0).unsqueeze(0)
                c_mid = torch.mean(c_mid,dim=0).unsqueeze(0)
                c_nose = torch.mean(c_nose,dim=0).unsqueeze(0)
                c_tip = torch.mean(c_tip,dim=0).unsqueeze(0)

            
            for p in [c_hip.cpu(),c_mid.cpu(),c_nose.cpu(),c_ass.cpu(),c_tip.cpu()]:
                ax.scatter(p[:,0,0],p[:,1,0],p[:,2,0],zdir='z', s=100, alpha = 0.1 , c='peru',rasterized=True)
            for p,q in zip([c_nose.cpu(),c_nose.cpu(),c_mid.cpu()],[c_mid.cpu(),c_tip.cpu(),c_ass.cpu()]):
                p = p.numpy()
                q = q.numpy()
                for ii in range(p.shape[0]):
                    if reduce_mean: 
                        ax.plot([p[ii,0,0],q[ii,0,0]],[p[ii,1,0],q[ii,1,0]],[p[ii,2,0],q[ii,2,0]],c='peru',lw=4)
                    else: 
                        ax.plot([p[ii,0,0],q[ii,0,0]],[p[ii,1,0],q[ii,1,0]],[p[ii,2,0],q[ii,2,0]],c='peru',alpha = 0.4)

                        
            if plot_ellipsoids:
                # plot the ellipsoids as well!
                # we need a_hip and b_hip
                s = self.position[:,11]
                a_hip = a_hip_0 + a_hip_delta * s
                b_hip = b_hip_0 + b_hip_delta * (1.-s)
                d_hip = .75 * a_hip
                if reduce_mean:
                    a_hip = torch.mean(a_hip,dim=0).unsqueeze(0).cpu().numpy()
                    b_hip = torch.mean(b_hip,dim=0).unsqueeze(0).cpu().numpy()

                    # not really a proper way to average rotation
                    R_body = torch.mean(R_body,dim=0).unsqueeze(0).cpu().numpy()
                    R_head = torch.mean(R_head,dim=0).unsqueeze(0).cpu().numpy()
                    R_nose = torch.mean(R_nose,dim=0).unsqueeze(0).cpu().numpy()
                    
                    def add_wireframe_to_plot(ax,R_body,c_hip,style='hip',this_color='k',this_alpha=.4):
                        # FIRST PLOT THE ELLIPSE, which is the hip
                        # generate points on a sphere
                        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

                        # get the mesh, by using the equation of an ellipsoid
                        if style == 'hip':
                            x=np.cos(u)*a_hip
                            y=np.sin(u)*np.sin(v)*b_hip
                            z=np.sin(u)*np.cos(v)*b_hip
                        if style == 'nose':
                            x=np.cos(u)*a_nose.cpu().numpy()
                            y=np.sin(u)*np.sin(v)*b_nose.cpu().numpy()
                            z=np.sin(u)*np.cos(v)*b_nose.cpu().numpy()
                        if style == 'impl':
                            x=np.cos(u)*r_impl.cpu().numpy()
                            y=np.sin(u)*np.sin(v)*r_impl.cpu().numpy()
                            z=np.sin(u)*np.cos(v)*r_impl.cpu().numpy()
                            
                                

                        # pack to matrix of positions
                        posi = np.vstack((x.ravel(),y.ravel(),z.ravel()))

                        # apply the rotatation and unpack
                        # posi_rotated = ((R_body @ (posi.T + c_hip).T ).T + t_body).T
                        # REMEBRE BODY SUPPORTS ARE [c_hip,c_ass,c_mid,c_nose,c_tip,c_impl,R_body,R_head,R_nose]
                        posi_rotated = np.einsum('ij,ja->ia',R_body[0,...],posi) + c_hip[0,...].cpu().numpy()

                        x = posi_rotated[0,:]
                        y = posi_rotated[1,:]
                        z = posi_rotated[2,:]

                        # reshape for wireframe
                        x = np.reshape(x, (u.shape) )
                        y = np.reshape(y, (u.shape) )
                        z = np.reshape(z, (u.shape) )

                        h_hip = ax.plot_wireframe(x, y, z, color=this_color,alpha = this_alpha)
                        return h_hip
                    h_hip = add_wireframe_to_plot(ax,R_body,c_hip,style='hip',this_color='peru')
                    h_hip = add_wireframe_to_plot(ax,R_nose,c_nose,style='nose',this_color='peru')
#                     h_hip = add_wireframe_to_plot(ax,R_nose,c_impl,style='impl')
                                               
                        
                        
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        zmin,zmax = ax.get_zlim()

        max_range = np.array([xmax-xmin,ymax-ymin,zmax-zmin]).max() / 2.0

        mid_x = (xmax+xmin) * 0.5
        mid_y = (ymax+ymin) * 0.5
        mid_z = (zmax+zmin) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        #         ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(0, 2*max_range)

#         ax.set_xlabel('x (mm)',fontsize=16)
#         ax.set_ylabel('y (mm)',fontsize=16)
#         zlabel = ax.set_zlabel('z (mm)',fontsize=16)
    
#         ax.view_init(elev=11., azim=-130.)
        
        az = -57.
        el = 38.
        # top view
        view_style = 'top'
        for view_style in ['top','side','mix']:
            if view_style == 'top':
                az = -30
                el = 90
            if view_style == 'side':
                az = -15
                el = 9
            if view_style == 'mix':
                az = -34
                el = 28
                
            ax.view_init(elev=el, azim=az)
            if example_plot:
                print("WHAT")
                # hard coded for zoomed example figure
                mid_x = .12 
                mid_y = -.08 
                mid_z = 0.
                max_range = .075
                plt.xticks(np.array([-1,0,1])*.5*max_range + mid_x )
                plt.yticks(np.array([-1,0,1])*.5*max_range + mid_y )
                ax.set_zticks(np.array([0,1,2,3,4])*.5*max_range + mid_z )


                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                #         ax.set_zlim(mid_z - max_range, mid_z + max_range)
                ax.set_zlim(0, 2*max_range)

                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])

                #frame1.axes.yaxis.set_ticklabels([])

            if starting:

                # save the 
                plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/figure_number_starting_'+view_style+'.pdf',transparent=True)
                plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/figure_number_starting_'+view_style+'.png',transparent=True)

            if final:
                
                # save the 
                plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/figure_number_final_'+view_style+'.pdf',transparent=True)
                plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/figure_number_final_'+view_style+'.png',transparent=True)                
                
            else:       
                if self.figure_counter is not None:
                    # save the 
                    plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/figure_number_'+str(self.figure_counter).zfill(3)+'_'+view_style+'.pdf', transparent=True)
                    plt.savefig('figure_raw_pics/figure_5/fitting_cartoon/figure_number_'+str(self.figure_counter).zfill(3)+'_'+view_style+'.png', transparent=True)
                    # update hacky counter
            plt.show()
        if self.figure_counter is not None:
            # after doing both views, update counter
            self.figure_counter += 1
            plt.close('all')
    
    
    def enforce_bounds(self):
        upper_bound = self.upper_bound.view(1,self.dimensionality)
        lower_bound = self.lower_bound.view(1, self.dimensionality)
        self.position = torch.max(torch.min(self.position,upper_bound),lower_bound)
        self.velocity = torch.max(torch.min(self.velocity,self.velocity_limit),-self.velocity_limit)
    
    def update_global_best(self):
        if self.sorted_loss is None:
            self.sorted_loss, self.idx_sorted = torch.sort(self.loss_flat)
        
        if self.sorted_loss[0] < self.loss_winner:
            self.loss_winner = self.sorted_loss[0]
            self.winner = torch.zeros(1,self.dimensionality)
            self.winner[:,:9] = self.position[self.idx0_flat[self.idx_sorted[0]],:9]
            self.winner[:,9:] = self.position[self.idx1_flat[self.idx_sorted[0]],9:]

        
        # we have already reasampled, so they weighting is automatic: 
        self.meanwinner = torch.mean(self.position,dim = 0).unsqueeze(0)
        
        if self.save_history:
            self.histo_mu.append(torch.mean(self.position,dim = 0))
            self.histo_var.append(torch.std(self.position,dim = 0))
            self.histo_loss.append(self.sorted_loss[:self.swarm_size])

    def save_state_history_pre_resample(self):
        # save all the particles, the sorted loss and all other losses
        self.state_history_pre_resample.append( [self.position.clone(),                               
                                    self.loss_2d.clone(),self.r_impl_2d.clone(),
                                   self.r_nose_2d.clone(),self.r_ear_2d.clone(),
                                   self.r_ass_2d.clone()] )
    def save_state_history_post_resample(self):
        # save all the particles, the sorted loss and all other losses
        self.state_history_post_resample.append( [self.position.clone(),
                                                  self.sorted_loss.clone(), 
                                                  self.idx_sorted.clone()] )
        
    def run(self,verbose=True,cinema=False):
        self.populate(sobol = True)
        self.loss_winner = 10000.
        
        if cinema:
            self.plot_status()
        # # do first rough resample
        self.update_loss_flat()
        self.resample_max()
        if cinema:
            self.plot_status()
        
        # do the steps
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
               
            # explode the current state of the particles
            if (iteration == 0) or (iteration == 2):
                self.blow_up(style='big')
            else:
                self.blow_up(style='small')
            
            # clip the particles to be inside the global bounds
            self.enforce_bounds()
            
            # update the loss
            self.update_loss_flat()
            
            if cinema:
                self.plot_status()
              
            # resample the particles based on the loss
            self.resample_max()
            # update the global best!
            self.update_global_best()
            
            if cinema:
                self.plot_status()

            toc = time.monotonic()
        
            if verbose:
                print("it {} of {}, best loss is {}, time {}".format( iteration, self.max_iterations, self.loss_winner,toc-tic ))

                
    def run2(self,verbose=True,cinema=False,save_history=False):
        # kinda hacky for making figures:
        self.figure_counter = 0
        
        # for plotting examples 
        self.populate(sobol = True)
        self.loss_winner = 10000.
        
        if cinema:
            self.plot_status()
            plt.title("starting")
        # # do first rough resample
        self.update_loss_flat()
        if save_history:
            self.save_state_history_pre_resample()

        self.resample_max()
        if save_history:
            self.save_state_history_post_resample()

        self.multiplier_schedule = [1,1,.5,.25,.1,.05,.025,.01]
        if cinema:
            self.plot_status()
            # plot the status version
            self.plot_status(reduce_mean=True, plot_ellipsoids=True)
      
        # do the steps
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
               
            # explode the current state of the particles
            if (iteration == 0):
                self.blow_up(style='big')
            else:
                self.blow_up(style='small', multiplier=self.multiplier_schedule[iteration])
            
            # clip the particles to be inside the global bounds
            self.enforce_bounds()
            
            # update the loss
            self.update_loss_flat()
            if save_history:
                self.save_state_history_pre_resample()

            if cinema:
                self.plot_status()
              
            # resample the particles based on the loss
            self.resample_max()
            if save_history:
                self.save_state_history_post_resample()
            # update the global best!
            self.update_global_best()
            
            if cinema:
                self.plot_status()
                # save after resampling as well
                self.plot_status(reduce_mean=True, plot_ellipsoids=True)

            toc = time.monotonic()
        
            if verbose:
                print("it {} of {}, best loss is {}, time {}".format( iteration, self.max_iterations, self.loss_winner,toc-tic ))                
                
                
    def run_separately(self,verbose=True,cinema=False):
        self.populate(sobol = True)
        self.loss_winner = 10000.
        
        if cinema:
            self.plot_status()
        # # do first rough resample
        self.update_loss_flat()
        self.resample_max()
        if cinema:
            self.plot_status()
        
        # do the steps
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
               
            # explode the current state of the particles
            if (iteration == 0) or (iteration == 2):
                self.blow_up(style='big')
            else:
                self.blow_up(style='small')
            
            # clip the particles to be inside the global bounds
            self.enforce_bounds()
            
            # update the loss
            self.update_loss_flat()
            
            if cinema:
                self.plot_status()
              
            # resample the particles based on the loss
            self.resample_max()
            # update the global best!
            self.update_global_best()
            
            if cinema:
                self.plot_status()

            toc = time.monotonic()
        
            if verbose:
                print("it {} of {}, best loss is {}, time {}".format( iteration, self.max_iterations, self.loss_winner,toc-tic ))                
                
                
                
#     def plot_winner(self):
#         dist0,_,body_support_0 = particles_to_distance(part[:,:9],pos,implant = True)
#         dist1,_,body_support_1 = particles_to_distance(part[:,9:],pos,implant = False)
#         body_supports = [body_support_0,body_support_1]

#         positions = self.pos.numpy()
#         best_mouse = self.winner.numpy()[0]
#         # best_mouse = pzo.global_best.detach().cpu().numpy()[0]
#         # best_mouse = torch.mean(pzo.particle_best,dim=0).numpy()

#         plot_fitted_mouse_new_nose(positions,x0_start,best_mouse,keyp = keyp,ikeyp = ikeyp,body_supports=body_supports)
#         # geolm_convergence_single(history)
        








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





