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
