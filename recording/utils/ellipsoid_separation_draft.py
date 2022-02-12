#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:26:28 2018

@author: chrelli
"""
import numpy as np

# re-do the matlab function

from fitting_utils import mouse_body_geometry

def ellipsoid_output(x0):
    # this just runs on one mouse
    # x0 has the parameters of the function, need to unpack the angles, the translation and the angles
    beta = x0[0]
    gamma = x0[1]
    s = x0[2]
    theta = x0[3]
    phi = x0[4]
    t_body = x0[5:8]

    # get the coordinates c of the mouse body in it's own reference frame
    R_body,R_nose,c_mid,c_hip,c_nose,a_hip,b_hip,a_nose,b_nose,Q_hip,Q_nose = mouse_body_geometry(beta,gamma,s,theta,phi)

    # and convert them to real space!
    # Now, calculate the distance vectors from the origin of the hip, mid and head
    c_hip = ( c_hip + t_body ).T
    c_mid = ( R_body @ c_mid + t_body ).T
    c_nose = ( R_body @ c_nose + t_body ).T

    # for the algebraic function:
    #    if string == 'hip':
    # the coefficients:
    coeff_canon_hip = np.array([a_hip,b_hip,b_hip])
    # center coordinate in 'real space!'
    r_hip = c_hip
    # the rotation matrix in 'real space!'
    A_hip = R_body

    #    if string == 'head':
    # the coefficients:
    coeff_canon_nose = np.array([a_nose,b_nose,b_nose])
    # center coordinate in 'real space!'
    r_nose = c_nose
    # the rotation matrix in 'real space!'
    A_nose = R_nose

    return [coeff_canon_hip,r_hip,A_hip],[coeff_canon_nose,r_nose,A_nose]



tracking_holder = np.load('/home/chrelli/git/3d_sandbox/mycetrack0p4/green_pink_042.npy')

i_frame = 84750
x0 = tracking_holder[:,i_frame]

print(x0)


hip_pack_0,nose_pack_0 = ellipsoid_output(x0[0:8])         
hip_pack_1,nose_pack_1 = ellipsoid_output(x0[8:])

coeff_i,coeff_j,r_i,r_j,A_i,A_j = hip_pack_0[0],hip_pack_1[0],hip_pack_0[1],hip_pack_1[1],hip_pack_0[2],hip_pack_1[2]

coeff_i,coeff_j,r_i,r_j,A_i,A_j = hip_pack_0[0],nose_pack_0[0],hip_pack_0[1],nose_pack_0[1],hip_pack_0[2],nose_pack_0[2]


#%% try to plot the ellipsoids!

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_fit = x0
ax,h_hip_0,h_nose_0 = add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],color = 'r')
if len(x_fit) > 8:
    x_fit = x0[8:]
    ax,h_hip_1,h_nose_1 = add_mouse_for_video(ax,x_fit[0],x_fit[1],x_fit[2],x_fit[3],x_fit[4],x_fit[5:8],color = 'orange')

#%%
Algebraic_Separation_Condition(coeff_canon_i,coeff_canon_j,r_i,r_j,A_i,A_j)

#%%
#%%
def Algebraic_Separation_Condition(coeff_i,coeff_j,r_i,r_j,A_i,A_j):
    #%ALGEBRAIC_SEPARATION_CONDITION Algebraic condition stating the contact
    #%                               status between two ellpsoids.
    #%   STATUS = ALGEBRAIC_SEPARATION_CONDITION(COEFF_I,COEFF_J,R_I,R_J,A_I,A_J)
    #%   outputs the contact status (e.g., separated, single point contact, and
    #%   overlapped) between 2 rigid ellipsoids by solving a 4th order
    #%   polynomial.
    #%
    #%       ARGUMENT DESCRIPTION:
    #%               COEFF_I  -  ellipsoid radii (x,y,z) of surface i.
    #%               COEFF_J  -  ellipsoid radii (x,y,z) of surface j.
    #%                   R_I  -  position vector of surface i's centroid.
    #%                   R_J  -  position vector of surface j's centroid.
    #%                   A_I  -  rotation matrix of surface i.
    #%                   A_J  -  rotation matrix of surface j.
    #%
    #%       OUTPUT DESCRIPTION:
    #%                 STATUS - 'yes' if ellipsoids are apart and 'no' if overlapped.
    #%
    #%   Example
    #%   -------------
    #%     coeff_i = [1.0, 2.340, 1.0];
    #%     coeff_j = [1.0, 1.0, 1.5];
    #%     r_i = [rand(1)*5, rand(1)*5, rand(1)*5]';
    #%     r_j = [rand(1)*5, rand(1)*5, rand(1)*5]';
    #%     u_i = [rand(1) rand(1) rand(1)]'; u_i = u_i./norm(u_i);
    #%     u_j = [rand(1) rand(1) rand(1)]'; u_j = u_j./norm(u_j);
    #%     A_i = rotation_angle_axis(deg2rad(rand(1)*90),u_i);
    #%     A_j = rotation_angle_axis(deg2rad(rand(1)*90),u_j);
    #%     status = Algebraic_Separation_Condition(coeff_i,coeff_j,r_i,r_j,A_i,A_j);
    #%
    #% See also ellipsoid.
    #
    #% References:
    #%   Wang, W., Wang, J., Kim, M.-S.
    #%   An algebraic condition for the separation of two ellipsoids.
    #%   Computer Aided Geometric Design,
    #%   18(6):531�539, 2001.
    #%
    #%   Jia, X., Choi, Y.-K., Mourrain, B., Wang, W.
    #%   An algebraic approach to continuous collision detection for ellipsoids.
    #%   Computer Aided Geometric Design,
    #%   28:164�176, 2011.
    #%
    #% Credits:
    #% Daniel Simoes Lopes
    #% IDMEC
    #% Instituto Superior Tecnico - Universidade Tecnica de Lisboa
    #% danlopes (at) dem ist utl pt
    #% http://web.ist.utl.pt/daniel.s.lopes/
    #%
    #% July 2011 original version.
    # Converted for numpy by C. Ebbesen, christian.ebbesen@nyumc.org, June 2018




    #%__________________________________________________________________________
    #%  Characteristic polynomial:
    #%  f(lambda) = det(lambda*A - Ma'*(Mb^-1)'*B*(Mb^-1)*Ma)
    #
    #% Ellipsoid matrices in the canonical form.
    #% Surface i.
    #A = [1/coeff_canon_i(1)^2 0 0 0;
    #     0 1/coeff_canon_i(2)^2 0 0;
    #     0 0 1/coeff_canon_i(3)^2 0;
    #     0 0 0 -1];
    #
    ## Surface j.
    #B = [1/coeff_canon_j(1)^2 0 0 0;
    #     0 1/coeff_canon_j(2)^2 0 0;
    #     0 0 1/coeff_canon_j(3)^2 0;
    #     0 0 0 -1];

    # make the matrices in numpy:
    A = np.diag(np.hstack((coeff_i**-2,-1)))
    B = np.diag(np.hstack((coeff_j**-2,-1)))

    #% Rigid body transformations.
    #T_i = [A_i    r_i;
    #       0 0 0   1];
    #T_j = [A_j    r_j;
    #       0 0 0  1];
    #Ma = T_i;
    #Mb = T_j;

    T_i = np.vstack((np.column_stack((A_i,r_i)),[0,0,0,1]))
    T_j = np.vstack((np.column_stack((A_j,r_j)),[0,0,0,1]))
    Ma = T_i
    Mb = T_j



    #% aij belongs to A in det(lambda*A - Ma'*(Mb^-1)'*B*(Mb^-1)*Ma).
    # Python index nightmare, to matrix notation
    a11 = A[0,0];
    a12 = A[0,1];
    a13 = A[0,2];
    a14 = A[0,3];
    a21 = A[1,0];
    a22 = A[1,1];
    a23 = A[1,2];
    a24 = A[1,3];
    a31 = A[2,0];
    a32 = A[2,1];
    a33 = A[2,2];
    a34 = A[2,3];
    a41 = A[3,0];
    a42 = A[3,1];
    a43 = A[3,2];
    a44 = A[3,3];

    #% bij belongs to b = Ma'*(Mb^-1)'*B*(Mb^-1)*Ma .
    #aux = inv(Mb)*Ma;
    #b = aux'*B*aux;
    aux = np.linalg.inv(Mb)@Ma
    b = aux.T @ B @ aux

    b11 = b[0,0];
    b12 = b[0,1];
    b13 = b[0,2];
    b14 = b[0,3];
    b21 = b[1,0];
    b22 = b[1,1];
    b23 = b[1,2];
    b24 = b[1,3];
    b31 = b[2,0];
    b32 = b[2,1];
    b33 = b[2,2];
    b34 = b[2,3];
    b41 = b[3,0];
    b42 = b[3,1];
    b43 = b[3,2];
    b44 = b[3,3];

    #% Coefficients of the Characteristic Polynomial.
    T4 = (-a11*a22*a33);
    T3 = (a11*a22*b33 + a11*a33*b22 + a22*a33*b11 - a11*a22*a33*b44);
    T2 = (a11*b23*b32 - a11*b22*b33 - a22*b11*b33 + a22*b13*b31 -
          a33*b11*b22 + a33*b12*b21 + a11*a22*b33*b44 - a11*a22*b34*b43 +
          a11*a33*b22*b44 - a11*a33*b24*b42 + a22*a33*b11*b44 -
          a22*a33*b14*b41);
    T1 = (b11*b22*b33 - b11*b23*b32 - b12*b21*b33 + b12*b23*b31 +
          b13*b21*b32 - b13*b22*b31 - a11*b22*b33*b44 + a11*b22*b34*b43 +
          a11*b23*b32*b44 - a11*b23*b34*b42 - a11*b24*b32*b43 +
          a11*b24*b33*b42 - a22*b11*b33*b44 + a22*b11*b34*b43 +
          a22*b13*b31*b44 - a22*b13*b34*b41 - a22*b14*b31*b43 +
          a22*b14*b33*b41 - a33*b11*b22*b44 + a33*b11*b24*b42 +
          a33*b12*b21*b44 - a33*b12*b24*b41 - a33*b14*b21*b42 +
          a33*b14*b22*b41);
    T0 = (b11*b22*b33*b44 - b11*b22*b34*b43 - b11*b23*b32*b44 +
          b11*b23*b34*b42 + b11*b24*b32*b43 - b11*b24*b33*b42 -
          b12*b21*b33*b44 + b12*b21*b34*b43 + b12*b23*b31*b44 -
          b12*b23*b34*b41 - b12*b24*b31*b43 + b12*b24*b33*b41 +
          b13*b21*b32*b44 - b13*b21*b34*b42 - b13*b22*b31*b44 +
          b13*b22*b34*b41 + b13*b24*b31*b42 - b13*b24*b32*b41 -
          b14*b21*b32*b43 + b14*b21*b33*b42 + b14*b22*b31*b43 -
          b14*b22*b33*b41 - b14*b23*b31*b42 + b14*b23*b32*b41);

    #%__________________________________________________________________________
    #%  Roots of the characteristic_polynomial (lambda0, ... , lambda4).
    #characteristic_polynomial = [T4 T3 T2 T1 T0]';
    #r = roots(characteristic_polynomial);
    print("TT")
    print([T4,T3,T2,T1,T0])
    r = np.roots(np.array([T4,T3,T2,T1,T0]))

    print("r")
    print(r)
    #% Correct numerical error of real valued polynomial zeros that are
    #% accompanied by complex numbers.
    #% for k = 1:4
    #% 	if (imag(r(k)) <= 10^-3)
    #% 		r(k) = real(r(k));
    #% 	end
    #% end

    #%__________________________________________________________________________
    #%  Algebraic condition for contact detection between ellipsoids.
    #% Find complex roots.
    #for k = 1:4
    #    complex_roots(k) = ~isreal(r(k));
    #end
    # complex_roots = ~np.isreal(r) # don't need this

    #% Find the (real) negative roots.
    #negative_roots_ids = find( (~complex_roots').*r < 0);
    negative_roots_ids = np.where(np.isreal(r) * r<0 )[0]
    print("negative_roots_ids")
    print(negative_roots_ids)
    #% Contact detection status.
    #if length(negative_roots_ids) == 2
    #    if  r(negative_roots_ids(1)) ~= r(negative_roots_ids(2))
    #        disp('Separation Condition: quadric surfaces are separated.')
    #        status = 'y';
    #        return
    #    elseif abs(r(negative_roots_ids(1)) - r(negative_roots_ids(2))) <= 10^-3
    #            % r(negative_roots_ids(1)) == r(negative_roots_ids(2))
    #        disp(['Separation Condition: quadric surfaces share a single',...
    #              'contact point.'])
    #        status = 'n';
    #        return
    #    end
    #else
    #    disp('Separation Condition: quadric surfaces are not separated (overlapping).')
    #    status = 'n';
    #    return
    #end

    if len(negative_roots_ids) == 2:
        if r[negative_roots_ids[0]] != r[negative_roots_ids[1]]:
            print('Separation Condition: quadric surfaces are separated.')
            return True
        elif np.abs(r[negative_roots_ids[0]] - r[negative_roots_ids[1]]) <= 1e-3:
            print(['Separation Condition: quadric surfaces share a single contact point.'])
            return False
    else:
        print('Separation Condition: quadric surfaces are not separated (overlapping).')
        return False

#%%

def rotation_angle_axis(theta,u):
#    %ROTATION_ANGLE_AXIS The Rodrigues' formula for rotation matrices.
#    %   R = ROTATION_ANGLE_AXIS(THETA,U) The formula recieves an angle of rotation given by theta and a unit vector, u, that
#    %   defines the axis of rotation.
#    %
#    %       ARGUMENT DESCRIPTION:
#    %           THETA - angle of rotation (radians).
#    %               U - unit vector
#    %
#    %       OUTPUT DESCRIPTION:
#    %               R - rotation matrix.
#    %
#    %   Example
#    %   -------------
#    %   R = rotation_angle_axis(deg2rad(pi/6),[sqrt(2)/2, 0.0, sqrt(2)/2])
#    %
#
#    % Credits:
#    % Daniel Simoes Lopes
#    % IDMEC
#    % Instituto Superior Tecnico - Universidade Tecnica de Lisboa
#    % danlopes (at) dem ist utl pt
#    % http://web.ist.utl.pt/daniel.s.lopes/
#    %
#    % July 2011 original version.
#
#
#    %__________________________________________________________________________
#    %  Rodrigues' rotation formula.
    u = u/np.linalg.norm(u);
    S = np.array([[    0,  u[2], -u[1]],[-u[2],   0,   u[0]],[u[1], - u[0], 0]])
    R = np.eye(3) + np.sin(theta)*S + (1-np.cos(theta))*S**2;

    return R

def asc_demo():

    coeff_i = np.array([1.0, 2.340, 2.865]);
    coeff_j = np.array([1.0, 1.0, 1.5]);

    r_i = np.array([np.random.uniform(0,1)*5, np.random.uniform(0,1)*5, np.random.uniform(0,1)*5]);
    r_j = np.array([np.random.uniform(0,1)*5, np.random.uniform(0,1)*5, np.random.uniform(0,1)*5]);
    u_i = np.random.uniform(0,1,3); u_i = u_i/np.linalg.norm(u_i);
    A_i = rotation_angle_axis(np.random.uniform(0,np.pi/2),u_i);
    u_j = np.random.uniform(0,1,3); u_j = u_j/np.linalg.norm(u_j);
    A_j = rotation_angle_axis(np.random.uniform(0,np.pi/2),u_j);

    A_j = A_i
    u_j = u_i
    r_j = r_i +2

#    r_i = [    3.6122,    0.7493,    3.2980]
#    r_j = [    2.5930,    4.8649,    3.2450]
#    A_i = [[    0.7228  ,  0.6656,   -0.1856],
#   [-0.1532   , 0.4162  ,  0.8963],
#   [ 0.6738  , -0.6194  ,  0.4029]]
#
#    A_j = [[    0.8405  ,  0.4640  , -0.2796],
#   [-0.3898 ,   0.8765  ,  0.2826],
#    [0.3762  , -0.1285  ,  0.9176]]

#    graph_ellipsoid(coeff_i,r_i,A_i)
#    hold on
#    graph_ellipsoid(coeff_j,r_j,A_j)

#    alpha(0.5)
#    view(45,45)
#    axis equal
#    rotate3d
    print(r_i)
    print('')
    status = Algebraic_Separation_Condition(coeff_i,coeff_j,r_i,r_j,A_i,A_j);
    print(status)


asc_demo()
