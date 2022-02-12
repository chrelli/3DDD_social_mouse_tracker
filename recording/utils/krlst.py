#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:05:14 2018

@author: chrelli
"""


class Gaussian:
    """
    A Gaussian kernel.
    """

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, X, Z):
        return Gaussian.kernel(X, Z, self.sigma)

    @classmethod
    def kernel(cls, X, Z, sigma):
        """
        Computes the Gaussian kernel for the matrices X and Z.
        """
        # make sure X and Z are Numpy matrices, and float values
        X = np.matrix(X,)
        Z = np.matrix(Z)
#        # Normalise between different bandwidths
#        if hasattr(sigma, '__iter__'):
#           X /= 1.4142 * (np.array(sigma))
#           Z /= 1.4142 * (np.array(sigma))
#           sigma = 1.0
#        else:
#           sigma = float(sigma)
        n, m = X.shape[0], Z.shape[0]
        XX = np.multiply(X, X)
        XX = XX.sum(axis = 1)
        ZZ = np.multiply(Z, Z)
        ZZ = ZZ.sum(axis = 1)
        d = np.tile(XX, (1, m)) + np.tile(ZZ.T, (n, 1)) - 2 * X * Z.T
        Kexpd = np.exp(-d.T / (2 * sigma * sigma))
        return np.matrix(Kexpd)


"""
% Kernel Recursive Least-Squares Tracker algorithm
%
% S. Van Vaerenbergh, M. Lazaro-Gredilla, and I. Santamaria, "Kernel
% Recursive Least-Squares Tracker for Time-Varying Regression," IEEE
% Transactions on Neural Networks and Learning Systems, vol. 23, no. 8, pp.
% 1313-1326, Aug. 2012, http://dx.doi.org/10.1109/TNNLS.2012.2200500
%
% Remark: using back-to-the-prior forgetting
%
% This file is part of the Kernel Adaptive Filtering Toolbox for Matlab.
% https://github.com/steven2358/kafbox/

Go over everythng and check which are col vectors and which are not!!
"""

import numpy as np
# initialize the class
class krlst:

    def __init__(self, kernel, params={}):
        self.kernel = kernel        # Instance of a kernel to use, should return K of (X,Y), so will have sigma in itself
        # initialize with some reasonable values
        # inital value guesses, can be set from outside
        self.Lambda = .999; # forgetting factor, capitalized not to conflict w python lambda
        self.sn2 = 1E-2; # noise to signal ratio (regularization parameter)
        self.M = 50; # dictionary size
        self.jitter = 1E-6; # jitter noise to avoid roundoff error

#        self.kerneltype = 'gauss'; # kernel type
#        self.kernelpar = 1; % kernel parameter
#
        # these are the trainables, set for start!
        # these will start as empty lists, maybe make None instead?
        self.dico = []; # dictionary, renamed from dict to dico
        self.Q = []; # inverse kernel matrix
        self.mu = []; # posterior mean
        self.Sigma = []; # posterior covariance
        self.nums02ML = 0;
        self.dens02ML = 0;
        self.s02 = 0; # signal power, adaptively estimated
        self.prune = False; # flag
        self.reduced = False; # flag
        self.calc_variance = False; # Flag, can be turned off to save a bit of computations later

        # Allow overriding the attributes through the params dictionary
        for k in params:
            if hasattr(self, k):
                setattr(self, k, params[k])

#    def make_column(self,x):
#        # make into a column vector (matrix)
#        if len(x.shape) == 1:
#            x = np.matrix(x).T
#        return x

    def set_defaults(self,string):
        # these are all for 5-embedding
        if string == 'def':
            sigma_est =  0.01
            self.sn2 =  1e-2
            self.Lambda = 0.999
        elif string == 'x':
            sigma_est =  0.1949
            self.sn2 =  7.345067e-07
            self.Lambda = 0.9999
        elif string == 'y':
            sigma_est =  0.1813
            self.sn2 =  7.696719e-06
            self.Lambda = 0.9998
        elif string == 'z':
            sigma_est =  0.0661
            self.sn2 =  4.833406e-07
            self.Lambda = 0.9994

        elif string == 'beta':
            # try both?
#            sigma_est = 8.9936
#            self.sn2 =  4.339712e-05
#            self.Lambda = 0.9943

            sigma_est =   46.2523
            self.sn2 = 1.352459e-06
            self.Lambda = 0.9978

        elif string == 'gamma':
            sigma_est =  5.5514
            self.sn2 = 1.775386e-04
            self.Lambda: 0.9951

        elif string == 's':
            sigma_est =   8.5777
            self.sn2 =       7.723837e-06
            self.Lambda =    0.9943

        elif string == 'theta':
            sigma_est = 21.9346
            self.sn2 = 1.165146e-04
            self.Lambda = 0.9927

        elif string == 'phi':
            sigma_est =  2.0668
            self.sn2 = 2.451947e-04
            self.Lambda = 0.9899

        # and set the kernel:
        self.kernel = Gaussian(sigma = sigma_est)

    def evaluate(self,x): # predicts y given x
        # turn x into a column vector for all the matlab style math to work?
#        x = self.make_column(x)
        # check if there is a dictionary
        if len(self.dico) > 0:
            # if there is, do the prediction step!
            k = self.kernel(self.dico,x)
            q = self.Q * k.T
            mean_test = q.T * self.mu # predictive mean
            if self.calc_variance:
                # not converted yet!
                # this gets only the diagonal, or what?
                #ktt = kernel(x,x,[kaf.kerneltype '-diag'],kaf.kernelpar);
                #sf2 = ktt + kaf.jitter + sum(k.*((kaf.Q*kaf.Sigma*kaf.Q-kaf.Q)*k),1)';
                #sf2(sf2<0) = 0;
                #var_test = kaf.s02*(kaf.sn2 + sf2); % predictive variance
                print('not implemented')
            else:
                var_test = np.zeros((x.shape[0],1))

        else:
            # if there is no dictionary, return zeros and nans
            mean_test = np.zeros((1,1))
            var_test = np.nan * np.zeros((x.shape[0],1))

        return mean_test,var_test

    def train(self,x,y): # trains the algorithm!
        # turn x into a column vector for all the matlab style math to work?
#        x = self.make_column(x)
        # y is a scalar for right now!

        m = int(len(self.Sigma))
        if m < 1: # initialize!
            k = self.kernel(x,x);
            k = k + self.jitter;
            self.Q = 1/k;
            self.mu = y*k/(k+self.sn2);
            self.Sigma = k - k**2/(k+self.sn2);
            self.dico = np.mat(x); # dictionary bases are rows (for some reason)!!!
            self.nums02ML = float(y**2/(k+self.sn2) );
            self.dens02ML = 1;
            self.s02 = self.nums02ML / self.dens02ML;

        else:
            # forget using back-to-the-prior forgetting rule
            K = self.kernel(self.dico,self.dico) + self.jitter*np.eye(m);
            self.Sigma = self.Lambda*self.Sigma + (1-self.Lambda)*K; # forget Sigma
            self.mu = np.sqrt(self.Lambda)*self.mu; # forget mu

            # predict
            k = self.kernel(np.vstack((self.dico,x)),x);
            # all but the last one, as a column!
            kt = k[0,0:-1].T;
            # the last one:
            ktt = k[0,-1] + self.jitter;
            # actually, since there is apossibility of some numnerical error, let's just set ktt 1+jit by hand
            ktt = 1 + self.jitter;
            # q is a column
            q = self.Q*kt;
            y_mean = q.T*self.mu; # predictive mean
            y_mean = y_mean[0,0] # convert to float from 1x1 matrix
            gamma2 = ktt - kt.T*q;
            gamma2[gamma2<0]=0+self.jitter/10; # projection uncertainty
            # make gamma2 a number, not a 1x1 matrix!
            gamma2 = float(gamma2)
            # calc h
            h = self.Sigma*q;

            sf2 = gamma2 + q.T*h; sf2[sf2<0]=0; # noiseless prediction variance
            sy2 = self.sn2 + float(sf2);
            # y_var = s02*sy2; % predictive variance
            # include a new sample and add a basis
            Qold = self.Q; # old inverse kernel matrix

            p = np.vstack((q,-1));
            # pad Q with zeros on the right and bottom edge around it!
            self.Q = np.pad(self.Q,(0,1),mode = 'constant', constant_values=(0)) + 1/gamma2*(p*p.T);

            p = np.vstack((h,sf2));
            self.mu = np.vstack((self.mu,y_mean)) + (y - y_mean)/sy2*p; # posterior mean

            self.Sigma = np.vstack(( np.hstack((self.Sigma,h)),np.hstack((h.T,sf2)) )) - 1/sy2*(p*p.T); # posterior covariance
            m = m + 1;
            self.dico = np.vstack((self.dico,x.T));

            # estimate s02 via ML
            self.nums02ML = self.nums02ML + self.Lambda*(y - y_mean)**2/sy2;
            self.dens02ML = self.dens02ML + self.Lambda;
            self.s02 = self.nums02ML/self.dens02ML;

            self.prune = False;
            # delete a basis if necessary, if the basis size is above M
            if (m>self.M  or gamma2<self.jitter):
                if gamma2<self.jitter: # to avoid roundoff error
                    if gamma2<self.jitter/10:
                        print('Numerical roundoff error too high, you should increase jitter noise') #ok<WNTAG>
                    #set the criterion to one, except the newest one!
                    criterion = np.ones(m);
                    criterion[-1] = 0

                else: # MSE pruning criterion
                    # make the errors a column vector!
                    errors = (self.Q*self.mu).T/np.diag(self.Q);
                    # and make the criterion an an array!
                    criterion = np.asarray(abs(errors))[0];

                # remove element r, which incurs in the minimum error
                r = np.argmin(criterion)
                # make an index without r
                smaller = np.delete(np.arange(m),r)

                if r == m: # if we must remove the element we just added, perform reduced update instead
                    self.Q = Qold;
                    self.reduced = True;
                else:
                    Qs = self.Q[smaller, r];
                    # use np delet to drop the rows/cols which correspond to r!
                    qs = self.Q[r,r]; self.Q = np.delete(np.delete(self.Q,r,axis=0),r,axis=1);
                    self.Q = self.Q - (Qs*Qs.T)/qs;
                    self.reduced = False;

                # now set the updated values!
                self.mu = self.mu[smaller];
                # use the same numpy array as above to clean the rows and cols of sigma!
                self.Sigma = np.delete(np.delete(self.Sigma,r,axis=0),r,axis=1)
                # the dictionary has the basis as rows, so just delete that row!
                self.dico = np.delete(self.dico,r,axis = 0)
                self.prune = True;
