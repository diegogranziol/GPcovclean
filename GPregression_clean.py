#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:12:21 2018

@author: binxinru
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy import stats,spatial
from scipy.linalg import cholesky,cho_solve
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split 
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
#from random import Random
#% Function 3: Posteriori/Predictive distribution
def posterior(xtest,xob,theta):
    d = xob.shape[1]
    s=theta[0]
    l = theta[1:d+1]
    varn = theta[d+1]
#    n = xob.shape[0]
    '''return the mean and covariance matrix of the predictive distribution'''
    # computes the noisy kernel using RBF method
    kerneln = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn) 
    kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)

    Kn = kerneln(xob)
    # kernel between new and old data points
    K_newob=kernel(xtest,xob) 
    # kernel between new data points
    K_new=kernel(xtest,xtest)
    
    #% compute mean for posterior
    try:
        invK_f=np.linalg.solve(Kn,yob)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            
            invK_f=np.linalg.lstsq(Kn,yob)[0]       
        else:
            raise
    mean_po=K_newob.dot(invK_f)
    
    # compute covariance matrix/kernel for posterior
    try:
        invK_K=np.linalg.solve(Kn,K_newob.T)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            
            invK_K=np.linalg.lstsq(Kn,K_newob.T)[0]       
        else:
            raise
    cov_po=K_new-np.dot(K_newob,invK_K) 
    return [mean_po,cov_po]

## Function for computing the log marginal likelihood and its gradient 
def negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=False):
    """Returns - log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        Returns
        -------
        - log_likelihood : float
            Log-marginal likelihood of theta for training data.
        - log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
    d = xob.shape[1]
    theta=np.exp(lntheta)
    s=theta[0]
    l = theta[1:d+1]
    varn = theta[d+1]

    alpha = 1e-10     
    kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn)# +ConstantKernel() + Matern(length_scale=2, nu=3/2)
    
    if eval_gradient:
        K, K_gradient = kernel(xob, eval_gradient=True)
    else:
        K = kernel(xob)
    
    
    K[np.diag_indices_from(K)] += alpha 
#    Kernelob=[]
    try:
        L = cholesky(K, lower=True)  # Line 2
    except np.linalg.LinAlgError:
        return (-np.inf, np.zeros_like(theta)) \
            if eval_gradient else -np.inf
    
    # Support multi-dimensional output of self.y_train_
    y_train = yob
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
    
    alpha = cho_solve((L, True), y_train)  # Line 3
    
    # Compute log-likelihood (compare line 7)
    log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
    log_likelihood_dims -= np.log(np.diag(L)).sum()
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
    
    if eval_gradient:  # compare Equation 5.9 from GPML
        tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
        tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        log_likelihood_gradient_dims = \
            0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
        log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
    
    if eval_gradient:
        return -log_likelihood_gradient
    else:
        return -log_likelihood
    
def plot_eigspectrum (eigs,gridsize,sharpness):        
        neweig = eigs/max(eigs)
        gmax = 1
        gmin = min(neweig)
        lines = gridsize
        x1 = np.linspace(gmin,gmax,lines)
        sharpness = (1e-4)
        y1 = 0;
        for i in range(0, len(neweig)):
            y1 = y1+(1/float(len(neweig)))*mlab.normpdf(x1, neweig[i], sharpness)
        fig = plt.figure()

        line1, = plt.plot(x1,y1, label="Gaussians")
        fig = plt.gca()
        #set log or normal plot
        fig.set_xlim([gmin,1])
        fig.set_xscale('log')
        maxy = max([max(y1)])
        fig.set_ylim([0,maxy])
        plt.rcParams["figure.figsize"] = (10,10)
        
def print_coord(lntheta):
        global xob,yob,xtest
        d = xob.shape[1]
        theta=np.exp(lntheta)
        s=theta[0]
        l = theta[1:d+1]
        varn = theta[d+1]
        # compute the covariance matrix for observation data
        kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn)# +ConstantKernel() + Matern(length_scale=2, nu=3/2)
        Kn = kernel(xob)
        
        # compute eigenvalues of the covariance matrix
        eigs = np.linalg.eigvalsh(Kn)
#        neweig = eigs/max(eigs)
        m  = 100
        # log likelihood
        log_likelihood = negative_log_marginal_likelihood_clean (lntheta, xob,yob, m )

        # compute RMSE
        ypred,covpred = posterior_clean(xtest,xob,theta,m)
        #%  Compute the rms error
        RME=np.mean(np.sqrt((ypred-ftest)**2))

        # plot the eigen spectrum using Gaussian kernel smoothing
        plot_eigspectrum (eigs,1e4+1,1e-4)       
        titlespectra = ['Spectral Density: loglikelihood=' + str(log_likelihood) + \
                        ';  RMSE=' + str(RME)]
        
        plt.title(titlespectra)


# compute negative marginal likelihood based on clean covariance
def negative_log_marginal_likelihood_clean(lntheta, xob,yob,m):
    n,d = xob.shape
    theta=np.exp(lntheta)
    s=theta[0]
    l = theta[1:d+1]
    varn = theta[d+1]
    kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn)# +ConstantKernel() + Matern(length_scale=2, nu=3/2)
    K = kernel(xob)
    # specify the number of eigvalue outliers, m 
#    m = 100
    # generate the outlier eigenvalues and corresponding eigenvectors
    eigs_trucated, V_trucated = sp.sparse.linalg.eigsh(K, k = m)
#    eigs_trucated, V_trucated  = np.linalg.eigh(Kn)

#    eigs,V = np.linalg.eigh(K)   # v[:, i] is each normalised eigenvacetor 
#    eigs_trucated = eigs[-m:]
#    V_trucated = V[:,-m:]
    # construct the covariance matrix based on the outliers
    #C_mm = np.dot(V_trucated, np.diag(eigs_trucated)).dot(V_trucated.T)
    # compute the floor eigenvalue and normalised random vectors 
#    eig_0 = (np.trace(K) - sum(eigs_trucated))/(n-m)
#    np.random.seed(1)
#    Z = np.random.randn(n,n-m) 
#    Z = Z / sp.linalg.norm(Z, axis=0)
    
    # compute likelihood terms 
#    T1 = (n-m) * np.log(eig_0)
    T2 = np.sum( np.log(eigs_trucated) )
    T3 = np.sum( ( np.dot(yob.T, V_trucated)** 2 ) / eigs_trucated )
#    T4 = np.sum( ( np.dot(yob.T, Z) ** 2 ) / eig_0 )
    T5 = n * np.log(2* np.pi)
    
#    L = 0.5 * (T1 + T2 + T3 + T4 + T5)
    L = 0.5 * (T2 + T3 + T5)

    return L

# compute posterior mean and variance based on clean covariance
def posterior_clean(xtest,xob,theta,m):
    n,d = xob.shape
    s=theta[0]
    l = theta[1:d+1]
    varn = theta[d+1]
    #    n = xob.shape[0]
    '''return the mean and covariance matrix of the predictive distribution'''
    # computes the noisy kernel using RBF method
    kerneln = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn) 
    kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)
    Kn = kerneln(xob)
    
    # specify the number of eigvalue outliers, m 
    # generate the outlier eigenvalues and corresponding eigenvectors
    eigs_trucated, V_trucated = sp.sparse.linalg.eigsh(Kn, k = m)
#    eigs_trucated, V_trucated  = np.linalg.eigh(Kn)
    # construct the covariance matrix based on the outliers
    C_mm = np.dot(V_trucated, np.diag(eigs_trucated)).dot(V_trucated.T)
    # compute the floor eigenvalue and normalised random vectors 
    eig_0 = (np.trace(Kn) - sum(eigs_trucated))/(n-m)
#    eig_0 = 0
    np.random.seed(1)
    Z = np.random.randn(n,n-m) 
    Z = Z / sp.linalg.norm(Z, axis=0)
    # construct the covariance matrix based on the floor eigenvalues and random vector
    C_nm = np.dot(Z, eig_0 * np.eye(n-m)).dot(Z.T)
    # construct the clean covariance matrix
#    C_clean = C_mm + C_nm
    C_clean = C_mm
    # kernel between new and old data points
    K_newob=kernel(xtest,xob) 
    # kernel between new data points
    K_new=kernel(xtest)
    
    #% compute mean for posterior
    try:
        invK_f=np.linalg.solve(C_clean,yob)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            
            invK_f=np.linalg.lstsq(C_clean,yob)[0]       
        else:
            raise
    mean_po=K_newob.dot(invK_f)
    
    # compute covariance matrix/kernel for posterior
    try:
        invK_K=np.linalg.solve(C_clean,K_newob.T)
    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' in err.message:
            
            invK_K=np.linalg.lstsq(C_clean,K_newob.T)[0]       
        else:
            raise
    cov_po=K_new-np.dot(K_newob,invK_K) 
    return [mean_po,cov_po]
        
#%% Get Boston Housing Data
#Boston Housing Dataset
cd = 'train.csv'
train = pd.read_csv(cd)
##candidates = candidates.fillna(value='Null')
train.drop(['ID'],axis=1,inplace=True)
print(train.head())
tranx = train
trany = train
trany = (trany['medv'])
tranx.drop(['medv'], axis=1,inplace=True)

# TODO: Shuffle and split the data into training and testing subsets
xob, xtest, y_train, y_test = train_test_split(tranx, trany, test_size=0.1, random_state=1)
yob = y_train[:,None]
ftest = y_test[:,None]

#% set initial hyperparameter values
dim = xob.shape[1]
l = list(np.ones(dim))
theta0 = np.array([10]+l+[1e-6])
lntheta_01 = np.log(theta0)


#%% optimise hyperparameters using MLE for clean covariance
m = 298
NL_clean = lambda lntheta: negative_log_marginal_likelihood_clean(lntheta, xob,yob,m)
res = minimize(NL_clean, lntheta_01, method='L-BFGS-B', tol=1e-6)

#%% run GP regression using the optmal hyperparameter set
theta_opt = np.exp(res.x)
ypred,covpred = posterior_clean(xtest,xob,theta_opt,m)
#%  Compute the rms error
RMSE=np.mean( np.sqrt((ypred-ftest)**2) )
# print mean rms error
print(RMSE) 


#%% Kernel spectral analysis
theta_opt = np.exp(res.x)
n,d = xob.shape
#theta=np.exp(lntheta)
s=theta_opt[0]
l = theta_opt[1:d+1]
varn = theta_opt[d+1]
kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn)# +ConstantKernel() + Matern(length_scale=2, nu=3/2)
C = kernel(xob)
eigs,V = np.linalg.eigh(C)   # v[:, i] is each normalised eigenvacetor # C = V Eig V.T

# specify the number of eigvalue outliers, m 
m = 100
#neweig = eigs/max(eigs)
#diff_eigs = np.diff(neweig)
# generate the outlier eigenvalues and corresponding eigenvectors
eigs_trucated = eigs[-m:]
V_trucated = V[:,-m:]
# construct the covariance matrix based on the outliers
C_mm = np.dot(V_trucated, np.diag(eigs_trucated)).dot(V_trucated.T)
# compute the floor eigenvalue and normalised random vectors 
eig_0 = (np.trace(C) - sum(eigs_trucated))/(n-m)
Z = np.random.randn(n,n-m) 
Z = Z / sp.linalg.norm(Z, axis=0)
# construct the covariance matrix based on the floor eigenvalues and random vector
C_nm = np.dot(Z, eig_0 * np.eye(n-m)).dot(Z.T)
# construct the clean covariance matrix
C_clean = C_mm + C_nm

eigs_clean,V_clean = np.linalg.eigh(C_clean)   
# % visualise the eigen-spectrum 
plot_eigspectrum (eigs_clean,1e4+1,1e-4)
#plot_eigspectrum (eigs,1e4+1,1e-4)
#plot_eigspectrum (eigs_trucated,1e4+1,1e-2)
#%% check prediction performance
Kn = C_clean
# kernel between new and old data points
K_newob=kernel(xtest,xob) 
# kernel between new data points
K_new=kernel(xtest,xtest)
#% compute mean for posterior
invK_f=np.linalg.solve(Kn,yob)
mean_po=K_newob.dot(invK_f)
RME=np.mean(np.sqrt((mean_po-ftest)**2))