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
#% Function 2: Gaussian kernel 
def func2D(X):
    X = np.atleast_2d(X)
    x1=X[:,0]*15-5
    x2=X[:,1]*15
    y = (x2-5.1/(4*np.pi**2)*x1**2+5*x1/np.pi-6)**2+ 10*(1-1/(8*np.pi))*np.cos(x1)+10
    output = y/10 - 15
    return output[:,None]

def func1d(x):
    c = 0.3
    d = 0.3
    y = c*np.sin(x)**3# + d
    return y

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
## %%
#def obj_func(lntheta, xob,yob, eval_gradient=True):
#    if eval_gradient:
#        lml, grad = log_marginal_likelihood(lntheta, xob,yob, eval_gradient=True)
#    return -lml, -grad   
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
        eigs = np.linalg.eigvals(Kn)
        neweig = eigs/max(eigs)

        # log likelihood
        log_likelihood = - negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=False)

        # compute RMSE
        ypred,covpred = posterior(xtest,xob,theta)
        #%  Compute the rms error
        RME=np.mean(np.sqrt((ypred-ftest)**2))

        # plot the eigen spectrum using Gaussian kernel smoothing
        gmax = 1
        gmin = min(neweig)
        lines = 1e4+1
        x1 = np.linspace(gmin,gmax,lines)
        sharpness = (1e-4)
        y1 = 0;
        for i in range(0, len(neweig)):
            y1 = y1+(1/float(len(neweig)))*mlab.normpdf(x1, neweig[i], sharpness)
        fig = plt.figure()

        titlespectra = ['Spectral Density: \
                        loglikelihood=' + str(log_likelihood) + \
                        'RMSE=' + str(RME)]
        line1, = plt.plot(x1,y1, label="Gaussians")
        fig = plt.gca()
        #set log or normal plot
        fig.set_xlim([gmin,1])
        fig.set_xscale('log')
        maxy = max([max(y1)])
        fig.set_ylim([0,maxy])
        plt.rcParams["figure.figsize"] = (10,10)
        plt.title(titlespectra)

#        first_legend = plt.legend(handles=[line1], loc=1)
#
#        # Add the legend manually to the current Axes.
#        ax = plt.gca().add_artist(first_legend)
#
#        plt.title(titlespectra)
#        plt.show()
#        THETA = (xk[0],xk[1],xk[2],xk[3])
#        print('Log Marginal Likelihood = '+str(gp.log_marginal_likelihood(theta=THETA, eval_gradient=False)))
#        gp1 = gaussian_process.GaussianProcessRegressor(kernel=kernelfin)
#        print(gp1)
#        gp1.fit(X, y)
#        #x_pred = np.linspace(-6, 6).reshape(-1,1)
#        x_pred = x.reshape(-1,1)
#        y_pred, sigma = gp1.predict(x_pred, return_std=True)
#        print('RMSE = '+str(np.sqrt(sum((y-y_pred)**2/len(y)))))
#        #print(y_pred)
#        #print(gp1.predict(x_pred))
#        plt.figure(figsize=(10,8))
#        sns.regplot(x, y, fit_reg=False, label='Data')
#        plt.plot(x_pred, y_pred, color='blue', label='Prediction')
#        #c = 0.3
#        #d = 0.3
#        #xtruth = np.linspace(-6,6,101)
#        #ytruth = c*np.sin(xtruth)**3
#        #plt.plot(xtruth, ytruth, color='red', label='Truth')
#        plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
#                 np.concatenate([y_pred - 2*sigma,
#                                (y_pred + 2*sigma)[::-1]]),
#                 alpha=.5, fc='grey', ec='None', label='95% CI')
#        plt.xlabel('$x$')
#        plt.ylabel('$f(x)$')
#        plt.xlim(-6, 6)
#        plt.ylim(-3, 3)
#        plt.legend(loc='lower left');
#        plt.show()
        
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

#%% set initial hyperparameter values
dim = xob.shape[1]
l = list(np.ones(dim))
theta0 = np.array([10]+l+[1e-6])
lntheta_01 = np.log(theta0)
#ypred,ypred = posterior(xtest,xob,theta0)

#% optimise hyperparameters using MLE
NL = lambda lntheta: negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=False)
dNL = lambda lntheta: negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=True)
#bnds=((0,1e4),(0.0,1.0),(0.0,1.0),(0,1))
res = minimize(NL, lntheta_01, method='L-BFGS-B',jac=dNL, tol=1e-6,callback=print_coord)

#% run GP regression using the optmal hyperparameter set
theta_opt = np.exp(res.x)
ypred,covpred = posterior(xtest,xob,theta_opt)
#%  Compute the rms error
E=np.sqrt((ypred-ftest)**2)
# print mean rms error
print(np.mean(E))


#%%
##L = np.array([0.3,0.3])
#kern = RBF(length_scale=L, length_scale_bounds=(1e-05, 100.0)) # +ConstantKernel() + Matern(length_scale=2, nu=3/2)
#K,dK = kern(xob,eval_gradient=True)
#
#
#
##n,d = xob.shape
###theta=np.exp(lntheta).flatten()
##"""theta is a vector contain all hyperparameters: theta=[sig,length,varn]"""
## # computes the kernel using RBF method
##Kn = kern(xob,xob) + varn*np.eye(n)
### compute mean for posterior
##e=1e-8
##try:
##    invK_f=np.linalg.solve(Kn,yob)
##except:
##    while (np.linalg.matrix_rank(Kn) != Kn.shape[0]):
##        e*=10
##        Kn=Kn+e*np.eye(Kn.shape[0])
##        print ('singular matrix and e=%.3f' %(e))
##        
##    invK_f=np.linalg.solve(Kn,yob) 
##T1=-np.dot(yob.T,invK_f)/2.0
##T2=-np.log(np.linalg.det(Kn))/2.0
##T3=-d*np.log(2*np.pi)/2.0
##NML=-(T1+T2+T3)
#
#n,d = xob.shape
#h=0.0001
##theta=np.exp(lntheta)
#der= np.zeros_like(theta)
#Kn = kern(xob,xob) + varn*np.eye(n)
#e=0.1
#try:
#    invK_f=np.linalg.solve(Kn,yob)[:,None]
#except:
#    while (np.linalg.matrix_rank(Kn) != Kn.shape[0]):
#        e*=10
#        Kn=Kn+e*np.eye(Kn.shape[0])
#        print ('singular matrix and e=%.3f' %(e))
#    invK_f=np.linalg.solve(Kn,yob)[:,None]
#for i in range(len(lntheta)):
#    H=np.zeros(len(lntheta))
#    H[i]=H[i]+h
#    dKdth=lambda theta: (kerneln(X,X,theta+H)-kerneln(X,X,theta-H))/(2*h)
#    dKdlntheta=dKdth(theta)
#    A=np.dot(np.dot(invK_f,invK_f.T),dKdlntheta)-np.linalg.solve(Kn,dKdlntheta)
#    der[i]=0.5*np.trace(A)
#der=der*np.exp(lntheta)
##    
#
#
##%%
#
### Function for negatie marginal likelihood    
##def NML(lntheta,xob,yob):
##    
##    theta=np.exp(lntheta).flatten()
##    s=theta[0]
##    L = theta[1:3]
##    varn = theta[3]
##    """theta is a vector contain all hyperparameters: theta=[sig,length,varn]"""
##    kernel = ConstantKernel(constant_value=s) * RBF(length_scale=l)+ WhiteKernel(noise_level=varn)# +ConstantKernel() + Matern(length_scale=2, nu=3/2)
##    
##    n,d = xob.shape
##    #theta=np.exp(lntheta).flatten()
##    """theta is a vector contain all hyperparameters: theta=[sig,length,varn]"""
##     # computes the kernel using RBF method
##    Kn = kern(xob) 
##    # compute mean for posterior
##    e=1e-8
##    try:
##        invK_f=np.linalg.solve(Kn,yob)
##    except:
##        while (np.linalg.matrix_rank(Kn) != Kn.shape[0]):
##            e*=10
##            Kn=Kn+e*np.eye(Kn.shape[0])
##            print ('singular matrix and e=%.3f' %(e))
##            
##        invK_f=np.linalg.solve(Kn,yob) 
##    T1=-np.dot(yob.T,invK_f)/2.0
##    T2=-np.log(np.linalg.det(Kn))/2.0
##    T3=-d*np.log(2*np.pi)/2.0
##    NML=-(T1+T2+T3)
##    return NML
##    
#### Function for the gradient of negative marginal likelihood
##def NML_derive(lntheta,X,Y):
##    d = X.shape[1]
##    h=0.0001
##    theta=np.exp(lntheta)
##    der= np.zeros_like(theta)
##    Kn=kerneln(X,X,theta)
##    e=0.1
##    try:
##        invK_f=np.linalg.solve(Kn,Y)[:,None]
##    except:
##        while (np.linalg.matrix_rank(Kn) != Kn.shape[0]):
##            e*=10
##            Kn=Kn+e*np.eye(Kn.shape[0])
##            print ('singular matrix and e=%.3f' %(e))
##        invK_f=np.linalg.solve(Kn,Y)[:,None]
##    for i in range(len(lntheta)):
##        H=np.zeros(len(lntheta))
##        H[i]=H[i]+h
##        dKdth=lambda theta: (kerneln(X,X,theta+H)-kerneln(X,X,theta-H))/(2*h)
##        dKdlntheta=dKdth(theta)
##        A=np.dot(np.dot(invK_f,invK_f.T),dKdlntheta)-np.linalg.solve(Kn,dKdlntheta)
##        der[i]=0.5*np.trace(A)
##    der=der*np.exp(lntheta)
##    return -der  
#
#
#kerneln(X,X,theta)
#L = NML(np.log(theta),X,Y)
#print(L)
##%%
##NL = lambda lntheta: NML(lntheta,xob,yob)
#lntheta_01 = np.log(theta)
#def obj_func(lntheta, xob,yob, eval_gradient=True):
#    if eval_gradient:
#        lml, grad = negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=True)
#    return lml, grad
#
#NL = lambda lntheta: obj_func(lntheta, xob,yob, eval_gradient)
##%%
##lntheta_01 =np.atleast_2d( np.log(theta) )
#
#NL = lambda lntheta: negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=False)
#dNL = lambda lntheta: negative_log_marginal_likelihood(lntheta, xob,yob, eval_gradient=True)
#
##bnds=((0,1e4),(0.0,1.0),(0.0,1.0),(0,1))
#res = minimize(NL, lntheta_01, method='L-BFGS-B',jac=dNL, tol=1e-6)
##res = sp.optimize.fmin_bfgs(NL, x0=lntheta_01, maxiter=500)
#
##theta_ei1=sp.optimize.fmin_bfgs(NL, lntheta_01)

