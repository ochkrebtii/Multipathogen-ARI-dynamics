"""
                    PARALLEL TEMPERING MCMC
VERSION 3 With Linear Model
In this version we incorporate the data of FLU and RSV with linear model (Binomial Distribution)
     
This algorithm can be found in:
        Golightly, Andrew, Daniel A. Henderson, and Christopher Sherlock. 
        "Efficient particle MCMC for exact inference in stochastic biochemical 
        network models through approximation of expensive likelihoods." (2012).

This implementation is for a SIR model with two pathogens.
The data are from San Luis Potosi - MEX

"""
#==============================================================================

from __future__ import division
import numpy as np
np.set_printoptions(threshold=1000)
import scipy.stats as ss
from odeintw import odeintw # required to solve matrix differential equations
import random
import pylab as pl
import matplotlib
import matplotlib.pyplot as plt
#import corner # correlation plots; may cause issues running on the cluster

class HaltException(Exception): pass  # define my own exception


#==============================================================================
#==============================================================================
# Forward Model
#==============================================================================
#==============================================================================


def CovSol(time,State,pars):
    Sol = odeintw(ODE_vec_field,State,time,args=(pars,))
    return Sol
    

def ODE_vec_field(State,t,pars):
    """System coupled differential equations:
       Matrix A(phi,beta) and B(phi,beta) are time dependent through phi(t) and beta(t),
       so we should solve the deterministic equatioP1ge0nDr1verns at the same time that the equation for C.
    Input:
         State: Matrix (dim+1) x dim (dim = 8 -- number of deterministic equations)
         t  : time
         pars  : parameters
    Output:
        K: Matrix (dim +1) x dim
             K[0,:]  Corresponds to the solution of the deterministic equation
             K[1:,:] Correponds to the solution of matrix C
    """
    mu    = 1.0/70   # death/birth rate
    gamma  = 365.0/7    # recovery rate
    vac =  0.0
    beta1 = np.array(pars[0]) # define parameters
    beta2 = np.array(pars[1])
    sig1 = np.array(pars[2])
    sig2 = np.array(pars[3])
    lam1 = np.array(State[0,1]+State[0,6])
    lam2 = np.array(State[0,3]+State[0,4])
    
    K  = np.zeros([9,8]) #this part correspond to the deterministic equation
    K[0,0] = mu-beta2*lam2*State[0,0]-beta1*lam1*State[0,0]-(mu+vac)*State[0,0]
    K[0,1] = beta1*lam1*State[0,0]-(gamma+mu)*State[0,1]
    K[0,2] = gamma*State[0,1]+vac*State[0,0]-sig2*beta2*lam2*State[0,2]-mu*State[0,2]
    K[0,3] = beta2*lam2*State[0,0]-(gamma+mu)*State[0,3]
    K[0,4] = sig2*beta2*lam2*State[0,2]-(gamma+mu)*State[0,4]
    K[0,5] = gamma*State[0,3]-sig1*beta1*lam1*State[0,5]-mu*State[0,5]
    K[0,6] = sig1*beta1*lam1*State[0,5]-(gamma+mu)*State[0,6]
    K[0,7] = gamma*State[0,6]+gamma*State[0,4]-mu*State[0,7]
    
    A = np.zeros([8,8])
    A[0,0] = -beta2*lam2-beta1*lam1-vac-mu
    A[0,1] = -beta1*State[0,0]
    A[0,3] = -beta2*State[0,0]
    A[0,4] = -beta2*State[0,0]
    A[0,6] = -beta1*State[0,0]
    A[1,0] =  beta1*lam1
    A[1,1] =  beta1*State[0,0]-(gamma+mu)
    A[1,6] =  beta1*State[0,0]
    A[2,0] =  np.array(vac)
    A[2,1] =  np.array(gamma)
    A[2,2] = -beta2*lam2*sig2-mu
    A[2,3] = -beta2*State[0,2]*sig2
    A[2,4] = -beta2*State[0,2]*sig2
    A[3,0] =  beta2*lam2
    A[3,3] =  beta2*State[0,0]-(mu+gamma)
    A[3,4] =  beta2*State[0,0]
    A[4,2] =  beta2*lam2*sig2
    A[4,3] =  beta2*State[0,2]*sig2
    A[4,4] =  beta2*State[0,2]*sig2-(gamma+mu)
    A[5,1] = -beta1*State[0,5]*sig1
    A[5,3] =  np.array(gamma)
    A[5,5] = -beta1*lam1*sig1-mu
    A[5,6] = -beta1*State[0,5]*sig1
    A[6,1] =  beta1*State[0,5]*sig1
    A[6,5] =  beta1*lam1*sig1
    A[6,6] =  beta1*State[0,5]*sig1-(gamma+mu)
    A[7,4] =  np.array(gamma)
    A[7,6] =  np.array(gamma)
    A[7,7] =  np.array(-mu)
    
    B = np.zeros([8,8])  # define and fill in matrix B
    B[0,0] = beta2*State[0,0]*lam2+beta1*State[0,0]*lam1+State[0,0]*(mu+vac)+mu
    B[1,1] = beta1*State[0,0]*lam1+State[0,1]*(mu+gamma)
    B[2,2] = gamma*State[0,1]+vac*State[0,0]+beta2*State[0,2]*lam2*sig2+mu*State[0,2]
    B[3,3] = beta2*State[0,0]*lam2+State[0,3]*(mu+gamma)
    B[4,4] = State[0,4]*(mu+gamma)+beta2*State[0,2]*lam2*sig2
    B[5,5] = gamma*State[0,3]+mu*State[0,5]+beta1*State[0,5]*lam1*sig1
    B[6,6] = (gamma+mu)*State[0,6]+beta1*State[0,5]*lam1*sig1
    B[7,7] = gamma*State[0,4]+gamma*State[0,6]+mu*State[0,7]
    B[0,1] = -beta1*State[0,0]*lam1 
    B[1,0] = np.array(B[0,1])
    B[0,2] = -vac*State[0,0]
    B[2,0] = np.array(B[0,2])
    B[0,3] = -beta2*lam2*State[0,0]
    B[3,0] = np.array(B[0,3])
    B[1,2] = -gamma*State[0,1]
    B[2,1] = np.array(B[1,2])
    B[2,4] = -beta2*lam2*sig2*State[0,2]
    B[4,2] = np.array(B[2,4])
    B[3,5] = -gamma*State[0,3]
    B[5,3] = np.array(B[3,5])
    B[4,7] = -gamma*State[0,4]
    B[7,4] = np.array(B[4,7])
    B[5,6] = -beta1*lam1*sig1*State[0,5]
    B[6,5] = np.array(B[5,6])
    B[6,7] = -gamma*State[0,6]
    B[7,6] = np.array(B[6,7])

    C = np.array(State[1:,:])
    K[1:,:]  = np.dot(C,np.transpose(A)) + np.dot(A,C) + np.array(B)
    
    return K      


#==============================================================================
#==============================================================================
# PTMCMC functions
#==============================================================================
#==============================================================================


def qqq(p,cov,ind,Omega): 
     """ Proposal mechanism """   
     if ind == 1: # if we are proposing the initial states
         pout = np.zeros(p.shape[0])
         while np.sum(pout)!= 1.0:
             randind = random.sample(range(p.shape[0]),1)
             restinds = np.delete(range(p.shape[0]),randind)
             c = np.array(cov[restinds,:][:,restinds])
             pout[restinds] = np.random.multivariate_normal(p[restinds],c)
             pout[randind] = 1- np.sum(pout[restinds])
     else:   # if we are proposing any other variables
         pout = np.random.multivariate_normal(p,cov)      
     return pout

## Alternative proposal mechanism
#def qqq(p,cov,ind,Omega): 
#    """ Proposal mechanism """   
#    if ind == 1: # if we are proposing the initial states
#        pout = -1.0*np.ones(p.shape[0])
#        randind = random.sample(range(p.shape[0]),1)
#        restinds = np.delete(range(p.shape[0]),randind)
#        while pout[randind] < 0:
#            c = np.array(cov[restinds,:][:,restinds])
#            pout[restinds] = np.random.multivariate_normal(p[restinds],c)
#            pout[randind] = 1- np.sum(pout[restinds])
#    else:   # if we are proposing any other variables
#        pout = np.random.multivariate_normal(p,cov)      
#    return pout


def rho(log_like1,log_prior1,log_like2,log_prior2):
    """Compute Metropolis hasting acceptance rate """
    log_rate = log_like2 + log_prior2 - log_like1 - log_prior1
    if log_rate < 0.0:
        return np.exp(log_rate)
    else:
        return 1.0


def rhoswap(temp1,log_like1,temp2,log_like2):
    """Compute Parallel Tempering swap acceptance rate """
    log_rate = temp1*log_like2 + temp2*log_like1 - temp2*log_like2 - temp1*log_like1
    if log_rate < 0.0:
        return np.exp(log_rate)
    else:
        return 1.0


def propcovadapt(arate,pcov,blockinds,pars):
    """proposal covariance adaptation during burn-in phase""" 
    temp = []
    if pars.shape[1] == pcov.shape[0]:
        pars = np.transpose(pars)
    
    r = np.nan*np.zeros((pars.shape[0],pars.shape[0]))
    
    for ind in range(len(blockinds)):
        try:
            #cor = np.corrcoef(np.array(pars[blockinds[ind],:]))
            var = np.var(np.transpose(pars[blockinds[ind],:]),0)       
            eps = 1e-42 
            getind = np.argwhere(var < eps)
            repl = 0.01*pcov[:,blockinds[ind]][blockinds[ind],:] 
            var[np.array(getind)] = np.array(repl[np.array(getind),np.array(getind)])
    
            if arate[ind] > 0.01:
                factor = arate[ind]
            else:
                factor = 0.01
            
            weighted_avg = np.max(pcov[:,blockinds[ind]][blockinds[ind],:])#(0.75*np.max(pcov[:,blockinds[ind]][blockinds[ind],:]) + 0.25*np.max(var))
            
            if pcov[:,blockinds[ind]][blockinds[ind],:].shape[0] < 4 and (arate[ind]<0.18 or arate[ind]>0.28):
                temp.append((factor/0.23)*weighted_avg*np.diag(np.array(var)/np.max(var)))
            elif pcov[:,blockinds[ind]][blockinds[ind],:].shape[0] >= 4 and (arate[ind]<0.18 or arate[ind]>0.28):
                temp.append((factor/0.23)*weighted_avg*np.diag(np.array(var)/np.max(var)))
            else:
                temp.append(weighted_avg*np.diag(np.array(var)/np.max(var)))

## In some cases, we may adapt off-diagnoal entries (not suggested here)               
#            if (ind == 0) or (ind == 2):  # adapt off-diagnoal entries
#                for row in range(len(blockinds[ind])):        
#                    for col in range(row):            
#                        if abs(cor[row,col])>0.6: 
#                            temp[ind][row,col] = np.sqrt(temp[ind][row,row]*temp[ind][col,col])*np.sign(cor[row,col])*(abs(cor[row,col])-0.3)   
#                            temp[ind][col,row] = np.array(temp[ind][row,col])
#                        else:
#                            temp[ind][row,col],temp[ind][col,row] = 0,0  
                
        except:
            temp[ind] = pcov[:,blockinds[ind]][blockinds[ind],:]
        
        rind = 0
        cind = 0
        for row in blockinds[ind]:        
                for col in blockinds[ind]:           
                    r[row,col] = temp[ind][rind,cind] 
                    cind = np.array(cind) + 1
                rind = np.array(rind) + 1
                cind = 0
    
    return r


def is_pos_sdef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def sqexp(x,sigvar):
    """ Produce a squared exponential covariance matrix """
    A, B = np.meshgrid(x,x)
    lenscale = 0.02
    nugget = 1e-8  # a small nugget effect (used for numerical reasons)
    covmat = sigvar*np.exp(-0.5*((A-B)/lenscale)**2) + nugget*np.eye(len(x),len(x))
    return covmat


def logapriori(p): 
    """ Compute the log prior for all parameters """  
    try:         
        pr_lpdf  = ss.gamma.logpdf(p[0],20.0,scale=3.0) + ss.gamma.logpdf(p[1],20.0,scale=3.0)  #betai - baseline transmission rate
        pr_lpdf = np.array(pr_lpdf) + ss.gamma.logpdf(p[2],10.0,scale=0.1) + ss.gamma.logpdf(p[3],10.0,scale=0.1)  #sig1 and sig2 - cross immunities
        pr_lpdf = np.array(pr_lpdf) + ss.gamma.logpdf(p[13],1.0,scale=1.0) #sigsq 
        pr_lpdf = np.array(pr_lpdf) + ss.beta.logpdf(p[14],1.0,1.0)  #r - reporting rate     
        pr_lpdf = np.array(pr_lpdf) + ss.beta.logpdf(p[15],1.0,1.0)  # delta mean
        pr_lpdf = np.array(pr_lpdf) + ss.beta.logpdf(p[16],1.0,1.0) # AR1 model parameter for delta
        pr_lpdf = np.array(pr_lpdf) + ss.gamma.logpdf(p[17],1.0,scale=1.0)  # variance inflation factor
    except:
        return float('-inf')*np.ones(1)  
    return pr_lpdf


def SMC(AGGdata,INFdata,RSVdata,Omega,p):
    """Sequential Monte Carlo"""
    
    # compute the log prior        
    ndata = len(AGGdata)
    logprior = np.array(logapriori(p)) 
    if np.isnan(logprior) or np.isinf(logprior): 
        return float('-inf')*np.ones(1), float('-inf')*np.ones(1), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata))    
                
    #define observation operators
    G1          = np.array([0,1,0,0,0,0,1,0,0])   #INF = phi1 + phi6
    G2          = np.array([0,0,0,1,1,0,0,0,0])   #RSV = phi3 + phi4 
    G3          = np.array([0,0,0,0,0,0,0,0,1])
    G           = G1+G2+G3
    
    # set up data and parameters
    C0          = np.array(p[12])
    ssq         = np.array(p[13]*(Omega**2))
    r           = np.array(p[14])
    c           = np.array(p[15])
    kappa       = np.array(C0)
    nu          = np.array(p[16])
    vinf        = np.array(p[17])
    phi0        = np.array(p[range(4,12)])
    dim         = len(phi0) + 1
    pc          = 266761/Omega      #proportion of children <5yrs in the population
    #pcr         = 19.8/1000         # proportion of ARI reports in children
    yvec        = np.array(AGGdata)
    dt          = 1.0/ndata
     
    # initialize mean and covariance for analysis and solver
    z           = np.append(np.array(phi0),np.array(c))
    m           = np.array(Omega*z)
    C           = np.array(C0*np.eye(dim,dim))
    C[dim-1,dim-1] = np.array(kappa*np.sqrt(Omega)) # see algorithm regarding Omega
    V           = np.array(Omega*C) 
    
    # forecast/analysis/smoothing distributions transformed by observation process
    forecast_obs_mean      = np.zeros([3,ndata])
    forecast_obs_var       = np.zeros(ndata)
    analysis_mean          = np.zeros([dim,ndata])
    analysis_var           = np.zeros((dim,dim,ndata))
    smooth_draw            = (-1.0)*np.ones([dim,ndata])
    forecast_obs_draw      = (-1.0)*np.ones([3,ndata])
    
    loglike         = 0 
    Init_SolCOV     = np.zeros([(dim-1)+1,(dim-1)])
  
    for i in range(ndata):   #sampling from p(y|pars) using forward recursion
        
        if i>0:
            # forecast distribution x_t|y_1:t-1 mean and variance         
            tsteps            = np.linspace((i-1)*dt,dt*i,1000)
            Init_SolCOV[0,:]  = np.array(m[0:dim-1]/Omega)
            Init_SolCOV[1:,:] = np.array(V[0:dim-1,0:dim-1]/Omega)
            SolCOV            = CovSol(tsteps,Init_SolCOV,p[0:4])
            z[0:dim-1]        = np.array(SolCOV[-1,0,:]) 
            z[dim-1]          = np.array(c + nu*m[dim-1]/Omega) # lagged background model with drift
            #z[dim-1]          = np.array(c*(1-nu) + nu*m[dim-1]/Omega) # AR1 background model with no drift
            C[0:dim-1,0:dim-1]= np.array(SolCOV[-1,1:,:])
            C[dim-1,dim-1]    = np.array(C[dim-1,dim-1]*nu**2 + kappa*np.sqrt(Omega))  # Omega is here on purpose
            
            if any(y < 0 for y in np.diag(C)) or any(x < 0 for x in z): 
                return float('-inf')*np.ones(1),float('-inf')*np.ones(1), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata))     


        forecast_obs_mean[0,i] =  np.array(r*Omega*np.dot(G1,z))
        forecast_obs_mean[1,i] =  np.array(r*Omega*np.dot(G2,z))
        forecast_obs_mean[2,i] =  np.array(r*Omega*np.dot(G3,z))
        forecast_obs_var[i]    =  np.array((r**2)*Omega*np.dot(np.dot(G,C),G) + ssq)  #Omega in first term not squared on purpose, see paper
        
        # p(y|theta) component of likelihood
        loglike =  np.array(loglike) + ss.norm.logpdf(yvec[i], loc=np.sum(forecast_obs_mean[:,i]), scale=np.sqrt(forecast_obs_var[i]))
        
        # step-ahead predictive distribution
        m = np.array(Omega*z) + np.array(r*Omega)*np.dot(G,C)*(np.array(forecast_obs_var[i])**(-1))*(yvec[i] - np.sum(forecast_obs_mean[:,i]))
        V = np.array(Omega*C) - np.array((r*Omega)**2)*(np.matrix(np.dot(G,C)*(np.array(forecast_obs_var[i])**(-1))).T)*np.matrix(np.dot(C,G))           
        
        # analysis distribution x_t|y_1:t mean and variance 
        analysis_mean[:,i] = np.array(m)
        analysis_var[:,:,i] = np.array(V) # Missing term of forecast variance is part of the approximation
        
        if any(x < 0 for x in analysis_mean[:,i]) or is_pos_sdef(analysis_var[:,:,i])==False: 
            return float('-inf')*np.ones(1),float('-inf')*np.ones(1), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata))
        
        try:
            counter = 0
            while any(s < 0 for s in smooth_draw[:,i]) and counter < 100:
                smooth_draw[:,i] = np.random.multivariate_normal(mean=analysis_mean[:,i], cov=analysis_var[:,:,i])
                counter = np.array(counter) + 1
            if counter == 100:
                return float('-inf')*np.ones(1),float('-inf')*np.ones(1), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)) 
            counter = 0
            while any(f < 0 for f in forecast_obs_draw[:,i]) and counter < 100:
                forecast_obs_draw[:,i] = np.random.multivariate_normal(mean=forecast_obs_mean[:,i], cov=np.eye(3,3)*(forecast_obs_var[i]/3))
                counter = np.array(counter) + 1
            if counter == 100:
                return float('-inf')*np.ones(1),float('-inf')*np.ones(1), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)) 
        except:
            return float('-inf')*np.ones(1),float('-inf')*np.ones(1), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)), 0*np.ones(shape=(3,ndata)) 


#        while any(s < 0 for s in smooth_draw[:,i]):
#            smooth_draw[:,i] = np.random.multivariate_normal(mean=analysis_mean[:,i], cov=analysis_var[:,:,i])
#        while any(f < 0 for f in forecast_obs_draw[:,i]):
#            forecast_obs_draw[:,i] = np.random.multivariate_normal(mean=forecast_obs_mean[:,i], cov=np.eye(3,3)*(forecast_obs_var[i]/3))          

#    # sampling from p(x|y,pars) using backward recursion
#    smooth_mean[:,-1] = analysis_mean[:,-1]
#    smooth_var[:,:,-1] = analysis_var[:,:,-1]
#    
#    while any(s < 0 for s in smooth_draw[:,-1]):
#        smooth_draw[:,-1] = np.random.multivariate_normal(smooth_mean[:,-1],smooth_var[:,:,-1])
#    
#    for i in reversed(range(ndata-1)):      
#        
#        Jt = analysis_var[:,:,i]*np.linalg.inv(forecast_var[:,:,i+1])
#        smooth_mean[:,i] = analysis_mean[:,i] + np.dot(Jt,smooth_mean[:,i+1] - forecast_mean[:,i+1])
#        smooth_var[:,:,i] = analysis_var[:,:,i] + np.mat(Jt)*np.mat(smooth_var[:,:,i+1] - forecast_var[:,:,i+1])*np.mat(Jt)  
#        
#        while any(s < 0 for s in smooth_draw[:,i]):
#            smooth_draw[:,i] = np.random.multivariate_normal(smooth_mean[:,i],smooth_var[:,:,i])
#        print(i)
    
    muINF = np.array(np.dot(G1,smooth_draw)*pc*r*0.005)
    muRSV = np.array(np.dot(G2,smooth_draw)*pc*r*0.005)
    muBAC = np.array(np.dot(G3,smooth_draw)*pc*r*0.005)
    aINF = vinf*muINF
    aRSV = vinf*muRSV
    aBAC = vinf*muBAC
    bINF = (1/vinf)*np.ones(len(aINF))
    bRSV = (1/vinf)*np.ones(len(aRSV))
    bBAC = (1/vinf)*np.ones(len(aBAC))
    prINF = 1/(1+bINF)
    prRSV = 1/(1+bRSV)
    prBAC = 1/(1+bBAC)
    
    # E(negbin) = mu, V(negbin) = mu(1+1/vinf)
    # variance is inflated relative to Poisson and Binomial models
    
    loglike =  np.array(loglike) + np.sum(ss.nbinom.logpmf(INFdata,n = aINF, p = prINF))    
    loglike =  np.array(loglike) + np.sum(ss.nbinom.logpmf(RSVdata,n = aRSV, p = prRSV))
    loglike =  np.array(loglike) + np.sum(ss.nbinom.logpmf(BACdata,n = aBAC, p = prBAC))
    
#    plt.plot(RSVdata)    
#    plt.plot(muRSV)
#    plt.plot(ss.nbinom.logpmf(RSVdata,n = aRSV, p = prRSV))
#
#    plt.plot(INFdata)
#    plt.plot(muINF)
#    plt.plot(ss.nbinom.logpmf(INFdata,n = aINF, p = prINF))
#    
#    plt.plot(BACdata)
#    plt.plot(muBAC)
#    plt.plot(ss.nbinom.logpmf(BACdata,n = aBAC, p = prBAC))

    smooth_obs_mean = np.stack([r*np.dot(G1,analysis_mean),r*np.dot(G2,analysis_mean),r*np.dot(G3,analysis_mean)])
    smooth_obs_draw = np.stack([r*np.dot(G1,smooth_draw),r*np.dot(G2,smooth_draw),r*np.dot(G3,smooth_draw)])
    postpred_obs_draw = np.array(forecast_obs_draw)

    return loglike, logprior, smooth_obs_mean, smooth_obs_draw, postpred_obs_draw
            

#==============================================================================
# Plotting
#==============================================================================


def pathplots(INF,RSV,Back,AGGdata,INFdata,RSVdata,name):
   """ Produce plots of sample paths """    
   fig  = pl.figure(facecolor='white')
   fig.subplots_adjust(wspace=0.05)
   nticks = 6
   t_data = range(52)
   aggpathfig  = fig.add_subplot(1,2,1) 
   aggpathfig.title.set_fontsize(20)
   aggpathfig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
   aggpathfig.set_xlim([0.0,52.0])
   aggpathfig.set_ylim([0.0,30000.0])
   
   if INF.shape==(52,):
       aggpathfig.plot(t_data,INF+RSV+Back,color='r', label='SMC',lw=2.0)   
   else:
       q1 = np.percentile(INF+RSV+Back,2.5,axis=0)
       q2 = np.percentile(INF+RSV+Back,97.5,axis=0)
       aggpathfig.fill_between(t_data,q1,q2,color='lightgray',alpha=0.9)
       nplots = min(INF.shape[0],20)
       for i in range(0,INF.shape[0],int(INF.shape[0]/nplots)):   
           aggpathfig.plot(t_data,INF[i,:]+RSV[i,:]+Back[i,:],color='k', label='SMC',lw=2,alpha=0.1+1/nplots)   
       
   aggpathfig.plot(t_data,AGGdata, '.--',color='g',label='Data',lw=2,alpha=1)   
   aggpathfig.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks    
   
   disaggfig = fig.add_subplot(1,2,2)
   par1  = disaggfig.twinx() 
   par1.fill_between(t_data,INFdata,0,facecolor='r', alpha=0.6, label= 'INF')
   par1.fill_between(t_data,RSVdata,0,facecolor='b', alpha=0.4, label= 'RVS')
   par1.fill_between(t_data,BACdata,0,facecolor='k', alpha=0.2, label= 'Backg')
   par1.set_ylim([0.0,40.0])
   disaggfig.title.set_fontsize(20)
   disaggfig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
   disaggfig.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
   disaggfig.set_xlim([0.0,52.0])
   disaggfig.set_ylim([0.0,30000.0])
   disaggfig.set_yticks([])
   
   if INF.shape==(52,):
       disaggfig.plot(t_data,INF ,'-.',color='r', label='SMC Influenza ',lw=2,alpha=1.0)
       disaggfig.plot(t_data,RSV ,'--',color='b',label='SMC RSV',lw=2, alpha=1.0)
       disaggfig.plot(t_data,Back ,'--',color='k',label='SMC RSV',lw=2, alpha=1.0) 
   else:
       for i in range(0,INF.shape[0],int(INF.shape[0]/nplots)):   
           disaggfig.plot(t_data,INF[i,:],color='r', label='SMC',lw=2,alpha=0.1+1/nplots)   
           disaggfig.plot(t_data,RSV[i,:],color='b', label='SMC',lw=2,alpha=0.1+1/nplots)  
           disaggfig.plot(t_data,Back[i,:],color='k', label='SMC',lw=2,alpha=0.1+1/nplots)  
           
       q1 = np.percentile(INF,2.5,axis=0)
       q2 = np.percentile(INF,97.5,axis=0)
       disaggfig.fill_between(t_data,q1,q2,color='lightgray',alpha=0.5)
   
       q1 = np.percentile(RSV,2.5,axis=0)
       q2 = np.percentile(RSV,97.5,axis=0)
       disaggfig.fill_between(t_data,q1,q2,color='lightgray',alpha=0.5)
       
       q1 = np.percentile(Back,2.5,axis=0)
       q2 = np.percentile(Back,97.5,axis=0)
       disaggfig.fill_between(t_data,q1,q2,color='lightgray',alpha=0.5)
   
   fig = pl.gcf()
   fig.autofmt_xdate()
   pl.rcParams['axes.grid'] = True
   pl.rcParams['grid.linestyle'] = ':'
   fig.text(0.020,0.5,'Aggregated ARI Reports ',va='center',rotation='vertical',fontsize=20)
   fig.text(0.5,0.1,'Time (Weeks)',ha='center',fontsize=20)
   pl.autoscale(enable=True, axis='x', tight=True)
   fig.text(0.93,0.5,"Laboratory Samples",va='center',rotation='vertical',fontsize=20)
   disaggfig.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks
   fig.set_size_inches(18.5, 10.5)
   fig.savefig("Output/" + str(name) + ".png")#,dpi=100)
   pl.close(fig)
   
   
   
def traceplots(Parameters,parnames,logpost,name):
   """ Produce sample trace plots """
   fig  = pl.figure(facecolor='white', figsize=(20,10))
   fig.subplots_adjust(left=0.08,right=0.96,wspace=0.05,hspace=0.2,top=0.88,bottom=0.8)
   a = 4
   b = 5
   
   for ax in range(len(parnames)):
       axfig  = fig.add_subplot(a,b,ax+1) 
       axfig.plot(Parameters[:,ax],lw=1.5)
       axfig.set_xlabel(parnames[ax])
   
   axfig  = fig.add_subplot(a,b,len(parnames)+1)
   axfig.plot(logpost,lw=1.5)
   axfig.set_xlabel('Log posterior')

   pl.tight_layout()
   fig.set_size_inches(30, 15)
   fig.savefig("Output/" + str(name) + ".png")#,dpi=100)
   pl.close(fig)
   
   

#def corrplots(p,pnames,ind):
#   """ Produce posterior sample correlation plots """
#   fig = corner.corner(p, labels=pnames,label_kwargs={'fontsize':25},
#                 truths=None,max_n_ticks=1, smooth=2.0,color='k',truth_color='b',
#                 quantiles=[0.05, 0.5, 0.95]) #
#   
#   pl.tight_layout()
#   fig.set_size_inches(15, 15)
#   fig.savefig("Output/corplots-block" + str(ind) + '.png')
#   plt.close(fig)
   


#==============================================================================
#==============================================================================
# Load data from San Luis Potosi
#==============================================================================
#==============================================================================

Omega  = 2.5*10**6  # population size
window = 52         # number of weeks in one year

# epidemic begins at week 32
# first year we have full data is 2003-2004 (starting week 32 of 2003, ending week 31 of 2004)
# last year we have full data is 2008-2009 

year   = 1          # year 1 is 2003-2004 and year 6 is 2008-2009  
                    # for simulation, the year should always be 1
                    
#data = np.array(np.genfromtxt("Data/iras_data_full_load.csv", delimiter=",")) # for real data
data = np.array(np.genfromtxt("Data/iras_data_full_load_sim.csv", delimiter=","))# for simulated data (all years)

# col 0 is the week number     
# col 1 is the aggregate reports for kids under 5
# col 2 is the total aggregate reports, including kids
# col 3 is number of rsv virological samples
# col 4 is number of inf virological samples
# col 5 is number of "other" virological samples

yrrange = range((year-1)*window,year*window)
AGGdata = data[yrrange,2]
AGGkids = data[yrrange,1]
INFdata = data[yrrange,4]
RSVdata = data[yrrange,3]
BACdata = data[yrrange,5]

ttt = np.linspace(0,1,len(AGGdata)) 
ndata = len(AGGdata)
    

#==============================================================================
#==============================================================================
# Set up PTMCMC
#==============================================================================
#==============================================================================

Nchains = 4               # number of parallel tempering chains
Thinby = 10               # how many iterations to thin by
Itersfactor = 500000      # how many thinned draw to sample in total
Itercheckfactor = 50      # number of thinned draws before each covariance adaptation
lagfactor = 200           # plotting: how many thinned draws to look at

Iters = Itersfactor*Thinby  #number of iterations, must be a multiple on Thinby 
ThinIterInds = np.arange(0,Iters+1,Thinby)
Itercheck = Itercheckfactor*Thinby  # check accepts at every Itercheck iterations
burnfactor = np.int(np.ceil(Itersfactor/2)) # plotting: how many thinned draws to burn 

if Nchains == 1:  # set chain temperatures
    temp_scale = np.array([1.0])
else:
    temp_scale = np.linspace(0.5,1.0,num=Nchains ,endpoint=True)


print("------------------------------------------------------------------") 
print("total number of PTMCMC chains = ", Nchains)
print("temperature schedule is: ")
print(temp_scale)
print("total number of iterations = ", Iters)
print("check covariance every ", Itercheck, " iterations")
print("draws saved at every ", Thinby, " iterations")
print("------------------------------------------------------------------") 

parnames = ['${\\beta}_{10}$','${\\beta}_{20}$','${\\sigma}_{1}$','${\\sigma}_{2}$',
            '$X_{ss}(0)$','$X_{is}(0)$','$X_{rs}(0)$','$X_{si}(0)$',
            '$X_{ri}(0)$','$X_{sr}(0)$','$X_{ir}(0)$','$X_{rr}(0)$',
            '$C_0$','${\\Sigma}$','$r$','$c$',
            '${\\nu}$','$V_{inf}$']                 

npars = len(parnames)

# Load the last adapted proposal covariance from the saved file, or use hard-coded covariance
try:
    proposalcovtemp = np.array(np.genfromtxt("Output/pt_propcovar.csv", delimiter=","))
    proposalcov = np.zeros([Nchains,len(parnames),len(parnames)])
    for chain in range(Nchains):
        proposalcov[chain,:,:] = proposalcovtemp[chain*len(parnames):(chain+1)*len(parnames),:]
    print("Starting proposal covariances are loaded from file")
except:
    proposalcovtemp = np.diag([1.0 , 1.0, 0.1, 0.1,
                               1e-12 , 1e-12, 1e-12, 1e-12,1e-12 , 
                               1e-12, 1e-12, 1e-12,
                               1e-12, 1e-12, 1e-6, 1e-6, 
                               1e-12, 1e-12])     
    proposalcov = np.tile(proposalcovtemp,(Nchains,1,1))
    print("Starting proposal covariance is hard coded into script")
    
# Load the last set of parameters from the saved file, or use hard-coded parameters
try:
    thetatemp = np.genfromtxt("Output/parchain.csv", delimiter=",")
    theta = np.zeros([Nchains,len(parnames)])
    for chain in range(Nchains):
        theta[chain,:] = thetatemp[chain*len(parnames):(chain+1)*len(parnames)]
    print ("Starting values of theta and delta are loaded from file:")
except:
    start_pt = np.zeros([2,len(parnames)]) # start at one of two distinct regions of the parameter space
    start_pt[0,] = np.array([ 70, 70, 0.6, 0.2, 
                          9.12854214e-01, 9.01699835e-06,   1.69383245e-02,   8.66398896e-05,   
                          4.81188964e-05,   1.10513557e-02,   3.06268657e-07,  0.059012023746992992,
                          1e-2, 1e-6, 1e-1, 1e-2,
                          0.2, 0.1])
    start_pt[1,] = np.array(start_pt[0,])
    theta = np.zeros([Nchains,len(parnames)])
    for chain in range(Nchains):
        if chain < 2:
            ind = chain
        else:
            ind = np.random.randint(2, size=1)
        theta[chain,] = np.array(start_pt[ind,])	
    print("Starting value of theta is hard coded into script:")
    
                    
print(theta)
        

blockinds = [] # define sampling block indices for the parameters as a list 

blockinds.append([0,1,2,3]) # SDE model parameters
blockinds.append([4,5,6,7,8,9,10,11])  # SDE initial states np.array(range(4,12))
blockinds.append([13,14,15,16,17]) # Probability model parameters

gibbsinds = [] #gibbs

# initialize counters that are used in the loops
totcon = np.tile(len(blockinds)*[1.0],(Nchains,1))
acccon = np.tile(len(blockinds)*[1.0],(Nchains,1))
acc = np.tile(len(blockinds)*[1.0],(Nchains,1))
swaptot = len(temp_scale)*[1.0]
swapcon = len(temp_scale)*[1.0]
swapacc = len(temp_scale)*[1.0]
thinnum = 0


print("------------------------------------------------------------------") 

# pre-allocate variables for PTMCMC
log_like = np.zeros([Nchains,1])
log_prior = np.zeros([Nchains,1])
log_post = np.zeros([Nchains,1])
smooth_mean = np.zeros([Nchains,3,len(AGGdata)])
smooth_sample = np.zeros([Nchains,3,len(AGGdata)])
postpred_sample = np.zeros([Nchains,3,len(AGGdata)])
log_like_proposed = np.zeros([Nchains,1])
log_prior_proposed = np.zeros([Nchains,1])
smooth_mean_proposed = np.zeros([Nchains,3,len(AGGdata)])
smooth_sample_proposed = np.zeros([Nchains,3,len(AGGdata)])
postpred_sample_proposed = np.zeros([Nchains,3,len(AGGdata)])

INFsmean = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
RSVsmean = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
BACKsmean = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
INFsdraw = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
RSVsdraw = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
BACKsdraw = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
INFppred = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
RSVppred = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])
BACKppred = np.zeros([Nchains,len(ThinIterInds),len(AGGdata)])

Logpost = np.zeros([Nchains,len(ThinIterInds)])
Parchainall = np.zeros([Nchains,len(ThinIterInds),len(parnames)])
Parchaincheck = np.zeros([Nchains,Itercheck,len(parnames)])

# Compute starting values for MCMC
for chain in range(Nchains):
    log_like[chain],log_prior[chain],smooth_mean[chain,:,:],smooth_sample[chain,:,:],postpred_sample[chain,:,:] = SMC(AGGdata,INFdata,RSVdata,Omega,np.array(theta[chain,:]))

#pathplots(smooth_mean[-1,0,:],smooth_mean[-1,1,:],smooth_mean[-1,2,:],AGGdata,INFdata,RSVdata,"initial-forecast-mean-sd")
#pathplots(smooth_sample[-1,0,:],smooth_sample[-1,1,:],smooth_sample[-1,2,:],AGGdata,INFdata,RSVdata,"initial-smooth-sample-sd")            
#pathplots(postpred_sample[-1,0,:],postpred_sample[-1,1,:],postpred_sample[-1,2,:],AGGdata,INFdata,RSVdata,"initial-post-pred-sample-sd")            

log_post = np.multiply(temp_scale,np.transpose(log_like)) + np.transpose(np.array(log_prior))
if np.isnan(np.sum(log_post)) or np.isinf(np.sum(log_post)):
        raise HaltException("Must start with finite log-posterior") 

INFsmean[:,0,:] = np.array(smooth_mean[:,0,:])
RSVsmean[:,0,:] = np.array(smooth_mean[:,1,:])
BACKsmean[:,0,:] = np.array(smooth_mean[:,2,:])
INFsdraw[:,0,:] = np.array(smooth_sample[:,0,:])
RSVsdraw[:,0,:] = np.array(smooth_sample[:,1,:])
BACKsdraw[:,0,:] = np.array(postpred_sample[:,2,:])
INFppred[:,0,:] = np.array(postpred_sample[:,0,:])
RSVppred[:,0,:] = np.array(postpred_sample[:,1,:])
BACKppred[:,0,:] = np.array(postpred_sample[:,2,:])

Logpost[:,0] = np.array(log_post)
Parchainall[:,0,:] = np.array(theta)

print(["initial log-posterior:",Logpost[:,0]])

#==============================================================================
#==============================================================================
# Run PTMCMC
#==============================================================================
#==============================================================================

 
for i in range(Iters):  
       
#    print(i)
    
    theta_proposed = np.array(theta)
        
    if Nchains > 1 and ss.uniform.rvs() < 0.3:  # propose a swap 30% of the time
        chain_swap = np.sort(np.random.choice(Nchains,2,replace=0))
        swaptot[chain_swap[0]] = np.array(swaptot[chain_swap[0]])+1
        swaptot[chain_swap[1]] = np.array(swaptot[chain_swap[1]])+1
        arate = rhoswap(np.array(temp_scale[chain_swap[0]]),np.array(log_like[chain_swap[0]]),np.array(temp_scale[chain_swap[1]]),np.array(log_like[chain_swap[1]]))
        if ss.uniform.rvs() < np.array(arate):
            swapcon[chain_swap[0]] = np.array(swapcon[chain_swap[0]])+1
            swapcon[chain_swap[1]] = np.array(swapcon[chain_swap[1]])+1
            theta[chain_swap[0],:],theta[chain_swap[1],:] = np.array(theta[chain_swap[1],:]),np.array(theta[chain_swap[0],:])
            log_like[chain_swap[0]],log_like[chain_swap[1]] = np.array(log_like[chain_swap[1]]),np.array(log_like[chain_swap[0]])
            log_prior[chain_swap[0]],log_prior[chain_swap[1]] = np.array(log_prior[chain_swap[1]]),np.array(log_prior[chain_swap[0]])
            smooth_mean[chain_swap[0],:,:],smooth_mean[chain_swap[1],:,:] = np.array(smooth_mean[chain_swap[1],:,:]),np.array(smooth_mean[chain_swap[0],:,:])
            smooth_sample[chain_swap[0],:,:],smooth_sample[chain_swap[1],:,:] = np.array(smooth_sample[chain_swap[1],:,:]),np.array(smooth_sample[chain_swap[0],:,:])
            postpred_sample[chain_swap[0],:,:],postpred_sample[chain_swap[1],:,:] = np.array(postpred_sample[chain_swap[1],:,:]),np.array(postpred_sample[chain_swap[0],:,:])
            
    else: # if no swap proposed,     
        
        for chain in range(Nchains):
            
           # Block-wise MH step
           for ind in range(len(blockinds)):
               totcon[chain,ind] = np.array(totcon[chain,ind])+1
               theta_proposed[chain,blockinds[ind]] = qqq(theta[chain,blockinds[ind]],proposalcov[chain,:,blockinds[ind]][:,blockinds[ind]],ind,Omega) 
               log_like_proposed[chain],log_prior_proposed[chain],smooth_mean_proposed[chain,:,:],smooth_sample_proposed[chain,:,:],postpred_sample_proposed[chain,:,:] = SMC(AGGdata,INFdata,RSVdata,Omega,np.array(theta_proposed[chain,:]))
               if np.isnan(log_prior_proposed[chain])==0 and np.isinf(log_prior_proposed[chain])==0 and np.isnan(log_like_proposed[chain])==0 and np.isinf(log_like_proposed[chain])==0:
                   arate = rho(np.array(temp_scale[chain])*np.array(log_like[chain]),np.array(log_prior[chain]),np.array(temp_scale[chain])*np.array(log_like_proposed[chain]),np.array(log_prior_proposed[chain])) 
                   if ss.uniform.rvs() < np.array(arate):
                       acccon[chain,ind] = np.array(acccon[chain,ind])+1
                       theta[chain,:] = np.array(theta_proposed[chain,:])
                       log_like[chain]  = np.array(log_like_proposed[chain])
                       log_prior[chain] = np.array(log_prior_proposed[chain])
                       smooth_mean[chain,:,:] = np.array(smooth_mean_proposed[chain,:,:])
                       smooth_sample[chain,:,:] = np.array(smooth_sample_proposed[chain,:,:])
                       postpred_sample[chain,:,:] = np.array(postpred_sample_proposed[chain,:,:])
          
    log_post = np.multiply(temp_scale,np.transpose(log_like)) + np.transpose(np.array(log_prior))    
    Parchaincheck[:,np.array(i%Itercheck),:] = np.array(theta)
    
    if (np.sum(i == ThinIterInds) == 1) & (i>0):
        
        thinnum = np.array(thinnum) + 1
        Logpost[:,thinnum] = np.array(log_post)
        Parchainall[:,thinnum,:] = np.array(theta)
        
        INFsmean[:,thinnum,:] = smooth_mean[:,0,:]
        RSVsmean[:,thinnum,:] = smooth_mean[:,1,:]
        BACKsmean[:,thinnum,:] = smooth_mean[:,2,:]
        INFsdraw[:,thinnum,:] = smooth_sample[:,0,:]
        RSVsdraw[:,thinnum,:] = smooth_sample[:,1,:]
        BACKsdraw[:,thinnum,:] = smooth_sample[:,2,:]
        INFppred[:,thinnum,:] = postpred_sample[:,0,:]
        RSVppred[:,thinnum,:] = postpred_sample[:,1,:]
        BACKppred[:,thinnum,:] = postpred_sample[:,2,:]
    
        if (i%Itercheck==0) & (i>0):
                
                proposalcovtemp = np.zeros([Nchains*npars,npars])
                parchaintemp = np.zeros([np.array(thinnum),Nchains*npars])
                
                print("------------------------------------------------------------------")   
                print("------------------------------------------------------------------")   
                print(["We have",i,"iterations with block acceptance rates:"])      
                                
                for chain in range(Nchains): 
                    acc[chain,:] = np.array(acccon[chain])/np.array(totcon[chain]) #last Itercheck iters only
                    print(np.round(acc[chain,:],2))
                    parchaintemp[:,chain*npars:(chain+1)*npars] = np.array(Parchainall[chain,0:thinnum,])  

                    if i <= burnfactor*Thinby:  # proposal covariance adaptation during burn-in 
                        proposalcov[chain,:,:] = propcovadapt(acc[chain],np.array(proposalcov[chain,:,:]),blockinds,Parchaincheck[chain,:,:])
                        proposalcovtemp[chain*npars:(chain+1)*npars,:] = np.array(proposalcov[chain,:,:])
                        try:
                            np.savetxt('Output/pt_propcovar.csv',proposalcovtemp,delimiter=',') 
                        except:
                            print('saving error')
                        print(["Next adapted proposal variance for chain ",chain,"is:"])
                        print(np.diag(proposalcov[chain,:,:]))                                
                    else:
                        print('Proposals are no longer being adapted')
                    
                print("------------------------------------------------------------------")   
                swapacc = np.array(swapcon)/np.array(swaptot) #last Itercheck iters only
                print("The swap rate is :")
                print(np.round(np.array(swapacc),2))
                
                try:
                    np.savetxt('Output/parchain.csv', parchaintemp[-1],delimiter=',') 
                    np.savetxt('Output/pt_parchain.csv', parchaintemp[0:thinnum,-theta.shape[1]:],delimiter=',') 
                    np.savetxt('Output/pt_logposteriors.csv',Logpost[-1,0:thinnum],delimiter=',') 
                    np.savetxt('Output/pt_INFsamplesmeans.csv',INFsmean[-1,0:thinnum,:],delimiter=',')   
                    np.savetxt('Output/pt_RSVsamplesmeans.csv',RSVsmean[-1,0:thinnum,:],delimiter=',')  
                    np.savetxt('Output/pt_Backsamplesmeans.csv',BACKsmean[-1,0:thinnum,:],delimiter=',') 
                    np.savetxt('Output/pt_INFsamplesdraws.csv',INFsdraw[-1,0:thinnum,:],delimiter=',')   
                    np.savetxt('Output/pt_RSVsamplesdraws.csv',RSVsdraw[-1,0:thinnum,:],delimiter=',')  
                    np.savetxt('Output/pt_Backsamplesdraws.csv',BACKsdraw[-1,0:thinnum,:],delimiter=',') 
                    np.savetxt('Output/pt_INFsampleppreds.csv',INFppred[-1,0:thinnum,:],delimiter=',')   
                    np.savetxt('Output/pt_RSVsampleppreds.csv',RSVppred[-1,0:thinnum,:],delimiter=',')  
                    np.savetxt('Output/pt_Backsampleppreds.csv',BACKppred[-1,0:thinnum,:],delimiter=',') 
                except:
                    print('saving error')
                    
                # reset acceptance counters
                totcon = np.tile(len(blockinds)*[1.0],(Nchains,1))
                acccon = np.tile(len(blockinds)*[1.0],(Nchains,1))
                swaptot = len(temp_scale)*[1.0]
                swapcon = len(temp_scale)*[1.0]
                
                
                ## These graphics can cause some programs to crash
#                # displaying all iteration, a lag of iterations, and post-burnin sample       
#                if thinnum - burnfactor > 0:
#                    traceplots(parchaintemp[burnfactor:thinnum,-theta.shape[1]:],parnames,Logpost[-1,burnfactor:thinnum],"thinned-traceplot")
#                    pathplots(INFsmean[-1,burnfactor:thinnum,:],RSVsmean[-1,burnfactor:thinnum,:],BACKsmean[-1,burnfactor:thinnum,:],AGGdata,INFdata,RSVdata,"thinned-smooth-obs-mean-sd")
#                    pathplots(INFsdraw[-1,burnfactor:thinnum,:],RSVsdraw[-1,burnfactor:thinnum,:],BACKsdraw[-1,burnfactor:thinnum,:],AGGdata,INFdata,RSVdata,"thinned-smooth-obs-sample-sd")
#                    pathplots(INFppred[-1,burnfactor:thinnum,:],RSVppred[-1,burnfactor:thinnum,:],BACKppred[-1,burnfactor:thinnum,:],AGGdata,INFdata,RSVdata,"thinned-ppred-obs-sample-sd") 
#                    print("Note: these plots only show post-burnin samples")
#                else:
#                    traceplots(parchaintemp[0:thinnum,-theta.shape[1]:],parnames,Logpost[-1,0:thinnum],"thinned-traceplot")
#                    pathplots(INFsmean[-1,0:thinnum,:],RSVsmean[-1,0:thinnum,:],BACKsmean[-1,0:thinnum,:],AGGdata,INFdata,RSVdata,"thinned-smooth-obs-mean-sd")
#                    pathplots(INFsdraw[-1,0:thinnum,:],RSVsdraw[-1,0:thinnum,:],BACKsdraw[-1,0:thinnum,:],AGGdata,INFdata,RSVdata,"thinned-smooth-obs-sample-sd")
#                    pathplots(INFppred[-1,0:thinnum,:],RSVppred[-1,0:thinnum,:],BACKppred[-1,0:thinnum,:],AGGdata,INFdata,RSVdata,"thinned-ppred-obs-sample-sd")
#                    print("Note: these are pre-burnin samples. Burnin ends at ", burnfactor*Thinby," iterations")
#                    
#                if thinnum - lagfactor > 0:
#                    traceplots(parchaintemp[thinnum-lagfactor:thinnum,-theta.shape[1]:],parnames,Logpost[-1,thinnum-lagfactor:thinnum],"lagged-thinned-traceplot")
#                    pathplots(INFsmean[-1,thinnum-lagfactor:thinnum,:],RSVsmean[-1,thinnum-lagfactor:thinnum,:],BACKsmean[-1,thinnum-lagfactor:thinnum,:],AGGdata,INFdata,RSVdata,"lagged-thinned-smooth-obs-mean-sd")
#                    pathplots(INFsdraw[-1,thinnum-lagfactor:thinnum,:],RSVsdraw[-1,thinnum-lagfactor:thinnum,:],BACKsdraw[-1,thinnum-lagfactor:thinnum,:],AGGdata,INFdata,RSVdata,"lagged-thinned-smooth-obs-sample-sd")
#                    pathplots(INFppred[-1,thinnum-lagfactor:thinnum,:],RSVppred[-1,thinnum-lagfactor:thinnum,:],BACKppred[-1,thinnum-lagfactor:thinnum,:],AGGdata,INFdata,RSVdata,"lagged-thinned-postpred-obs-sample-sd") 
#                else:
#                    print('Lagged plots will be produced after ', lagfactor*Thinby, ' iterations')
#                
#                try:
#                    for ind in range(len(blockinds)):
#                        corrplots(parchaintemp[0:thinnum,-theta.shape[1]:][:,blockinds[ind]],[parnames[i] for i in blockinds[ind]],ind)
#                except:
#                    print('No posterior correlation plots produced')           
#      