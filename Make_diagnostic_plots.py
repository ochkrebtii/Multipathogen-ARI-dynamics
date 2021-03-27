"""
                    DIAGNOSTICS AND MCMC OUTPUT
"""
#==============================================================================

import numpy as np 
import pylab as pl
import matplotlib.ticker
import matplotlib.pyplot as plt
import corner
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot

#==============================================================================


#==============================================================================
# Plotting Functions
#==============================================================================

def pathplots(INF,RSV,Back,AGGdata,INFdata,RSVdata,name,INF_MAP,RSV_MAP,BAC_MAP):
    """ Produce plots of sample paths """    
    fig  = pl.figure(facecolor='white')
    fig.subplots_adjust(wspace=0.05)
    nticks = 6
    t_data = range(52)
    aggpathfig  = fig.add_subplot(1,2,1) 
    aggpathfig.title.set_fontsize(20)
    aggpathfig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    aggpathfig.set_xlim([0.0,52.0])
    aggpathfig.set_ylim([0.0,25000.0])
    
    if INF.shape==(52,):
        aggpathfig.plot(t_data,INF+RSV+Back,color='r', label='SMC',lw=2.0)   
    else:
        q1 = np.percentile(INF+RSV+Back,2.5,axis=0)
        q2 = np.percentile(INF+RSV+Back,97.5,axis=0)
        aggpathfig.fill_between(t_data,q1,q2,color='lightgray',alpha=0.9)
        nplots = min(INF.shape[0],25)
#        for i in range(0,INF.shape[0],int(INF.shape[0]/nplots)):   
#            aggpathfig.plot(t_data,INF[i,:]+RSV[i,:]+Back[i,:],color='k', label='SMC',lw=2,alpha=0.1+1/nplots)   
#        aggpathfig.plot(t_data,INF_MAP+RSV_MAP+BAC_MAP,'--',color='k', label='SMC',lw=1.5)   
        
    aggpathfig.plot(t_data,AGGdata, '.--',color='k',label='Data',lw=2,alpha=1)   
    aggpathfig.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks    
    
    disaggfig = fig.add_subplot(1,2,2)
    par1  = disaggfig.twinx() 
    par1.fill_between(t_data,INFdata,0,facecolor='r', alpha=0.6, label= 'INF')
    par1.fill_between(t_data,RSVdata,0,facecolor='b', alpha=0.4, label= 'RVS')
    par1.fill_between(t_data,BACdata,0,facecolor='g', alpha=0.2, label= 'BAC')
    par1.set_ylim([0.0,50.0])
    
    disaggfig.title.set_fontsize(20)
    disaggfig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    disaggfig.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    disaggfig.set_xlim([0.0,52.0])
    disaggfig.set_ylim([0.0,25000.0])
    #disaggfig.set_yticks([])
    
    if INF.shape==(52,):
        disaggfig.plot(t_data,INF ,'-.',color='r', label='INF',lw=2,alpha=1.0)
        disaggfig.plot(t_data,RSV ,'--',color='b',label='RSV',lw=2, alpha=1.0)
        disaggfig.plot(t_data,Back ,'--',color='g',label='BAC',lw=2, alpha=1.0) 
    else:
#        for i in range(0,INF.shape[0],int(INF.shape[0]/nplots)):   
#            disaggfig.plot(t_data,INF[i,:],color='r', label='INF',lw=2,alpha=0.1+1/nplots)   
#            disaggfig.plot(t_data,RSV[i,:],color='b', label='RSV',lw=2,alpha=0.1+1/nplots)  
#            disaggfig.plot(t_data,Back[i,:],color='g', label='BAC',lw=2,alpha=0.1+1/nplots)  
        disaggfig.plot(t_data,INF_MAP,'--',color='r', label='INF',lw=1.5)   
        disaggfig.plot(t_data,RSV_MAP,'--',color='b', label='RSV',lw=1.5)   
        disaggfig.plot(t_data,BAC_MAP,'--',color='g', label='BAC',lw=1.5)   
            
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
    fig.text(0.075,0.5,'Aggregated ARI Reports ',va='center',rotation='vertical',fontsize=20)
    fig.text(0.5,0.125,'Time (Weeks)',ha='center',fontsize=20)
    pl.autoscale(enable=True, axis='x', tight=True)
    fig.text(0.93,0.5,"Laboratory Samples",va='center',rotation='vertical',fontsize=20)
    disaggfig.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks
    fig.set_size_inches(18.5, 10.5)
    fig.savefig("Figures/" + str(name) + ".png")#,dpi=100)
    pl.close(fig)
        
    
def traceplots(Parameters,parnames,logpost,name):
    """ Produce sample trace plots """
    fig = pl.figure(facecolor='white', figsize=(20,10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
    a = 4
    b = 5
    
    for ax in range(len(parnames)):
        axfig  = fig.add_subplot(a,b,ax+1) 
        axfig.plot(Parameters[:,ax],lw=1.5)
        axfig.set_xticklabels([])
        axfig.set_xlabel(parnames[ax])
        #axfig.set_yticklabels([])
        plt.locator_params(axis='y', nbins=4)
        
    axfig  = fig.add_subplot(a,b,len(parnames)+1)
    axfig.set_xticklabels([])
    axfig.set_xlabel('Un-normalized log posterior')
    #axfig.set_yticklabels([])
    axfig.plot(logpost,lw=1.5)
    plt.locator_params(axis='y', nbins=4)
    
    plt.rc('font', size=18)
    plt.rc('ytick', labelsize=12) 
    fig.set_size_inches(30, 15)
    fig.savefig("Figures/" + str(name) + ".png")#,dpi=100)
    pl.close(fig)
    

def corrplots(p,pnames,ind):
    """ Produce posterior sample correlation plots """
    fig = corner.corner(p, labels=pnames,label_kwargs={'fontsize':25},
                  truths=None,max_n_ticks=1, quantiles=[0.05, 0.5, 0.95],
                  color='k',truth_color='b', smooth=1.0) 
       
    axes = np.array(fig.axes).reshape((np.shape(p)[1], np.shape(p)[1]))
    
    # Loop over the histograms
    for yi in range(np.shape(p)[1]):
        for xi in range(yi+1):
            ax = axes[yi, xi]
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
    pl.tight_layout()
    fig.set_size_inches(15, 15)
    fig.savefig("Figures/corplots-block" + str(ind) + '.png')
    plt.close(fig)



def residplots(INF,RSV,Back,AGGdata,INFdata,RSVdata,name):
    """ Produce plots of sample paths """    
    fig  = pl.figure(facecolor='white')
    fig.subplots_adjust(wspace=0.1)
    t_data = range(52)
    
    
    residplot  = fig.add_subplot(1,3,1)
    #residplot.title.set_fontsize(20)
    residplot.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    if INF.shape==(52,):
        residplot.plot(t_data,INF+RSV+Back-AGGdata,color='r', label='SMC',lw=2.0)   
    else:
        nplots = min(INF.shape[0],25)
        
        for i in range(0,np.shape(t_data)[0]):   
            for j in range(0,INF.shape[0],int(INF.shape[0]/nplots)):   
                residplot.plot(t_data[i],INF[j,i]+RSV[j,i]+Back[j,i]-AGGdata[i],'.',color='k', label='SMC',lw=2,alpha=0.1+1/nplots)   
        
    residplot.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks    
    residplot.set_ylabel('Residuals')    
    
    qqplotfig = fig.add_subplot(1,3,2)      
    #qqplotfig.title.set_fontsize(20)
    qqplotfig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    if INF.shape==(52,):
        qqplotfig.plot(t_data,INF+RSV+Back-AGGdata,color='r', label='SMC',lw=2.0)   
    else:
        nplots = min(INF.shape[0],25)
        residvec_avg = np.zeros(np.shape(t_data)[0]);
        
        for i in range(0,np.shape(t_data)[0]):   
            residvec_avg[i] = sum(INF[:,i]+RSV[:,i]+Back[:,i])/INF.shape[0]-AGGdata[i];
    
        qqplot(residvec_avg,fit='True',line ='45',ax=qqplotfig) #,line='45')
        
    
    acpfig = fig.add_subplot(1,3,3)      
    #acpfig.title.set_fontsize(20)
    acpfig.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    if INF.shape==(52,):
        acpfig.plot(t_data,INF+RSV+Back-AGGdata,color='r', label='SMC',lw=2.0)   
    else:
        nplots = min(INF.shape[0],25)
        residvec_avg = np.zeros(np.shape(t_data)[0]);
        
        for i in range(0,np.shape(t_data)[0]):   
            residvec_avg[i] = sum(INF[:,i]+RSV[:,i]+Back[:,i])/INF.shape[0]-AGGdata[i];
    
        sm.graphics.tsa.plot_acf(residvec_avg, lags=40, zero=False, alpha=None, ax=acpfig)
        acpfig.set_ylim([-1.0,1.0])
    
    fig = pl.gcf()
    fig.autofmt_xdate()
    pl.rcParams['axes.grid'] = True
    pl.rcParams['grid.linestyle'] = ':'
    #fig.text(0.075,0.5,'Aggregated ARI Reports ',va='center',rotation='vertical',fontsize=20)
    #fig.text(0.5,0.125,'Time (Weeks)',ha='center',fontsize=20)
    pl.autoscale(enable=True, axis='x', tight=True)
    #acpfig.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks
    fig.set_size_inches(32, 10.5)
    fig.savefig("Figures/" + str(name) + ".png")#,dpi=100)
    pl.close(fig)


######### Load Data ##############

Omega  = 2.5*10**6  # population size
window = 52         # number of weeks in one year

# epidemic begins at week 32
# first year we have full data is 2003-2004 (starting week 32 of 2003, ending week 31 of 2004)
# last year we have full data is 2008-2009 

year   = 1          # year 1 is 2003-2004 and year 6 is 2008-2009  
                    # for simulation, the year should always be 1
                    
#data = np.array(np.genfromtxt("Data/iras_data_full_load.csv", delimiter=",")) # for real data
data = np.array(np.genfromtxt("Data/iras_data_full_load_sim.csv", delimiter=",")) # for simulated data (all years)

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

parnames = ['${\\beta}_{10}$','${\\beta}_{20}$','${\\sigma}_{1}$','${\\sigma}_{2}$',
            '$X_{ss}(0)$','$X_{is}(0)$','$X_{rs}(0)$','$X_{si}(0)$',
            '$X_{ri}(0)$','$X_{sr}(0)$','$X_{ir}(0)$','$X_{rr}(0)$',
            '$C_0$ (fixed)','${\\Sigma}$','$r$','$c$',
            '${\\nu}$','$V_{inf}$']   
 
blockinds = [] # define sampling block indices for the parameters as a list 
blockinds.append([0,1,2,3]) # SDE model parameters
blockinds.append([4,5,6,7,8,9,10,11])  # SDE initial states np.array(range(4,12))
blockinds.append([13,14,15,16,17]) # Probability model parameters


######### Load MCMC Output ##############

Parameters     = np.genfromtxt('Output/pt_parchain.csv',   delimiter=',')    
Logpost        = np.genfromtxt('Output/pt_logposteriors.csv', delimiter=',')   

INF_spaths     = np.genfromtxt('Output/pt_INFsamplesdraws.csv',delimiter=',')   
RSV_spaths     = np.genfromtxt('Output/pt_RSVsamplesdraws.csv',delimiter=',')   
BAC_spaths     = np.genfromtxt('Output/pt_Backsamplesdraws.csv',delimiter=',')             

INF_ppredspaths     = np.genfromtxt('Output/pt_INFsampleppreds.csv',delimiter=',')   
RSV_ppredspaths     = np.genfromtxt('Output/pt_RSVsampleppreds.csv',delimiter=',')   
BAC_ppredspaths     = np.genfromtxt('Output/pt_Backsampleppreds.csv',delimiter=',')             


############# Compute relevant summaries ###############

burnin = 0 #np.int(np.floor(len(Logpost)/2))

# Calculate MAP estimates

maxindex = np.min([np.shape(INF_spaths)[0],np.shape(RSV_spaths)[0],np.shape(BAC_spaths)[0]])
inds = np.array(range(0,maxindex))#np.where(Parameters[0:maxindex,12]>1e-4)[0]
Logpost = np.array(Logpost[inds])
Parameters = np.array(Parameters[inds,:])

#Logpost = Logpost[0:maxindex]
INF_spaths = INF_spaths[0:maxindex,:]
RSV_spaths = RSV_spaths[0:maxindex,:]
BAC_spaths = BAC_spaths[0:maxindex,:]

MAP_ind = np.where(Logpost==np.max(Logpost))[0][0]
INF_MAP = np.array(INF_spaths[MAP_ind,:])
RSV_MAP = np.array(RSV_spaths[MAP_ind,:])
BAC_MAP = np.array(BAC_spaths[MAP_ind,:])
AGG_MAP = INF_MAP + RSV_MAP + BAC_MAP
pars_MAP = np.array(Parameters[MAP_ind,:])
pars_FIRST = np.array(Parameters[0,:])
pars_LAST = np.array(Parameters[-1,:])

np.savetxt("Data/MAP_INF.csv", INF_MAP, delimiter=",")
np.savetxt("Data/MAP_RSV.csv", RSV_MAP, delimiter=",")
np.savetxt("Data/MAP_BAC.csv", BAC_MAP, delimiter=",")
np.savetxt("Data/MAP_AGG.csv", AGG_MAP, delimiter=",")
np.savetxt("Data/MAP_pars.csv", pars_MAP, delimiter=",")


CR_AGG    = np.zeros([2,52])
CR_INF    = np.zeros([2,52])
CR_RSV    = np.zeros([2,52])
CR_BAC    = np.zeros([2,52])
AGGout = (INF_spaths + RSV_spaths + BAC_spaths)
INFout = INF_spaths
RSVout = RSV_spaths
BACout = BAC_spaths

for i in range(len(AGGout[0,:])):
    
    CR_AGG[0,i] = np.percentile(AGGout[burnin:-1,i],2.5,axis=0)
    CR_AGG[1,i] = np.percentile(AGGout[burnin:-1,i],97.5,axis=0)
 
    CR_INF[0,i] = np.percentile(INFout[burnin:-1,i],2.5,axis=0)
    CR_INF[1,i] = np.percentile(INFout[burnin:-1,i],97.5,axis=0)     
 
    CR_RSV[0,i] = np.percentile(RSVout[burnin:-1,i],2.5,axis=0)
    CR_RSV[1,i] = np.percentile(RSVout[burnin:-1,i],97.5,axis=0)

    CR_BAC[0,i] = np.percentile(BACout[burnin:-1,i],2.5,axis=0)
    CR_BAC[1,i] = np.percentile(BACout[burnin:-1,i],97.5,axis=0)

np.savetxt("Data/CR_AGG.csv", CR_AGG, delimiter=",")
np.savetxt("Data/CR_INF.csv", CR_INF, delimiter=",")
np.savetxt("Data/CR_RSV.csv", CR_RSV, delimiter=",")
np.savetxt("Data/CR_BAC.csv", CR_BAC, delimiter=",")

CI_pars = np.zeros([2,len(parnames)])
CI_pars[0,:] = np.percentile(Parameters[burnin:-1,:],2.5,axis=0)
CI_pars[1,:] = np.percentile(Parameters[burnin:-1,:],97.5,axis=0)
MED_pars = np.percentile(Parameters[burnin:-1,:],50,axis=0)

np.savetxt("Data/CI_pars.csv", CI_pars, delimiter=",")
np.savetxt("Data/MED_pars.csv", MED_pars, delimiter=",")



########### MAKE FIGURES ##################


residplots(INF_ppredspaths,RSV_ppredspaths,BAC_ppredspaths,AGGdata,INFdata,RSVdata,"residual_plot_full_sample")
residplots(INF_ppredspaths[burnin:-1,:],RSV_ppredspaths[burnin:-1,:],BAC_ppredspaths[burnin:-1,:],AGGdata,INFdata,RSVdata,"residual_plot_half_sample")

traceplots(Parameters[burnin:-1,:],parnames,Logpost[burnin:-1],"thinned-traceplot-half_sample")
traceplots(Parameters,parnames,Logpost,"thinned-traceplot-full_sample")

pathplots(INF_spaths[burnin:-1,:],RSV_spaths[burnin:-1,:],BAC_spaths[burnin:-1,:],AGGdata,INFdata,RSVdata,"thinned-posterior-samplepaths_half_sample",INF_MAP,RSV_MAP,BAC_MAP)
pathplots(INF_spaths,RSV_spaths,BAC_spaths,AGGdata,INFdata,RSVdata,"thinned-posterior-samplepaths_full_sample",INF_MAP,RSV_MAP,BAC_MAP)

try:                
    for ind in range(len(blockinds)):
        corrplots(Parameters[burnin:-1,blockinds[ind]],[parnames[i] for i in blockinds[ind]],ind)
except:
    print('no corrplots produced')
    
print('-----------------------------------')
print('Plots have been produced and placed in the folder "Figures"')
print('-----------------------------------')




fig  = pl.figure(facecolor='white')
#pl.title(yearname[year],fontsize=20)
fig.subplots_adjust(wspace=0.05)
nticks = 6
t_data = range(52)

ax1  = fig.add_subplot(1,2,1) 

##-------------------------------------------------------------------------
MAPYear1 = (INF_MAP + RSV_MAP + BAC_MAP)
ax1.plot(t_data,MAPYear1,color='g', label='SMC',lw=2.0)   
ax1.plot(t_data,AGGdata, '.--',color='k',label='Data',lw=1.2,alpha=0.85)
ax1.plot(CR_AGG[0,:],color='k',alpha=0.8,lw=1.0)
ax1.plot(CR_AGG[1,:],color='k',alpha=0.8,lw=1.0)
ax1.fill_between(t_data,CR_AGG[0,:],CR_AGG[1,:],color='lightgray',alpha=0.5)
ax1.title.set_fontsize(20)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_xlim([0.0,52.0])
ax1.set_ylim([5000.0, 25000.0])
   
##-------------------------------------------------------------------------
host1 = fig.add_subplot(1,2,2)
par1  = host1.twinx()

host1.plot(t_data,INF_MAP ,'-.',color='r', label='SMC Influenza ',lw=2,alpha=1.0)
host1.plot(t_data,RSV_MAP ,'--',color='b',label='SMC RSV',lw=2, alpha=1.0)
host1.plot(t_data,BAC_MAP ,'--',color='g',label='BAC RSV',lw=2, alpha=1.0)

host1.title.set_fontsize(20)
host1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
host1.plot(CR_INF[0,:],color='k',alpha=0.8,lw=1.0)
host1.plot(CR_INF[1,:],color='k',alpha=0.8,lw=1.0)
host1.fill_between(t_data,CR_INF[0,:],CR_INF[1,:],color='lightgray',alpha=0.5)
host1.plot(CR_RSV[0,:],color='k',alpha=0.8,lw=1.0)
host1.plot(CR_RSV[1,:],color='k',alpha=0.8,lw=1.0)
host1.fill_between(t_data,CR_RSV[0,:],CR_RSV[1,:],color='lightgray',alpha=0.5)
host1.plot(CR_BAC[0,:],color='k',alpha=0.8,lw=1.0)
host1.plot(CR_BAC[1,:],color='k',alpha=0.8,lw=1.0)
host1.fill_between(t_data,CR_BAC[0,:],CR_BAC[1,:],color='lightgray',alpha=0.5)
host1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
host1.set_xlim([0.0,52.0])
host1.set_ylim([0.0,14000.0])
host1.set_yticks([])

par1.plot(t_data,INFdata,color='r', label='INF',lw=1,alpha=0.5)
par1.fill_between(t_data,INFdata,0,facecolor='r', alpha=0.5)
par1.plot(t_data,RSVdata, color='b',label='RSV',lw=1, alpha=0.3)
par1.fill_between(t_data,RSVdata,0,facecolor='b', alpha=0.3, label= 'RVS')
par1.plot(t_data,BACdata, color='g',label='BAC',lw=1, alpha=0.25)
par1.fill_between(t_data,BACdata,0,facecolor='g', alpha=0.25, label= 'BAC')
par1.set_ylim([0.0,50.0])

fig = pl.gcf()
fig.autofmt_xdate()
pl.rcParams['axes.grid'] = True
pl.rcParams['grid.linestyle'] = ':'
fig.text(0.020,0.5,'Aggregated ARI Reports ',va='center',rotation='vertical',fontsize=20)
fig.text(0.5,0.1,'Time (Weeks)',ha='center',fontsize=20)
pl.autoscale(enable=True, axis='x', tight=True)
fig.text(0.93,0.5,"Laboratory Samples",va='center',rotation='vertical',fontsize=20)
ax1.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks   
host1.tick_params(axis='y', which='major', labelsize=15) #change the size of axis ticks
fig.set_size_inches(15,7.5)


#print(pars_FIRST - pars_LAST)
#print(pars_LAST)
#print(len(Logpost))
#
#print(pars_MAP)
#print(CI_pars)
#print(len(Logpost))