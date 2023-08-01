import os
import math
import time
import emcee
import corner
import logging
import traceback
import numpy as np
import pandas as pd
from mock_cmd import MockCMD
import matplotlib.pyplot as plt
plt.style.use("default")

# logfile in total.log

isochrones_dir = '/home/shenyueyue/Projects/Cluster/data/isocForMockCMD'

def lnlike(theta, n_stars, sample_obs): # logage, mh, fb, dm = theta
    start_time = time.time()
    logage, mh, fb, dm = theta
    try:
        if (logage>10.0) or (logage<6.6) or (mh<-2.0) or (mh>0.7) or (fb<0.05) or (fb>1) or (dm<2) or (dm>20):
            return -1e50
        m = MockCMD(sample_obs=sample_obs,isochrones_dir=isochrones_dir)
        sample_syn = m.mock_stars(theta,n_stars)
        c_syn, m_syn = MockCMD.extract_CMD(sample_syn, band_a='Gmag_syn', band_b='G_BPmag_syn', band_c='G_RPmag_syn')
        c_obs, m_obs = MockCMD.extract_CMD(sample_obs, band_a='Gmag', band_b='G_BPmag', band_c='G_RPmag')
        lnlikelihood = m.eval_lnlikelihood(c_obs, m_obs, c_syn, m_syn)
        # if math.isnan(lnlikelihood):
        #     return -1e50
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time lnlike() : {run_time:.4f} s")    
        return lnlikelihood
    
    except Exception as e:
        print("lnlike:",lnlikelihood)
        print("Error parameters: [%f, %f, %f, %f]"%(logage,mh,fb,dm))
        print(f"Error encountered: {e}")
        traceback.print_exc()
        return -1e50

def main():

    
    '''
    # MCMC
    name = 'Melotte_22'
    usecols = ['Gmag','G_BPmag','G_RPmag','phot_g_n_obs','phot_bp_n_obs','phot_rp_n_obs']
    print('start reading file')
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%(name), usecols=usecols)
    sample_obs = sample_obs.dropna().reset_index(drop=True)
    print('end reading file')
    
    logage, mh, fb, dm = theta
    theta = (7.89, 0.032, 0.35, 5.55)
    n_starts = 942
    
    print('start emcee')
    # set up the MCMC sampler
    nwalkers = 10
    ndim = 4
    # define the step sizes for each parameter
    step_sizes = np.array([0.05, 0.2, 0.1, 2])
    moves = [emcee.moves.StretchMove(a=step_sizes[i]) for i in range(ndim)]
    # create an array of initial positions with small random perturbations
    p0 = np.round((np.array([7.80, 0.02, 0.35, 5.55]) + step_sizes * np.random.randn(nwalkers, ndim)), decimals=2)
    
    print('start init emcee')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(n_stars, sample_obs), moves=moves)
    print('start burnin')
    # burn-in
    nburn = 50
    pos,_,_ = sampler.run_mcmc(p0, nburn) #, progress=True
    sampler.reset()
    # run the MCMC sampler
    nsteps = 200
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    fig = corner.corner(samples,
        labels=[r'log(age)',r'[M/H]',r'$f_b$',r'DM'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, 
            title_kwargs={"fontsize": 12},
            title_fmt = '.4f')
    fig.savefig('/home/shenyueyue/Projects/Cluster/code/point_source/figure/emcee_result.png',bbox_inches='tight')
    '''
    
    # test the randomness of lnlikelihood()
    lnlike_list = []
    theta = (8.89, 0.032, 0.35, 5.55)
    logage, mh, fb, dm = theta
    n_stars = 942
    for i in range(1000):
        lnlikelihood = lnlike(theta, n_stars, sample_obs)
        lnlike_list.append(lnlikelihood)
        if i%100 == 0 : print(i)
    fig,ax = plt.subplots(figsize=(5,5))
    ax.hist(lnlike_list, bins=50)
    ax.set_xlabel('lnlikelihood')
    ax.set_title(f"logage:{logage} [M/H]:{mh} fb:{fb} dm:{dm}")
    fig.show()
    
if __name__ == '__main__':
    main()