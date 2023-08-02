import os
import math
import time
import emcee
import corner
import logging
import traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool
from joblib import Parallel,delayed

from mock_cmd import MockCMD
import matplotlib.pyplot as plt
plt.style.use("default")

# logfile in total.log

isochrones_dir = '/home/shenyueyue/Projects/Cluster/data/isocForMockCMD/'

def download_isochrone(isochrones_dir, logage_grid, mh_grid, dm, n_jobs=10):
    astart,aend,astep = logage_grid
    mstart,mend,mstep = mh_grid
    abin = np.arange(astart,aend,astep)
    mbin = np.arange(mstart,mend,mstep)
    logage_mh = []
    for a in abin:
        for m in mbin:
            logage_mh.append([a,m])
    logging.info(f"dowanload in total : {len(logage_mh)} isochrones")
    
    # nested function, access variable in parent function
    def get_isochrone_wrapper(logage, mh):
        m = MockCMD(isochrones_dir=isochrones_dir)
        m.get_isochrone(logage=logage, mh=mh, dm=dm, logage_step=astep, mh_step=mstep)
        logging.info(f"get_iso : logage = {logage}, mh = {mh}")
        
    # parallel excution
    Parallel(n_jobs=n_jobs)(
        delayed(get_isochrone_wrapper)(logage, mh) for logage, mh in logage_mh
    )
    # with tqdm(total=len(logage_mh), desc='Downloading isochrones') as pbar:
    #     Parallel(n_jobs=n_jobs)(
    #         delayed(get_isochrone_wrapper)(logage, mh) for logage, mh in logage_mh
    #     )
    #     pbar.update(1) 

def lnlike(theta_part, n_stars, step, sample_obs): # logage, mh, fb, dm = theta
    start_time = time.time()
    #logage, mh, fb, dm = theta
    fb, dm = 0.35, 5.55
    logage, mh = theta_part
    theta = logage, mh, fb, dm
    try:
        if (logage>10.0) or (logage<6.6) or (mh<-0.9) or (mh>0.7) or (fb<0.05) or (fb>1) or (dm<2) or (dm>20):
            return -1e50
        m = MockCMD(sample_obs=sample_obs,isochrones_dir=isochrones_dir)
        sample_syn = m.mock_stars(theta,n_stars,step)
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
    
def test_randomness(sample_obs, theta, n_stars, time=1000, step=(0.05, 0.1)): # test the randomness of lnlikelihood()
    lnlike_list = []
    logage, mh, fb, dm = theta
    for i in range(time):
        lnlikelihood = lnlike(theta, n_stars, step, sample_obs)
        lnlike_list.append(lnlikelihood)
        if i%100 == 0 : print(i)
    fig,ax = plt.subplots(figsize=(5,5))
    ax.hist(lnlike_list, bins=50)
    ax.set_xlabel('lnlikelihood')
    # Add the text to the top right corner
    ax.text(0.7, 0.95, f"mean: {np.mean(lnlike_list):.1f}", transform=ax.transAxes)
    ax.text(0.7, 0.90, f"std: {np.std(lnlike_list):.1f}", transform=ax.transAxes)
    ax.set_title(f"logage:{logage} [M/H]:{mh} fb:{fb} dm:{dm} nstars:{n_stars}")
    fig.show()

def main():

    # download isochrone
    logage_grid = (6.6, 10, 0.01)
    mh_grid = (-0.9, 0.7, 0.01)
    dm = 5.55 
    download_isochrone(isochrones_dir=isochrones_dir, logage_grid=logage_grid, mh_grid=mh_grid, dm=dm)

    '''
    # read smaple_obs
    name = 'Melotte_22'
    usecols = ['Gmag','G_BPmag','G_RPmag','phot_g_n_obs','phot_bp_n_obs','phot_rp_n_obs']
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%(name), usecols=usecols)
    sample_obs = sample_obs.dropna().reset_index(drop=True)

    # MCMC
    n_stars = 942
    step = (0.05, 0.1)
    # parameter
    theta_part = np.array([7.89, 0.032]) # after round theta = (8.00, 0.0) 
    scale = np.array([0.05, 0.1])
    ndim = 2
    
    # set up the MCMC sampler
    nwalkers = 50
    # define the step sizes for each parameter
    # moves = [emcee.moves.StretchMove(a=step_sizes[i]) for i in range(ndim)]
    # create an array of initial positions with small random perturbations
    p0 = np.round((theta_part + scale * np.random.randn(nwalkers, ndim)), decimals=2)
    
    # parallelization
    with Pool() as pool:   
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(n_stars, step, sample_obs), pool=pool) # , moves=moves
        # burn-in
        nburn = 1000
        start_burn = time.time()
        pos,_,_ = sampler.run_mcmc(p0, nburn, progress=True)
        end_burn = time.time()
        time_burn = end_burn - start_burn
        print(f"burn-in time : {time_burn:.1f} seconds")
        sampler.reset()
        
        # run the MCMC sampler
        nsteps = 2000
        start_run = time.time()
        sampler.run_mcmc(pos, nsteps, progress=True)
        end_run = time.time()
        time_run = end_run - start_run
        print(f"run time : {time_run:.1f} seconds")
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    fig = corner.corner(samples,
        labels=[r'log(age)',r'[M/H]',r'$f_b$',r'DM'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, 
            title_kwargs={"fontsize": 12},
            title_fmt = '.4f')
    fig.savefig(f"/home/shenyueyue/Projects/Cluster/code/point_source/figure/mcmc_w{nwalkers}_b{nburn}_r{nsteps}.png",bbox_inches='tight')
    '''
    '''
    # test the randomness of lnlikelihood() vs logage
    theta1 = (7.89, 0.032, 0.35, 5.55)
    theta2 = (8.89, 0.032, 0.35, 5.55)
    n_stars = 942
    test_randomness(sample_obs, theta1, n_stars)
    test_randomness(sample_obs, theta2, n_stars)
    # test the randomness of lnlikelihood() vs n_stars
    n = [500, 942, 1000, 5000]
    for n_stars in n:
        print(f"test_randomness for n_stars={n_stars}")
        test_randomness(sample_obs, theta1, n_stars)
    '''
if __name__ == '__main__':
    main()
