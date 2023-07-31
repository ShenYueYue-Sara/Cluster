import math
import emcee
import corner
import numpy as np
import pandas as pd
from mock_cmd import MockCMD
import matplotlib.pyplot as plt
plt.style.use("default")

isochrones_dir = '/home/shenyueyue/Projects/Cluster/data/isocForMockCMD'

def lnlike(theta, n_stars, sample_obs): # logage, mh, fb, dm = theta
    logage, mh, fb, dm = theta
    if (logage>10.0) or (logage<6.6) or (mh<-2.0) or (mh>0.7) or (fb<0.05) or (fb>1) or (dm<2) or (dm>20):
        return -1e50
    m = MockCMD(sample_obs=sample_obs,isochrones_dir=isochrones_dir)
    sample_syn = m.mock_stars(theta,n_stars)
    c_syn, m_syn = MockCMD.extract_CMD(sample_syn, band_a='Gmag_syn', band_b='G_BPmag_syn', band_c='G_RPmag_syn')
    c_obs, m_obs = MockCMD.extract_CMD(sample_obs, band_a='Gmag', band_b='G_BPmag', band_c='G_RPmag')
    lnlikelihood = m.eval_lnlikelihood(c_obs, m_obs, c_syn, m_syn)
    if math.isnan(lnlikelihood):
        return -1e50
    return lnlikelihood

def main():
    name = 'Melotte_22'
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%(name))
    
    # theta = (7.89, 0.032, 0.35, 5.55)
    # logage, mh, fb, dm = theta
    n_stars = 1000
    
    # set up the MCMC sampler
    nwalkers = 10
    ndim = 4
    
    # define the step sizes for each parameter
    step_sizes = np.array([0.05, 0.2, 0.1, 2])
    moves = [emcee.moves.StretchMove(a=step_sizes[i]) for i in range(ndim)]
    # create an array of initial positions with small random perturbations
    p0 = np.round((np.array([7.80, 0.02, 0.35, 5.55]) + step_sizes * np.random.randn(nwalkers, ndim)), decimals=2)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(n_stars, sample_obs), moves=moves)
    
    # burn-in
    nburn = 25
    pos,_,_ = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()
    # run the MCMC sampler
    nsteps = 100
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    fig = corner.corner(samples,
        labels=[r'log(age)',r'[M/H]',r'$f_b$',r'DM'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True, 
            title_kwargs={"fontsize": 12},
            title_fmt = '.4f')
    fig.savefig('/home/shenyueyue/Projects/Cluster/code/point_source/figure/emcee_result.png',bbox_inches='tight')

if __name__ == '__main__':
    main()