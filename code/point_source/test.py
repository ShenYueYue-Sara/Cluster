import numpy as np
import pandas as pd

from mock_cmd import MockCMD
from gaia_magerror import MagError
import matplotlib.pyplot as plt
plt.style.use("default")
    
def test_MagErr_using_obs():
    """Test Magerr() class for DR2 & EDR3 data"""
    name = 'Melotte_22'
    usecols = ['Gmag','G_BPmag','G_RPmag',
               'phot_g_n_obs_dr2','phot_bp_n_obs_dr2','phot_rp_n_obs_dr2',
               'Gmag_err','G_BPmag_err','G_RPmag_err',
               'phot_g_mean_mag','phot_g_mean_mag_corrected','phot_bp_mean_mag','phot_rp_mean_mag',
               'phot_g_n_obs_edr3','phot_bp_n_obs_edr3','phot_rp_n_obs_edr3',
               'phot_g_mean_mag_error','phot_g_mean_mag_error_corrected','phot_bp_mean_mag_error','phot_rp_mean_mag_error',
               'phot_g_mean_mag_errcal','phot_bp_mean_mag_errcal','phot_rp_mean_mag_errcal',]
    sample = pd.read_csv('/home/shenyueyue/Projects/Cluster/data/melotte_22_edr3.csv',usecols=usecols)
    
    e2 = MagError(
        sample_obs=sample,
        bands=['phot_g_mean_mag_corrected','phot_bp_mean_mag','phot_rp_mean_mag'],
        nobs = ['phot_g_n_obs_edr3','phot_bp_n_obs_edr3','phot_rp_n_obs_edr3'])
    g_med_err, bp_med_err, rp_med_err = e2.estimate_med_photoerr(sample_syn=sample)
    g_syn_edr3, bp_syn_edr3, rp_syn_edr3 = e2.syn_sample_photoerr(sample_syn = sample)
    # draw hist for obs_err and syn_err
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
    for i,(err,errobs) in enumerate(
        zip((g_med_err, bp_med_err, rp_med_err), 
            (sample['phot_g_mean_mag_error'], sample['phot_bp_mean_mag_error'], sample['phot_rp_mean_mag_error']))):
        ax[i].hist(np.log10(errobs), bins=30, label='obs')
        ax[i].hist(np.log10(err), bins=30, label='bspline + sigma0')
        ax[i].legend()
          
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
    ax[0].scatter(sample['phot_g_mean_mag'],np.log10(sample['phot_g_mean_mag_error']), s=2, c='blue', label='obs')
    ax[0].scatter(sample['phot_g_mean_mag'],np.log10(g_syn_edr3-g_med_err), s=2, c='red', label='syn')
    ax[0].legend()

    
    # sample['Gmag_err_synedr3'], sample['G_BPmag_err_synedr3'], sample['G_RPmag_err_synedr3'] = (
    #     g_med_err_edr3, bp_med_err_edr3, rp_med_err_edr3 )
    # sample['Gmag_synedr3'], sample['G_BPmag_synedr3'], sample['G_RPmag_synedr3'] = (
    #     g_syn_edr3, bp_syn_edr3, rp_syn_edr3 )
    
    e1 = MagError(
        sample_obs=sample,
        nobs = ['phot_g_n_obs_dr2','phot_bp_n_obs_dr2','phot_rp_n_obs_dr2'])
    g_med_err_dr2, bp_med_err_dr2, rp_med_err_dr2 = e1.estimate_med_photoerr(sample_syn=sample)
    g_syn_dr2, bp_syn_dr2, rp_syn_dr2 = e1.syn_sample_photoerr(sample_syn = sample)
    sample['Gmag_err_syndr2'], sample['G_BPmag_err_syndr2'], sample['G_RPmag_err_syndr2'] = (
        g_med_err_dr2, bp_med_err_dr2, rp_med_err_dr2 )
    sample['Gmag_syndr2'], sample['G_BPmag_syndr2'], sample['G_RPmag_syndr2'] = (
        g_syn_dr2, bp_syn_dr2, rp_syn_dr2 )
    

    # draw mag-magerr for result checking
    # obsmag -- MagError --> synmag(witherr)
    fig,ax = plt.subplots(nrows=1,ncols=4,figsize=(22,4))
    bands_dr2 = ['Gmag','G_BPmag','G_RPmag']
    bands_edr3 = ['phot_g_mean_mag_corrected','phot_bp_mean_mag','phot_rp_mean_mag']
    zero = [0.0027553202, 0.0027901700, 0.0037793818]
    for i,(band,band_edr3) in enumerate(zip(bands_dr2,bands_edr3)):
        # ax[i].scatter(sample[band], np.log10(sample['%s_err'%band]), s=2, c='green', label='dr2 data')
        if i == 0:
            ax[i].scatter(sample[band_edr3], np.log10(sample['phot_g_mean_mag_error_corrected']), s=2, c='blue', label='edr3 data')
            # ax[i].scatter(
            #     sample[band_edr3], np.log10(sample['phot_g_mean_mag_errcal']), 
            #     s=2, c='skyblue', label='edr3 cal data')
            ax[i].set_ylim(-2.6,-2.4)
        else:
            ax[i].scatter(sample[band_edr3], np.log10(sample['%s_error'%band_edr3]), s=2, c='blue', label='edr3 data')
            # ax[i].scatter(
            #     sample[band_edr3], np.log10(sample['%s_errcal'%band_edr3]), 
            #     s=2, c='skyblue', label='edr3 cal data')
            ax[i].set_ylim(-2.75,-1)
        # ax[i].scatter(sample['%s_syndr2'%band], np.log10(sample['%s_err_syndr2'%band]), s=2, c='orange', label='syndr2 data')
        ax[i].scatter(sample['%s_synedr3'%band], np.log10(np.sqrt(sample['%s_err_synedr3'%band]**2 + zero[i]**2)), 
                      s=2, c='red', label='synedr3 data')
        # ax[i].scatter(sample['%s_synedr3'%band], np.log10(sample['%s_err_synedr3'%band]), 
        #             s=2, c='red', label='synedr3 data')
        ax[i].set_ylabel('%s_err'%band)
        ax[i].set_xlabel(band)
        if i == 1:
            ax[i].set_title('Mag-MagErr diagram for self checking')
        if i == 0:
            ax[i].legend()
    ax[3].scatter(
        sample[bands_dr2[1]+'_syndr2'] - sample[bands_dr2[2]+'_syndr2'], sample[bands_dr2[0]+'_syndr2'], 
        s=2, c='orange', label='syndr2 data')
    ax[3].scatter(
        sample[bands_dr2[1]+'_synedr3'] - sample[bands_dr2[2]+'_synedr3'], sample[bands_dr2[0]+'_synedr3'], 
        s=2, c='red', label='synedr3 data')
    ax[3].scatter(sample[bands_edr3[1]] - sample[bands_edr3[2]], sample[bands_edr3[0]], s=2, c='blue', label='edr3 data')
    ax[3].invert_yaxis()
    ax[3].legend()
    ax[3].set_xlabel('BP-RP (mag)')
    ax[3].set_ylabel('G (mag)')
    ax[3].set_title('CMD for checking')        
    
    #sample.to_csv("/home/shenyueyue/Projects/Cluster/data/%s_syn.csv"%name,index=False)

    
def test_MagErr_using_isoc():
    """To see which method for calculating error is resonable.
    
    Using observation data from gaia DR2 and EDR3 for comparison.
    DR2 mag error formula : e_Gmag=abs(-2.5/ln(10)*e_FG/FG)
    (E)DR3 mag error formulas : e_Gmag = sqrt((-2.5/ln(10)*e_FG/FG)**2 + sigmaG_0**2)
                                e_GBPmag = sqrt((-2.5/ln(10)*e_FGBP/FGBP)**2 + sigmaGBP_0**2))
                                e_GRPmag = sqrt((-2.5/ln(10)*e_FGRP/FGRP)**2 + sigmaGRP_0**2)) 
        with the G, G_BP, G_RP zero point uncertainties:
                                sigmaG_0 = 0.0027553202
                                sigmaGBP_0 = 0.0027901700
                                sigmaGRP_0 = 0.0037793818
    """
    name = 'Melotte_22'
    usecols = ['Gmag','G_BPmag','G_RPmag',
            'phot_g_n_obs_dr2','phot_bp_n_obs_dr2','phot_rp_n_obs_dr2',
            'Gmag_err','G_BPmag_err','G_RPmag_err',
            'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag',
            'phot_g_n_obs_edr3','phot_bp_n_obs_edr3','phot_rp_n_obs_edr3',
            'phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error']
    
    sample_obs = pd.read_csv('/home/shenyueyue/Projects/Cluster/data/melotte_22_edr3.csv',usecols=usecols)
    sample_obs = sample_obs.dropna().reset_index(drop=True)
    print(len(sample_obs))
    
    isochrones_dir = '/home/shenyueyue/Projects/Cluster/data/isocForMockCMD/'
    logage, mh, fb, dm = 7.89, 0.032, 0.35, 5.55
    logage_step, mh_step = 0.01, 0.01
    n_stars = len(sample_obs)
    
    # using DR2 obs data
    m = MockCMD(isochrones_dir=isochrones_dir)
    isochrone = m.get_isochrone(
        logage=logage,mh=mh, dm=dm, 
        logage_step=logage_step, mh_step=mh_step)
    sample_syn = m.sample_imf(fb, isochrone, n_stars)
    for _ in m.bands: 
        sample_syn[_+'_syn'] += dm
        
    e_dr2 = MagError(
        sample_obs=sample_obs,
        bands=['Gmag_syn','G_BPmag_syn','G_RPmag_syn'],
        nobs=['phot_g_n_obs_dr2','phot_bp_n_obs_dr2','phot_rp_n_obs_dr2']
    )
    sample_syn.loc[:,'Gmag_med_errsyndr2'], sample_syn.loc[:,'G_BPmag_med_errsyndr2'], sample_syn.loc[:,'G_RPmag_med_errsyndr2'] = (
        e_dr2.estimate_med_photoerr(sample_syn=sample_syn) 
    )
    sample_syn.loc[:,'Gmag_syndr2'], sample_syn.loc[:,'G_BPmag_syndr2'], sample_syn.loc[:,'G_RPmag_syndr2'] = (
        e_dr2.syn_sample_photoerr(sample_syn = sample_syn)
    )
    
    e_edr3 = MagError(
        sample_obs=sample_obs,
        bands=['Gmag_syn','G_BPmag_syn','G_RPmag_syn'],
        nobs = ['phot_g_n_obs_edr3','phot_bp_n_obs_edr3','phot_rp_n_obs_edr3']
    )
    sample_syn.loc[:,'Gmag_med_errsynedr3'], sample_syn.loc[:,'G_BPmag_med_errsynedr3'], sample_syn.loc[:,'G_RPmag_med_errsynedr3'] = (
        e_edr3.estimate_med_photoerr(sample_syn=sample_syn) 
    )
    sample_syn.loc[:,'Gmag_synedr3'], sample_syn.loc[:,'G_BPmag_synedr3'], sample_syn.loc[:,'G_RPmag_synedr3'] = (
        e_edr3.syn_sample_photoerr(sample_syn = sample_syn)
    )
    sample_obs.to_csv("/home/shenyueyue/Projects/Cluster/data/%s_obs.csv"%name,index=False)
    sample_syn.to_csv("/home/shenyueyue/Projects/Cluster/data/%s_syn.csv"%name,index=False)
    
        
    
if __name__=="__main__":
    test_MagErr_using_isoc()
    