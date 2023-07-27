import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
py_dir = os.path.dirname(sys.argv[0])
spline_csv = os.path.join(py_dir,'LogErrVsMagSpline.csv')

class Edr3LogMagUncertainty(object):
    """
    Estimate the log(mag) vs mag uncertainty for G, G_BP, G_RP based on Gaia EDR3 photometry.
    """
    
    def __init__(self, spline_csv):
        """
        """
        _df = pd.read_csv(spline_csv)
        splines = dict()
        splines['g'] = self.__init_spline(_df, 'knots_G', 'coeff_G')
        splines['bp'] = self.__init_spline(_df, 'knots_BP', 'coeff_BP')
        splines['rp'] = self.__init_spline(_df, 'knots_RP', 'coeff_RP')
        self.__splines = splines
        self.__nobs = {'g': 200, 'bp': 20, 'rp': 20}
    
    def estimate(self, band, nobs: np.array([], int) = 0, mag_range=None, mag_samples=1000):
        """
        Estimate the log(mag) vs mag uncertainty
        
        Parameters
        ----------
        band : str
            name of the band for which the uncertainties should be estimated (case-insentive)
        nobs : ndarray, int
            number of observations for which the uncertainties should be estimated.
            Must be a scalar integer value or an array of integer values.
        mag_range : array_like
            Magnitude range over which the spline should be evaluated.
            The default and maximum valid range is (4, 21)
        mag_samples : int
            Number evenly spaced magnitudes (over the mag_range interval) at which the splines 
            will be estimated. Default: 1000
        
        Returns
        -------
        df : DataFrame
            Pandas dataframe with the interpolated log(mag) uncertainty vs mag.
            The magnitude column is named mag_g, mag_bp, or mag_rp depending of the requested band.
            A column for each value of nobs is provided, in the default case the column is logU_200.
        """
        band = band.lower()
        if band not in ['g', 'bp', 'rp']:
            raise ValueError(f'Unknown band: {band}')
        if mag_range is None:
            mag_range = (4., 21.)
        else:
            if mag_range[0] < 4.:
                raise ValueError(f'Uncertainties can be estimated on in the range {band}[4, 21]')
            elif mag_range[1] > 21.:
                raise ValueError(f'Uncertainties can be estimated on in the range {band}[4, 21]')
            elif mag_range[0] > mag_range[1]:
                raise ValueError('Malformed magnitude range')
        xx = np.linspace(mag_range[0], mag_range[1], mag_samples)
        __cols = self.__compute_nobs(band, xx, nobs)
        __dc = {f'mag_{band}': xx, **__cols}
        return pd.DataFrame(data=__dc)
    
    def __init_spline(self, df, col_knots, col_coeff):
        __ddff = df[[col_knots, col_coeff]].dropna()
        return interpolate.BSpline(__ddff[col_knots], __ddff[col_coeff], 3, extrapolate=False)
    
    def __compute_nobs(self, band, xx, nobs):
        if isinstance(nobs, int):
            nobs = [nobs]
        __out = dict()
        for num in nobs:
            if num < 0:
                raise ValueError(f'Number of observations should be strictly positive')
            if num == 0:
                __out[f'logU_{self.__nobs[band]:d}'] = self.__splines[band](xx)
            else:
                __out[f'logU_{num:d}'] = self.__splines[band](xx) - np.log10(np.sqrt(num) / np.sqrt(self.__nobs[band]))
        return __out

class MagError(Edr3LogMagUncertainty):
    '''
    gaia Mag [G, BP, RP] -> Mag_err [G_err, BP_err, RP_err] 
    return median uncertainty for a sample which n_obs obeys poisson distribution (hypothesis)
    
    method1: from Edr3LogMagUncertainty
    method2: from observation mag_magerr relation (corrected Nobs effect)
    '''
    
    def __init__(self, sample_obs=None, med_nobs=None, spline_csv=spline_csv, bands=None):
        super(MagError,self).__init__(spline_csv)
        if bands:
            self.bands = bands
        else:
            self.bands=['Gmag','G_BPmag','G_RPmag']
        self.spline_g = self._Edr3LogMagUncertainty__splines['g']
        self.spline_bp = self._Edr3LogMagUncertainty__splines['bp']
        self.spline_rp = self._Edr3LogMagUncertainty__splines['rp']
        if med_nobs:
            self.med_nobs = med_nobs
        else:
            if sample_obs is not None:
                self.med_nobs = MagError.extract_med_nobs(sample_obs)
            else:
                raise ValueError('please enter med_nobs OR sample_obs')
    
    @staticmethod
    def extract_med_nobs(sample_obs, nobs=['phot_g_n_obs','phot_bp_n_obs','phot_rp_n_obs']):
        # extract the median value of n_obs(number of observation)
        med_nobs = []
        for i in range(3):
            med = int(np.median(sample_obs[nobs[i]]))
            med_nobs.append(med)
        return med_nobs
    
    def random_n_obs(self, n_stars):
        # generate n_obs(number of observation) which obeys poisson(miu=med_nobs)
        g_n_obs = np.random.poisson(self.med_nobs[0], n_stars)
        bp_n_obs = np.random.poisson(self.med_nobs[1], n_stars)
        rp_n_obs = np.random.poisson(self.med_nobs[2], n_stars)
        return g_n_obs, bp_n_obs, rp_n_obs
    
    def estimate_med_photoerr(self, sample_syn):
        # return statistic value (MEDIAN) of the error distribution, considering number of observation
        # step 1 : generate synthetic n_obs for each band
        n_stars = len(sample_syn)
        g_n_obs, bp_n_obs, rp_n_obs = self.random_n_obs(n_stars)
        # step 2 : calculate mag_err when Nobs = 200(for G) / 20(for BP,RP)
        g_med_err = 10**(self.spline_g(sample_syn[self.bands[0]]) - np.log(np.sqrt(g_n_obs) / np.sqrt(200)))
        bp_med_err = 10**(self.spline_bp(sample_syn[self.bands[1]]) - np.log(np.sqrt(bp_n_obs) / np.sqrt(20)))
        rp_med_err = 10**(self.spline_rp(sample_syn[self.bands[2]]) - np.log(np.sqrt(rp_n_obs) / np.sqrt(20)))
        return g_med_err, bp_med_err, rp_med_err
    
    def syn_sample_photoerr(self, sample_syn):
        # return synthetic band mag (with statistic error) which obey with N(band,band_med_err)
        n_stars = len(sample_syn)
        normal_sample = np.random.normal(n_stars)
        g_med_err, bp_med_err, rp_med_err = self.estimate_med_photoerr(sample_syn)
        g_syn = (g_med_err/0.67) * normal_sample + sample_syn[self.bands[0]]
        bp_syn = (bp_med_err/0.67) * normal_sample + sample_syn[self.bands[1]]
        rp_syn = (rp_med_err/0.67) * normal_sample + sample_syn[self.bands[2]]
        return g_syn, bp_syn, rp_syn
        
def main():
    name = 'Melotte_22'
    usecols = ['Gmag','G_BPmag','G_RPmag','phot_g_n_obs','phot_bp_n_obs','phot_rp_n_obs','Gmag_err','G_BPmag_err','G_RPmag_err']
    sample = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%name, usecols=usecols)
    sample = sample.dropna().reset_index(drop=True)

    e = MagError(sample_obs=sample)
    g_med_err, bp_med_err, rp_med_err = e.estimate_med_photoerr(sample_syn=sample)
    g_syn, bp_syn, rp_syn = e.syn_sample_photoerr(sample_syn = sample)
    sample['Gmag_err_syn'], sample['G_BPmag_err_syn'], sample['G_RPmag_err_syn'] = g_med_err, bp_med_err, rp_med_err
    sample['Gmag_syn'], sample['G_BPmag_syn'], sample['G_RPmag_syn'] = g_syn, bp_syn, rp_syn
    sample.to_csv("/home/shenyueyue/Projects/Cluster/data/%s_syn.csv"%name,index=False)
    
if __name__=="__main__":
    main()

