import numpy as np
import pandas as pd
from scipy import integrate
from berliner import CMD

class MockCMD:
    """
    mCMD = MockCMD(sample_obs)
    sample = mCMD.mock_stars(theta)
    cmd = mCMD.hist2d(sample, c_grid, m_grid)
    """

    def __init__(self, sample_obs):
        self.n_obs = len(sample_obs)
        self.n_tran_G, self.n_tran_GBP, self.n_tran_GRP = MockCMD.extract_hp(sample_obs)
        # adaptive c_grid, m_grid

    @staticmethod # Static methods do not need to be instantiated
    def extract_hp(sample_obs):
        n_tran_G = sample_obs['phot_g_n_obs']
        n_tran_GBP = sample_obs['phot_bp_n_obs']
        n_tran_GRP = sample_obs['phot_rp_n_obs']
        return n_tran_G, n_tran_GBP, n_tran_GRP

    def get_isochrone(self, logage, mh, model="parsec", photsys="gaiaDR2"):
        if model == 'parsec':
            # initialize berliner CMD
            c = CMD()
            isochrone = c.get_one_isochrone(
                logage=logage, z=None, mh=mh,
                photsys_file = photsys)
            # truncate isochrone, PMS ~ EAGB
            isochrone = isochrone[(isochrone['label']>=0) & (isochrone['label']<=7)]
            # extract info
            self.logteff = isochrone['logTe']
            self.G = isochrone['Gmag']
            self.GBP = isochrone['G_BPmag']
            self.GRP = isochrone['G_RPmag']
            self.mass_min = min(isochrone['Mini'])
            self.mass_max = max(isochrone['Mini'])
        # a truncated isochrone (label), so mass_min and mass_max defined
        return isochrone
    
    def set_imf(self, imf='kroupa_2001', alpha=2):
        if imf == 'kroupa_2001':
            self.imf = lambda x: x ** -1.3 if x < 0.5 else x ** -2.3  
        elif imf == 'salpeter':
            self.imf = lambda x: x ** -2.35
        elif imf == 'chabrier':
            self.imf = lambda x: x ** -1.55 if x < 1.0 else x ** -2.7
    
    def pdf_imf(self, m_i):
        if m_i < self.mass_min or m_i > self.mass_max:
            return 0
        else:
            return self.imf(m_i)/integrate.quad(self.imf, self.mass_min, self.mass_max)[0]

    def sample_imf_single(self, n_single):
        return  # a sample of mock single stars (N_s x 3) [G, GBP, GRP]

    def sample_imf_binary(self, n_binary):
        return  # a sample of mock binaries (N_b x 3) [G, GBP, GRP]

    def estimate_photoerror(self, sample):
        # interpolate & scale Gaia error table
        return  # uncertainties

    def mock_stars(self, theta):
        logage, mh, fb, dm, Av, mag_min, mag_max = theta
        # step 1: logage, m_h -> isochrone [mass, G, BP, RP]
        isochrone = self.get_isochrone(logage, mh)

        # step 2: sample isochrone -> (N_obs, 4) [Teff, G, BP, RP]
        # single stars + binaries
        # add dm & Av
        # sample -> [Teff, m_G, m_GBP, m_GRP]

        # step 3: photometric uncertainties
        # interpolate & scale Gaia error table
        # add uncertainties
        # sample -> [m_G_obs, m_GBP_obs, m_GRP_obs]

        return  # sample

    @staticmethod
    def hist2d(sample, c_grid=(0, 3, 0.1), m_grid=(0, 18, 0.1), ):
        return np.histogram2d(*sample, bins=(c_bin, m_bin))

    def eval_likelihood(sample_obs, sample_mock, c_grid=(0, 3, 0.1), m_grid=(0, 18, 0.1), ):
        cmd_obs = self.hist2d(sample_obs, c_grid, m_grid)
        cmd_mock = self.hist2d(sample_mock, c_grid, m_grid)
        return np.mean(np.square(cmd_obs - cmd_mock))
    
def main():
    name = 'Melotte_22'
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%(name))

if __name__=="__main__":
    main()