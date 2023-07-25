import numpy as np


class MockCMD:
    """
    mCMD = MockCMD(sample_obs)
    sample = mCMD.mock_stars(theta)
    cmd = mCMD.hist2d(sample, c_grid, m_grid)
    """

    def __init__(self, sample_obs):
        self.n_obs = len(sample_obs)
        self.n_tran_G, self.n_tran_GBP, self.n_tran_GRP \
            = MockCMD.extract_hp(sample_obs)
        # adaptive c_grid, m_grid

    @staticmethod
    def extract_hp(sample_obs):
        return 1, 1, 1, 1

    def set_imf(self, imf="kroupa", alpha=2):
        self.imf = lambda x: x ** 2

    def sample_imf_single(self, n_single):
        return  # a sample of mock single stars (N_s x 3) [G, GBP, GRP]

    def sample_imf_binary(self, n_binary):
        return  # a sample of mock binaries (N_b x 3) [G, GBP, GRP]

    def get_isochrone(self, tau, m_h, model="parsec"):
        # extract info
        self.teff = None
        self.Gmag = None
        self.GBP = None
        self.GRP = None
        return  # a truncated isochrone (label)

    def estimate_photoerror(self, sample):
        # interpolate & scale Gaia error table
        return  # uncertainties

    def mock_stars(self, theta):
        tau, m_h, fb, dm, Av, mag_min, mag_max = theta
        # step 1: tau, m_h -> isochrone [mass, G, BP, RP]
        isochrone = self.get_isochrone(tau, m_h)

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
