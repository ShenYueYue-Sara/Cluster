import numpy as np
import pandas as pd
from scipy import integrate
from berliner import CMD
import random
from scipy import interpolate

class MockCMD:
    """
    mCMD = MockCMD(sample_obs)
    sample_syn = mCMD.mock_stars(theta)
    cmd = mCMD.hist2d(sample, c_grid, m_grid)
    """

    def __init__(self, photsyn="gaiaDR2", sample_obs=None):
        self.photsyn = photsyn
        if sample_obs != None:
            self.n_obs = len(sample_obs) # necessary? add sample_obs=None
        # adaptive c_grid, m_grid

    @staticmethod # Static methods do not need to be instantiated
    def extract_hyper(sample_obs):
        # hyper for estimate_photoerror
        n_tran_G = sample_obs['phot_g_n_obs']
        n_tran_GBP = sample_obs['phot_bp_n_obs']
        n_tran_GRP = sample_obs['phot_rp_n_obs']
        return n_tran_G, n_tran_GBP, n_tran_GRP

    def get_isochrone(self, logage, mh, dm, model="parsec"):
        if model == 'parsec':
            if self.photsyn == 'gaiaDR2':
                mag_max = 18
                c = CMD() # initialize berliner CMD
                isochrone = c.get_one_isochrone(
                    logage=logage, z=None, mh=mh,
                    photsys_file = self.photsyn)
                # truncate isochrone, PMS ~ EAGB
                isochrone = isochrone[(isochrone['label']>=0) & (isochrone['label']<=7)]
                # extract info
                self.logteff = 'logTe'
                self.bands = ['Gmag','G_BPmag','G_RPmag']
                self.Mini = 'Mini'
                self.mass_min = min(isochrone[ (isochrone['Gmag']+dm) <= mag_max ][self.Mini])
                self.mass_max = max(isochrone[self.Mini])
                # add evolutionary phase info
                self.phase = ['PMS','MS','SGB','RGB','CHEB','CHEB_b','CHEB_r','EAGB']
                for i in range(8):
                    isochrone[isochrone['label']==i]['phase'] = self.phase[i]

        elif model == 'mist':
            print("wait for developing")
            pass
        # a truncated isochrone (label), so mass_min and mass_max defined
        return isochrone
    
    def set_imf(self, imf='kroupa_2001', alpha=2):
        if imf == 'kroupa_2001':
            self.imf = lambda x: 2 * x ** -1.3 if x < 0.5 else x ** -2.3  # x2 for scale
        elif imf == 'salpeter':
            self.imf = lambda x: x ** -2.35
        elif imf == 'chabrier':
            self.imf = lambda x: x ** -1.55 if x < 1.0 else x ** -2.7
    
    def pdf_imf(self, m_i, mass_min, mass_max):
        if m_i < mass_min or m_i > mass_max:
            return 0
        else:
            return self.imf(m_i)/integrate.quad(self.imf, mass_min, mass_max)[0]
    
    def random_imf(self, n, mass_min, mass_max):
        mass = []
        c = self.pdf_imf(mass_min, mass_min, mass_max)
        for i in range(n):
            m_flag = 0
            while m_flag == 0:
                m_x = random.uniform(mass_min,mass_max)
                m_y = random.uniform(0,1)
                if m_y < self.pdf_imf(m_x, mass_min, mass_max)/c:
                    mass.append(m_x)
                    m_flag = 1
        return mass # a sample of n mass
    
    def sample_imf(self, fb, isochrone, n_stars, method='simple'):
        n_binary = int(n_stars*fb)
        sample_syn = pd.DataFrame(np.zeros((n_stars,1)), columns=['mass_pri'])
        sample_syn['mass_pri'] = self.random_imf(n_stars, self.mass_min, self.mass_max)
        # add binaries
        # if mass_sec != NaN, then binaries
        secindex = random.sample(list(sample_syn.index), k=n_binary)
        # hypothesis : secondary stars are MS only ! 
        if method == 'hypothesis':
            sample_syn.loc[secindex,'mass_sec'] = self.random_imf(n_binary, 
                                                    min(isochrone[isochrone['phase'] == 'MS']['Mini']), 
                                                    max(isochrone[isochrone['phase'] == 'MS']['Mini']))
        # without hypothesis : secondary mass range the same with primary star (ex. RGB + RGB exists)
        elif method == 'simple':
            sample_syn.loc[secindex,'mass_sec'] = self.random_imf(n_binary, self.mass_min, self.mass_max)
        # add mag for each band
        for band in self.bands: 
            # piecewise mass_mag relation
            id_cut = self.phase.index('CHEB')
            range1 = self.phase[0:id_cut]
            range2 = self.phase[id_cut:]
            mass_cut = min(isochrone[isochrone['phase'].isin(range2)][self.Mini])
            mass_mag_1 = interpolate.interp1d(isochrone[isochrone['phase'].isin(range1)][self.Mini],\
                isochrone[isochrone['phase'].isin(range1)][band])
            mass_mag_2 = interpolate.interp1d(isochrone[isochrone['phase'].isin(range2)][self.Mini],\
                isochrone[isochrone['phase'].isin(range2)][band])
            # add mag for primary(including single) & secondary star
            for m in ['pri','sec']:
                sample_syn.loc[sample_syn['mass_%s'%m] < mass_cut, '%s_%s'%(band,m)] = \
                    mass_mag_1(sample_syn[sample_syn['mass_%s'%m] < mass_cut]['mass_%s'%m])
                sample_syn.loc[sample_syn['mass_%s'%m] >= mass_cut, '%s_%s'%(band,m)] = \
                    mass_mag_2(sample_syn[sample_syn['mass_%s'%m] >= mass_cut]['mass_%s'%m]) 
            # add syn mag (for binaries, syn = f(pri,sec) )
            sample_syn['%s_syn'%band] = sample_syn['%s_pri'%band]
            sample_syn.loc[secindex,'%s_syn'%band] = \
                -2.5*np.log10( pow(10,-0.4*sample_syn.loc[secindex,'%s_pri'%band]) + pow(10,-0.4*sample_syn.loc[secindex,'%s_sec'%band]))
        return sample_syn # a sample of mock single & binaries stars [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn] 

    def estimate_photoerror(self, sample):
        # interpolate & scale Gaia error table
        return  # uncertainties

    def mock_stars(self, theta, n_stars):
        logage, mh, fb, dm, Av = theta # mag_min, mag_max not included yet!
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

    def eval_likelihood(sample_obs, sample_syn, c_grid=(0, 3, 0.1), m_grid=(0, 18, 0.1), ):
        cmd_obs = self.hist2d(sample_obs, c_grid, m_grid)
        cmd_syn = self.hist2d(sample_syn, c_grid, m_grid)
        return np.mean(np.square(cmd_obs - cmd_syn))
    
def main():
    name = 'Melotte_22'
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%(name))

if __name__=="__main__":
    main()