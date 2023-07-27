import numpy as np
import pandas as pd
from scipy import integrate
from berliner import CMD
import random
from scipy import interpolate
from gaia_magerror import  MagError

class MockCMD:
    """
    mCMD = MockCMD(sample_obs)
    sample_syn = mCMD.mock_stars(theta)
    cmd = mCMD.hist2d(sample, c_grid, m_grid)
    """

    def __init__(self, photsyn="gaiaDR2", sample_obs=None): # better give sample_obs!! another way is untested
        self.photsyn = photsyn
        if sample_obs is not None:
            self.n_obs = len(sample_obs) # necessary? add sample_obs=None
            if self.photsyn == "gaiaDR2":
                self.med_nobs = MagError.extract_med_nobs(sample_obs)
        self.set_imf()
        # adaptive c_grid, m_grid

    @staticmethod # Static methods do not need to be instantiated
    def extract_hyper(sample_obs):
        # hyper for estimate_photoerror
        med_nobs = MagError.extract_med_nobs(sample_obs)
        return med_nobs
            
    def get_isochrone(self, model="parsec", **kwargs):
        logage = kwargs['logage']
        mh = kwargs['mh']
        dm = kwargs['dm']
        if model == 'parsec':
            if self.photsyn == 'gaiaDR2':
                mag_max = 18
                c = CMD() # initialize berliner CMD
                isochrone = c.get_one_isochrone(
                    logage=logage, z=None, mh=mh,
                    photsys_file = self.photsyn)
                # truncate isochrone, PMS ~ EAGB
                isochrone = isochrone[(isochrone['label']>=0) & (isochrone['label']<=7)].to_pandas()
                # extract info
                self.logteff = 'logTe'
                self.bands = ['Gmag','G_BPmag','G_RPmag']
                self.Mini = 'Mini'
                self.mass_min = min(isochrone[ (isochrone['Gmag']+dm) <= mag_max ][self.Mini])
                self.mass_max = max(isochrone[self.Mini])
                # add evolutionary phase info
                self.phase = ['PMS','MS','SGB','RGB','CHEB','CHEB_b','CHEB_r','EAGB']
                for i in range(8):
                    # isochrone[isochrone['label']==i]['phase'] = self.phase[i]
                    id = np.where(isochrone['label']==i)[0]
                    isochrone.loc[id,'phase'] = self.phase[i]

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
            masssec_min = min(isochrone[isochrone['phase'] == 'MS'][self.Mini])
            masssec_max = max(isochrone[isochrone['phase'] == 'MS'][self.Mini])
        # without hypothesis : secondary mass range the same with primary star (ex. RGB + RGB exists)
        elif method == 'simple':
            masssec_min = self.mass_min
            masssec_max = self.mass_max
        sample_syn.loc[secindex,'mass_sec'] = self.random_imf(n_binary, masssec_min, masssec_max)
        # add mag for each band
        for band in self.bands: 
            # piecewise mass_mag relation
            id_cut = self.phase.index('CHEB')
            range1 = self.phase[0:id_cut]
            range2 = self.phase[id_cut:]
            mass_cut = min(isochrone[isochrone['phase'].isin(range2)][self.Mini])
            mass_mag_1 = interpolate.interp1d(isochrone[isochrone['phase'].isin(range1)][self.Mini],\
                isochrone[isochrone['phase'].isin(range1)][band], fill_value='extrapolate')
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
    
    def estimate_syn_photoerror(self, sample_syn, **kwargs): # bands=['Gmag','G_BPmag','G_RPmag'],bands_err=['Gmag_err','G_BPmag_err','G_RPmag_err']
        # add photoerror for sample_syn
        # method 1 , wait for developing : interpolate & scale Edr3LogMagUncertainty
        if self.photsyn == "gaiaDR2":
            if self.med_nobs is not None:
                #e = MagError(med_nobs=self.med_nobs,bands=[_+'_syn',for _ in self.bands])
                e = MagError(med_nobs=self.med_nobs, bands=['Gmag_syn','G_BPmag_syn','G_RPmag_syn'])
            else:
                raise ValueError('please give sample_obs while initial MockCMD()')
            g_med_err, bp_med_err, rp_med_err = e.estimate_med_photoerr(sample_syn=sample_syn)
            g_syn, bp_syn, rp_syn = e.syn_sample_photoerr(sample_syn=sample_syn)
            sample_syn[self.bands[0]+'_syn'], sample_syn[self.bands[1]+'_syn'], sample_syn[self.bands[2]+'_syn'] = g_syn, bp_syn, rp_syn
            sample_syn[self.bands[0]+'_err_syn'], sample_syn[self.bands[1]+'_err_syn'], sample_syn[self.bands[2]+'_err_syn'] = g_med_err, bp_med_err, rp_med_err
        # method 2 : fit mag_magerr relation in sample_obs for each cluster directly
        # if 'bands' in kwargs: # sample_obs have DIFFERENT band name/order with sample_syn
        #     print('NOTE! PLEASE ENTER THE OBSERVATION BAND ORDER ALIGNED WITH %s'%(self.bands))
        #     bands = kwargs['bands']
        # bands = self.bands # sample_obs have SAME band name with sample_syn
        # if 'bands_err' in kwargs:
        #     bands_err = kwargs['bands_err']
        # bands_err = ['%s_err'%band for band in bands]
        # # fit mag_magerr relation for each band
        # for band,band_err in zip(bands,bands_err):
        #     x, y = sample_obs[band], sample_obs[band_err]
        #     mask = ~np.isnan(x) & ~np.isnan(y)
        #     x, y = x[mask], y[mask]
        #     mag_magerr = np.polyfit(x,y,3)
        return  sample_syn # a sample of mock stars WITH band error added [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn, _err_syn]

    def mock_stars(self, theta, n_stars):
        logage, mh, fb, dm, Av = theta # mag_min, mag_max not included yet!
        print(logage, mh, dm)
        # step 1: logage, m_h -> isochrone [mass, G, BP, RP]
        isochrone = self.get_isochrone(logage=logage,mh=mh, dm=dm)
        
        # step 2: sample isochrone -> n_stars [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn]
        # single stars + binaries
        sample_syn = self.sample_imf(fb, isochrone, n_stars)
        # add dm & Av
        for _ in self.bands: 
            sample_syn[_+'_syn'] += dm
        # sample -> [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn]
        
        # step 3: photometric uncertainties
        # interpolate & scale Edr3LogMagUncertainty
        # add uncertainties
        # sample -> [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn, _err_syn]
        sample_syn = self.estimate_syn_photoerror(sample_syn)
        return  sample_syn# sample
    
    @staticmethod
    def extract_CMD(sample,band_a,band_b,band_c):
        m = sample[band_a]
        c = sample[band_b] - sample[band_c]
        return c,m

    @staticmethod
    def hist2d_norm(c, m, c_grid=(0, 3, 0.1), m_grid=(6, 16, 0.1) ): #def hist2d(*sample.T,...):
        # adaptive grid wait for developing
        # define grid edges
        cstart,cend,cstep = c_grid
        mstart,mend,mstep = m_grid
        c_bin = np.arange(cstart,cend,cstep)
        m_bin = np.arange(mstart,mend,mstep)
        H,_,_ = np.histogram2d(c, m, bins=(c_bin, m_bin))
        H = H / np.sum(H)
        return H

    def eval_lnlikelihood(c_obs, m_obs, c_syn, m_syn, c_grid=(0, 3, 0.1), m_grid=(6, 16, 0.1) ):
        H_obs = MockCMD.hist2d_norm(c_obs, m_obs, c_grid, m_grid)
        H_syn = MockCMD.hist2d_norm(c_syn, m_syn, c_grid, m_grid)
        chi2 = np.square(H_obs - H_syn) / H_obs
        return -1/2*chi2
    
def main():
    name = 'Melotte_22'
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv"%(name))
    
    m = MockCMD(sample_obs=sample_obs)
    theta = (7.89, 0.032, 0.35, 5.55, 0)
    n_stars = 1000
    sample_syn = m.mock_stars(theta,n_stars)
    c_syn, m_syn = MockCMD.extract_CMD(sample_syn, band_a='Gmag_syn', band_b='G_BPmag_syn', band_c='G_RPmag_syn')
    c_obs, m_obs = MockCMD.extract_CMD(sample_obs, band_a='Gmag', band_b='G_BPmag', band_c='G_RPmag')

if __name__=="__main__":
    main()
    
#%%
import numpy as np
c_grid=(0, 3, 0.1)
cstart,cend,cstep = c_grid
np.arange(cstart,cend,cstep)

from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

xedges = [0, 1, 3, 5]
yedges = [0, 2, 3, 4, 6]

x = np.random.normal(2, 1, 100)
y = np.random.normal(1, 1, 100)
H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
print(H)
print('\n')
print(H[0])
print(np.sum(H))
print(H/np.sum(H))
# %%
