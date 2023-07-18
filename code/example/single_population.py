import configparser
import pandas as pd
import numpy as np
import scipy.interpolate as spl
import astropy.units as u
import random
import os
import glob
from scipy import integrate 
from joblib import Parallel,delayed
import re

config = configparser.ConfigParser()
config.read('config.ini')

dirs = config['DEFAULT']['PATH_PACk']+config['OBJECT']['NAME'] + '_' + config['OBJECT']['FIELD'] + '/' + 'ssp/'
if not os.path.exists(dirs):
    os.makedirs(dirs)


#generate the mass catalog follow the selected IMF 
def fun_IMF(m_x,label='Kroupa'):
    '''
    Initial Mass Funcation
    label = ['Salpeter','Kroupa','Chabrier']
    '''
    while label == 'Salpeter':
        if m_x < 0.1:
            return 0
        elif m_x < 100:
            return m_x**(-2.35)
        else : 
            return 0
    
    while label == 'Kroupa':
        if m_x < 0.08 : 
            return 0
        elif m_x < 0.5 : 
            return 2*m_x**(-1.3)
        elif m_x < 150:
            return m_x**(-2.3)
        else :
            return 0
        
        
    while label == 'Chabrier':
        if m_x < 0.07 : 
            return 0
        elif m_x < 1.0 : 
            return m_x**(-1.55)
        elif m_x < 100 : 
            return m_x**(-2.7)
        else : 
            return 0
            
    
def pdf_IMF(m_x,m_min,m_max,label='Kroupa'):
    if m_x < m_min or m_x > m_max:
        return 0
    else : 
        return fun_IMF(m_x,label)/integrate.quad(lambda x : fun_IMF(x,label), m_min, m_max)[0]
    
def random_IMF(m_n, m_min, m_max,label='Kroupa'):
    m_result = []
    c = pdf_IMF(m_min, m_min, m_max, label)
    for i in range(m_n):
        m_flag = 0
        while m_flag == 0:
            m_x = random.uniform(m_min, m_max)
            m_y = random.uniform(0,1)
            if m_y < pdf_IMF(m_x, m_min, m_max, label)/c:
                m_result.append(m_x)
                m_flag = 1
    return m_result

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts    

uncer = pd.read_csv(config['OBJECT']['PATH_OBS'] + 'data/' + config['OBJECT']['NAME'] + '/' +  config['OBJECT']['NAME'] + '_' +  config['OBJECT']['FIELD'] + '/UNCER_FIT.csv')
cmap = np.load('cmap.npy')
class generate_cat : 
    '''generate catalogue from the templated isochrone file iso/'''
    def __init__(self,dist_mod=15,n_jobs=6,labelIMF='Kroupa',fbin=0.35,
                 bands = ['F475Wmag','F606Wmag','F814Wmag'],complimit=0.2):
        '''
        n_jobs = 4
        dist_mod = 15
        bands = ['F435Wmag','F475Wmag','F555Wmag','F606Wmag','F625Wmag''F775Wmag','F814Wmag']
        labelIMF = 'Kroupa'
        fbin = 0.35
        '''
        self.n_jobs = n_jobs
        self.dist_mod = dist_mod
        self.bands = bands
        self.labelIMF = labelIMF
        self.fbin = fbin
        self.complimit = complimit
          
    def get_iso(self):
        '''
        chose isochrones from iso file
        '''
        cols = ['logAge','MH','Mini','label']
        cols.extend(self.bands)
        isoc_file = sorted(glob.glob(config['DEFAULT']['PATH_DATA']+'iso/'+config['DEFAULT']['PHOT_SYS']+'/*.csv'),key = numericalSort)   
        isoc_list = []
        for each in isoc_file:            
            temiso = pd.read_csv('%s'%(each))
            temiso = temiso[cols]
            isoc_list.append(temiso)
        self.isoc_list = isoc_list
        
        
    def get_cat(self,temiso):
        '''
        input iso
        dist in unit "Mpc"
        mag_limit : maximum magnitude at g band       
        '''
        tem = temiso
        tem = tem[(tem['label']>0)&(tem['label']<=7)]
        logAge = tem.iloc[0]['logAge']
        MH = tem.iloc[0]['MH']
        tem['F814Wmag_obs'] = tem['F814Wmag'] + self.dist_mod

        fband_mass = spl.interp1d(tem['F814Wmag_obs'],tem['Mini'])
        if float(config['OBJECT']['YMIN']) >  tem['F814Wmag_obs'].min():
            massmax = fband_mass(config['OBJECT']['YMIN'])
        if float(config['OBJECT']['YMIN']) <= tem['F814Wmag_obs'].min():
            massmax = tem.iloc[-1]['Mini']
        if float(config['OBJECT']['YMAX']) >  tem['F814Wmag_obs'].max():
            massmin = fband_mass(tem['F814Wmag_obs'].max())
        if float(config['OBJECT']['YMAX']) <= tem['F814Wmag_obs'].max():
            massmin = fband_mass(config['OBJECT']['YMAX'])
        
        mass = random_IMF(100000,massmin,massmax,label=self.labelIMF)
        cat = pd.DataFrame(np.zeros((100000,1)),columns=['primass'])
        cat['primass'] = mass
        cat['age'] = logAge
        cat['mh'] = MH
        secindex = random.sample(list(cat.index),k=int(len(cat)*self.fbin))
        cat['secmass'] = np.zeros(len(cat))
        bintem = temiso[temiso['label']==1]
        secmass = random_IMF(int(len(cat)*self.fbin),bintem['Mini'].min(),bintem['Mini'].max(),label=self.labelIMF)
        cat['secmass'][secindex] = secmass

        for band in self.bands: 
            cat['%s_pri'%(band)] = np.zeros(len(cat))
            cat['%s_sec'%(band)] = np.zeros(len(cat))
            
            if tem.iloc[-1]['label'] <= 3:
                iso1 = tem
                fmass_band1 = spl.interp1d(iso1['Mini'],iso1['%s'%(band)])
                for h in range(len(cat)):
                    cat['%s_pri'%(band)][h] = fmass_band1(cat['primass'][h])
                    
                cat['%s_true'%(band)] = cat['%s_pri'%(band)]
                for h in secindex: 
                    cat['%s_sec'%(band)][h] = fmass_band1(cat['secmass'][h])
                    cat['%s_true'%(band)][h] = -2.5*np.log10(pow(10,-0.4*cat['%s_pri'%(band)][h])+pow(10,-0.4*cat['%s_sec'%(band)][h]))
            
            if tem.iloc[-1]['label']>3: 
                iso1 = tem[tem['label']<=3]#MS-RGB
                fmass_band1 = spl.interp1d(iso1['Mini'],iso1['%s'%(band)],fill_value='extrapolate')       
                iso2 = tem[tem['label']>=4]#CHEB-EAGB,core He-burning, not include TP-AGB
                fmass_band2 = spl.interp1d(iso2['Mini'],iso2['%s'%(band)])
                mass_cut = iso2['Mini'].min()
                for h in range(len(cat)):
                    if cat['primass'][h] < mass_cut:
                        cat['%s_pri'%(band)][h] = fmass_band1(cat['primass'][h])   
                    else : 
                        cat['%s_pri'%(band)][h] = fmass_band2(cat['primass'][h])
                
                cat['%s_true'%(band)] = cat['%s_pri'%(band)]
                for h in secindex:
                    if cat['secmass'][h] < mass_cut:
                        cat['%s_sec'%(band)][h] = fmass_band1(cat['secmass'][h])
                    else :
                        cat['%s_sec'%(band)][h] = fmass_band2(cat['secmass'][h])
                    cat['%s_true'%(band)][h] = -2.5*np.log10(pow(10,-0.4*cat['%s_pri'%(band)][h])+pow(10,-0.4*cat['%s_sec'%(band)][h]))           
        
        for band in self.bands:
            cat['%s'%(band)] = cat['%s_true'%(band)]+self.dist_mod
            ferr_mag = spl.interp1d(uncer['mag'],uncer['%s_err'%(band)],fill_value='extrapolate')
            cat['%s_obs'%(band)] = np.zeros(len(cat))
    
            for i in range(len(cat)):
                temmag = cat['%s'%(band)][i]
                temerr = ferr_mag(temmag)
                cat['%s_obs'%(band)][i] = random.gauss(temmag,temerr)       
            cat['%s_err'%(band)] = cat['%s_obs'%(band)]-cat['%s'%(band)] 
        
        for band in self.bands:
            cat = cat[(cat['%s_err'%(band)]<0.5)&(cat['%s_err'%(band)]>-0.5)]
     
        #apply completeness map to the ssp with uncertainties 
        xbins = np.arange(float(config['OBJECT']['XMIN']),float(config['OBJECT']['XMAX']),float(config['OBJECT']['XBINSIZE']))
        ybins = np.arange(float(config['OBJECT']['YMIN']),float(config['OBJECT']['YMAX']),float(config['OBJECT']['YBINSIZE']))
        grid_x,grid_y = np.meshgrid(xbins,ybins)
        grid_z = np.zeros((len(ybins),len(xbins))) 

        cat['color_obs'] = cat['%s_obs'%(self.bands[0])] - cat['%s_obs'%(self.bands[1])]
        cat['mag_obs'] =   cat['%s_obs'%(self.bands[1])]
        temcat = cat.dropna()
        temcat = temcat[(temcat['color_obs']>=xbins[0])&(temcat['color_obs']<=xbins[-1])&(temcat['mag_obs']>=ybins[0])&(temcat['mag_obs']<=ybins[-1])]

        drop_index = []
        if len(temcat)>0:
            for i in range(len(ybins)-1):
                temcaty = temcat[(temcat['mag_obs']>=ybins[i])&(temcat['mag_obs']<ybins[i+1])]
                for j in range(len(xbins)-1):
                    temcatxy = temcaty[(temcaty['color_obs']>=xbins[j])&(temcaty['color_obs']<xbins[j+1])]
                    if cmap[i,j] >= float(config['OBJECT']['COMPLIMIT']):
                        temdrop_index = random.choices(temcatxy.index,k=int(len(temcatxy)*(1-cmap[i,j])))
                        drop_index.extend(temdrop_index)
                    if cmap[i,j] <  float(config['OBJECT']['COMPLIMIT']):
                        drop_index.extend(temcatxy.index)
                    grid_z[i,j] = len(temcatxy)
        savecat = temcat.drop(drop_index)
        savecat.to_csv(dirs+'cat_age_%s_mh_%s.csv'%(logAge,MH),index=False)

        with open(dirs+'ncat_age_%s_mh_%s.txt'%(logAge,MH),'a') as file:
            file.write('%-20s %-20s \n'%('N_ini' , len(cat)))
            file.write('%-20s %-20s \n'%('N_view', len(temcat)))
            file.write('%-20s %-20s \n'%('N_comp', len(savecat)))

    
    def apply_parallel(self):
        Parallel(n_jobs=self.n_jobs)(delayed(self.get_cat)(temiso=_iso) for _iso in (self.isoc_list))

#%%
g = generate_cat(dist_mod = float(config['OBJECT']['DIST_MOD']),
                 n_jobs = int(config['DEFAULT']['NUM_WORKER']),
                 bands = config['OBJECT']['BANDS'].split(';'),
                 complimit = float(config['OBJECT']['COMPLIMIT']))
g.get_iso()
g.apply_parallel()
