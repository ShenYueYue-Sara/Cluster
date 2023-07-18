from berliner.parsec import CMD
import pandas as pd
import os
import configparser
'''
download the isochrones from Padova model
save the grid of isochrone to iso/
'''
config = configparser.ConfigParser()
config.read('config.ini')

dirs = config['DEFAULT']['PATH_DATA']+'iso/'+config['DEFAULT']['PHOT_SYS']
if not os.path.exists(dirs):
    os.makedirs(dirs)

c = CMD()
grid_age = (6.6,8.6,0.1)
grid_mh = (-2.0,0.4,0.1)
isoc_lgage,isoc_mh,isoc_list = c.get_isochrone_grid_mh(
        grid_logage=grid_age,grid_mh=grid_mh,photsys_file=config['DEFAULT']['PHOT_SYS'],n_jobs=int(config['DEFAULT']['NUM_WORKER']))

for i in range(len(isoc_list)):
    iso = isoc_list[i].to_pandas()
    #iso = iso[(iso['label']>0)&(iso['label']<=7)]
    logAge = iso.iloc[0]['logAge']
    MH = iso.iloc[0]['MH']
    iso.to_csv(dirs+'/iso_age_%s_mh_%s.csv'%(logAge,MH),index=False)


grid_age = (8.65,10.2,0.05)
grid_mh = (-2.0,0.4,0.1)
isoc_lgage,isoc_mh,isoc_list = c.get_isochrone_grid_mh(
        grid_logage=grid_age,grid_mh=grid_mh,photsys_file=config['DEFAULT']['PHOT_SYS'],n_jobs=int(config['DEFAULT']['NUM_WORKER']))

for i in range(len(isoc_list)):
    iso = isoc_list[i].to_pandas()
    #iso = iso[(iso['label']>0)&(iso['label']<=7)]
    logAge = iso.iloc[0]['logAge']
    MH = iso.iloc[0]['MH']
    iso.to_csv(dirs+'/iso_age_%s_mh_%s.csv'%(logAge,MH),index=False)


