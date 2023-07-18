import numpy as np
import matplotlib.pyplot as plt
plt.style.use("default")

'''
    functions kit for convenience :
        draw_isoc(isoc,photsys_file)
'''

'''
    draw isochrone and color the stellar period
'''
def draw_isoc(isoc,photsys_file):
    PMS = np.where((isoc['label']==0))[0] # pre main sequence
    MS = np.where((isoc['label']==1))[0] # main sequence
    SGB = np.where((isoc['label']==2))[0] # subgiant branch, or Hertzsprung gap for more intermediate+massive stars
    RGB = np.where((isoc['label']==3))[0] # red giant branch, or the quick stage of red giant for intermediate+massive stars
    CHEB = np.where((isoc['label']==4))[0] # core He-burning for low mass stars, or the very initial stage of CHeB for intermediate+massive stars
    CHEB_b = np.where((isoc['label']==5))[0] # the blueward part of the Cepheid loop of intermediate+massive stars
    CHEB_r = np.where((isoc['label']==6))[0] # the redward part of the Cepheid loop of intermediate+massive stars
    EAGB = np.where((isoc['label']==7))[0] # the early asymptotic giant branch, or a quick stage of red giant for massive stars
    
    index = np.where((isoc['label']>=0) & (isoc['label']<=7))[0]
    massmin_id = np.where(isoc['Mini']==min(isoc['Mini'][index]))[0]
    massmax_id = np.where(isoc['Mini']==max(isoc['Mini'][index]))[0]    
    
    if photsys_file == 'GaiaDR2':
        fig,ax =plt.subplots(figsize=(5,6))
        ax.plot(isoc["G_BPmag"][PMS]-isoc["G_RPmag"][PMS], isoc["Gmag"][PMS], color='grey', label='pre main sequence')
        ax.plot(isoc["G_BPmag"][MS]-isoc["G_RPmag"][MS], isoc["Gmag"][MS], color='green', label='main sequence')
        ax.plot(isoc["G_BPmag"][SGB]-isoc["G_RPmag"][SGB], isoc["Gmag"][SGB], color='orange', label='subgiant branch')
        ax.plot(isoc["G_BPmag"][RGB]-isoc["G_RPmag"][RGB], isoc["Gmag"][RGB], color='red', label='red giant branch')
        ax.plot(isoc["G_BPmag"][CHEB]-isoc["G_RPmag"][CHEB], isoc["Gmag"][CHEB], color='blue', label='core He-burning')
        ax.plot(isoc["G_BPmag"][CHEB_b]-isoc["G_RPmag"][CHEB_b], isoc["Gmag"][CHEB_b], color='skyblue', label='blueward Cepheid loop')
        ax.plot(isoc["G_BPmag"][CHEB_r]-isoc["G_RPmag"][CHEB_r], isoc["Gmag"][CHEB_r], color='pink', label='redward Cepheid loop')
        ax.plot(isoc["G_BPmag"][EAGB]-isoc["G_RPmag"][EAGB], isoc["Gmag"][EAGB], color='purple', label='early asymptotic giant branch')
        ax.scatter(isoc["G_BPmag"][massmin_id]-isoc["G_RPmag"][massmin_id], isoc["Gmag"][massmin_id], color='grey', facecolors='none', label='$Mini_{min} = %.2f$'%(isoc["Mini"][massmin_id]))
        ax.scatter(isoc["G_BPmag"][massmax_id]-isoc["G_RPmag"][massmax_id], isoc["Gmag"][massmax_id], color='black', facecolors='none', label='$Mini_{max} = %.2f$'%(isoc["Mini"][massmax_id]))
        ax.invert_yaxis()
        ax.legend()
        ax.set_title('log(Age)=%s, [M/H]=%s'%(str(isoc['logAge'][0]),str(isoc['MH'][0])))
        ax.set_xlabel("BP-RP (mag)")
        ax.set_ylabel("G (mag)")