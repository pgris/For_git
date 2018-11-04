import numpy as np
import pylab as plt

from SN_Telescope import Telescope

telescope = Telescope(airmass = 1.1)

telescope.Plot_Throughputs()

bands = 'ugrizy'
colors_band=dict(zip(bands,['b','c','g','y','r','m']))

r = []
for z in np.arange(0,1.5,0.01):
    for band in bands:
        print(band,z,telescope.mean_wavelength[band]/(1. + z))
        r.append((band,z,telescope.mean_wavelength[band]/(1. + z)))

res = np.rec.fromrecords(r,names= ['band','z','Mean_wavelength'])

figa, axa = plt.subplots(ncols=1, nrows=1)

tot_label=[]
for band in np.unique(res['band']):
    idx = res['band'] == band
    sel = res[idx]
    tot_label.append(axa.errorbar(sel['z'],sel['Mean_wavelength'],color = colors_band[band],label=band+' - band'))

labs = [l.get_label() for l in tot_label]
axa.legend(tot_label, labs, ncol=2,loc='best',prop={'size':20},frameon=False)
axa.plot([np.min(res['z']),np.max(res['z'])],[300,300],color='k')
axa.plot([np.min(res['z']),np.max(res['z'])],[800,800],color='k')
axa.set_xlabel('$z$',fontsize=20)
axa.set_ylabel('Mean wavelength [nm]',fontsize=20)
axa.set_xlim([0.,1.2])
plt.yticks(size=20)
plt.xticks(size=20)
plt.gca().xaxis.grid(True)
plt.show()
