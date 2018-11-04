import numpy as np
import h5py
from astropy.table import Table
import pylab as plt
from SN_Rate import SN_Rate
import numpy.lib.recfunctions as rf
from matplotlib import colors as mcolors
from scipy import interpolate
import sncosmo
from SN_Telescope import Telescope
import astropy.units as u

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def Read_Simu(simu_name):
    f = h5py.File(simu_name, 'r')
    #print(f.keys())
    for i, key in enumerate(f.keys()):
        simu = Table.read(simu_name, path=key)

    return simu

def Print_Simu():
    fieldid = 100
    X1 = -2.0
    Color = 0.2
    simu_name = input_dir+'/Fit_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    
    simu = Read_Simu(simu_name)
    
    print(len(simu))
    print(simu.dtype)
    idx = simu['z'] == 0.05
    sel = simu[idx]
    print(np.sqrt(sel['Cov_cc']))
    plt.plot(simu['z'],simu['mbfit'])
    plt.show()

def Plot_SNR(fieldid = [100], X1=0.0, Color=0.0):
    fontsize = 20
    simu_all={}
    for val in fieldid:
        simu_name = 'Output_Fit/Fit_Fake_DDF_'+str(val)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
        simu_all[val] = Read_Simu(simu_name)
    """
    simu_nameb = 'Output_Fit_airmass/Fit_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    simub = Read_Simu(simu_nameb)

    """
    fig,ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle('(X1,Color) = ('+str(X1)+','+str(Color)+')',fontsize = 20)
    prefix = 'LSST::'
    tot_label = []
    #fig.suptitle(cadence[fieldid]+'  (X1,Color) = ('+str(X1)+','+str(Color)+')',fontsize = fontsize)
    print('hello',colors_band)
    ik = -1
    for key, simu in simu_all.items():
        ik+=1
        for band in 'grizy':
            idx = (simu['SNR_'+band] > 0.01)&(simu['SNR_'+band] < 200.)
            sel = simu[idx]
            if ik == 1:
                tot_label.append(ax.errorbar(sel['z'],sel['SNR_'+band], label = band+' band',color=colors_band[band]))
            else:
                ax.errorbar(sel['z'],sel['SNR_'+band], label = band+' band',color=colors_band[band],ls='--')
        if ik ==1:
            ax.errorbar([0.8,0.85],[160,160],ls='-',color='k')
            ax.text(0.9,160,cadence[key],fontsize=18)
        else:
            ax.errorbar([0.8,0.85],[145,145],ls='--',color='k')
            ax.text(0.9,145,cadence[key],fontsize=18)
            
        """
        idxb = (simub['SNR_'+band] > 0.01)&(simub['SNR_'+band] < 200.)
        selb = simub[idx]
        """
        #ax.errorbar(selb['z'],selb['SNR_'+band], label = band+' band',ls='--')
    labs = [l.get_label() for l in tot_label]
    ax.legend(tot_label, labs, ncol=2,loc='best',prop={'size':fontsize},frameon=False)
    plt.gca().xaxis.grid(True)
    ax.set_xlabel('z',fontsize=fontsize)
    ax.set_ylabel('SNR',fontsize=fontsize)
    plt.yticks(size=20)
    plt.xticks(size=20)
    plt.show()
    

def Print_LC(fieldid = 101, X1 = 0.0, Color=0.0):

     simu_name = 'Output_Simu/LC_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    
     f = h5py.File(simu_name, 'r')
     print(f.keys())
     zref=0.5
     for i, key in enumerate(f.keys()):
         simu = Table.read(simu_name, path=key)
         if np.abs(simu.meta['z'] - zref)<1.e-5:
             print(cadence[fieldid],'X1=',X1,'Color=',Color,'z=',zref)
             simu.sort( 'time')
             idx = simu['snr_m5'] >= 5.
             sel = simu[idx]
             print(len(sel))
             for (ti,b,mag,magerr) in sel[['time','band','mag','magerr']]:
                 print(np.round(ti,3),b,np.round(mag,3),np.round(magerr,3))

def Plot_LC(fieldid = 101, X1 = 0.0, Color=0.0):

    telescope = Telescope(airmass = 1.1)
    
    simu_name = 'Output_Simu/LC_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    
    f = h5py.File(simu_name, 'r')
    print(f.keys())
    zref=0.9
    for i, key in enumerate(f.keys()):
        simu = Table.read(simu_name, path=key)
        if np.abs(simu.meta['z'] - zref)<1.e-5:
            print(cadence[fieldid],'X1=',X1,'Color=',Color,'z=',zref)
            Plot_LC_sncosmo(simu,telescope)

def Plot_Nmeas(fieldid = 100, X1 = 0.0, Color=0.0):

    telescope = Telescope(airmass = 1.1)
    
    simu_name = 'Output_Simu/LC_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    
    f = h5py.File(simu_name, 'r')
    print(f.keys())
    r = []
    for i, key in enumerate(f.keys()):
        simu = Table.read(simu_name, path=key)
        z = simu.meta['z']
        for band in 'grizy':
            idx = simu['band'] == 'LSST::'+band
            sel = simu[idx]
            idxb = sel['flux'] > 0.
            selb = sel[idxb]
            idxc = selb['flux']/selb['fluxerr'] > 5.
            r.append((z,band,len(selb),len(selb[idxc])))
    res = np.rec.fromrecords(r, names = ['z','band','Nobs','Nobs_5'])

    figa, axa = plt.subplots(ncols=1, nrows=1)

    for band in 'y':
        idx = res['band'] == band
        sel = res[idx]
        axa.plot(sel['z'],sel['Nobs'],color=colors_band[band],ls='-')
        axa.plot(sel['z'],sel['Nobs_5'],color=colors_band[band],ls='--')
        
    plt.show()
    
def Plot_SNR_band(fieldid=[112,115], X1=0.0,Color=0.0,zref=0.9):

    lc={}
    zref = zref
    for val in fieldid:
         simu_name = 'Output_Simu/LC_Fake_DDF_'+str(val)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
         f = h5py.File(simu_name, 'r')
         for i, key in enumerate(f.keys()):
            simu = Table.read(simu_name, path=key)
            if np.abs(simu.meta['z'] - zref)<1.e-5:
                lc[val] = simu
                break

    figa, axa = plt.subplots(ncols=2, nrows=3)
    figa.suptitle('z = '+str(zref)+' - (X1,Color) = ('+str(X1)+','+str(Color)+')',fontsize = 12)
    inx=dict(zip('ugrizy',[(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]))
    yband=dict(zip('ugrizy',[0,5,30,30,20,2]))
    for band in 'grizy':
        tot_label=[]
        ixa=inx[band][0]
        ixb=inx[band][1]
        
        ik = -1
        for key, val in lc.items():
            ik+=1
            idx = val['band'] == 'LSST::'+band
            sel = val[idx]
            if ik == 1:
                tot_label.append(axa[ixa][ixb].errorbar(sel['time'],sel['snr_m5'],color = colors_band[band],label=cadence[key],marker='o',ms=2))
            else:
                 tot_label.append(axa[ixa][ixb].errorbar(sel['time'],sel['snr_m5'],color = colors_band[band],label=cadence[key],ls='--',marker='s',ms=5))
        labs = [l.get_label() for l in tot_label]
        axa[ixa][ixb].legend(tot_label, labs, ncol=1,loc='best',prop={'size':12},frameon=False)
        axa[ixa][ixb].set_xlabel('time [mjd]',fontsize=12)
        axa[ixa][ixb].set_ylabel('SNR',fontsize=12)
        axa[ixa][ixb].text(110,yband[band],band,fontsize=12)
        axa[ixa][ixb].plot([100,220],[5.,5.],color='k',ls='-')
        axa[ixa][ixb].set_xlim([100,220])
    plt.yticks(size=12)
    plt.xticks(size=12)    
    plt.show()
    
def Print_X0(fieldid = 101, X1 = 0.0, Color=0.0):

    simu_name = 'Output_Simu/LC_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    
    f = h5py.File(simu_name, 'r')
    print(f.keys())
    for i, key in enumerate(f.keys()):
        simu = Table.read(simu_name, path=key)
        print(simu.meta['z'],simu.meta['dL'],simu.meta['X0norm'],simu.meta['X0'])
        
def Print_ExpTime():

    fieldids = [100,101,102,103]
    dirname='../Fake_Cadence/'
    ref_time = 320.*8.*3600.
    season_length = 180.
    Nfields = 4
    for fieldid in fieldids:
        res = np.load(dirname+'Fake_DDF_'+str(fieldid)+'.npy')
        #print(res.dtype)
        bands = np.unique(res['band'])
        exp_per_year =0
        for band in bands:
            idx = res['band'] == band
            sel = res[idx]
            cad = np.median(sel['mjd'][1:]-sel['mjd'][:-1])
            texp = np.median(sel['visitTime'])
            #print(band, np.median(cad),np.median(sel['visitTime']))
            Nvisits = season_length/cad
            expo_band = texp*float(Nvisits)
            print(band, cad, texp, Nvisits,expo_band)
            exp_per_year += expo_band
        print(fieldid,Nfields*exp_per_year/ref_time)
        
def Plot(var):
    fig,ax = plt.subplots(ncols=1, nrows=1)
    tot_label = []
    fontsize = 20
    for fieldid in fieldids:
        for (X1,Color) in x1_colors:
            simu_name = input_dir+'/Fit_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
            simu = Read_Simu(simu_name)
            #idx = (np.sqrt(simu['Cov_cc']) < 0.06)&(np.sqrt(simu['Cov_cc']) > 1.e-5)
            #simu = simu[idx]
            if (X1,Color) == (0.0,0.0):
                tot_label.append(ax.errorbar(simu['z'],np.sqrt(simu[var]),color = mycolors[fieldid],ls=myls[(X1,Color)],label=cadence[fieldid]))
            else:
                ax.plot(simu['z'],np.sqrt(simu[var]),color = mycolors[fieldid],ls=myls[(X1,Color)])
            
    zr = np.arange(0.,1.2,0.1)

    #plt.plot(zr, [0.04]*len(zr),color='m',ls='-.')

    ax.set_xlabel('z',fontsize=20)
    ax.set_ylabel('$X_0$',fontsize=20)
    plt.yticks(size=20)
    plt.xticks(size=20)
    labs = [l.get_label() for l in tot_label]
    ax.legend(tot_label, labs, ncol=2,loc='best',prop={'size':fontsize},frameon=False)
    ax.set_xlim([0.,1.1])
    ax.set_ylim([0.,0.06])
    ax.set_xticks(np.arange(0.0,1.2,0.1))
    #plt.grid()
    plt.gca().xaxis.grid(True)
    plt.show()

def Get(input_dir, fieldid, X1, Color):

    simu_name = input_dir+'/Fit_Fake_DDF_'+str(fieldid)+'_X1_'+str(X1)+'_Color_'+str(Color)+'.hdf5'
    simu = Read_Simu(simu_name)
    #idx = (np.sqrt(simu['Cov_cc']) < 0.06)&(np.sqrt(simu['Cov_cc']) > 1.e-5)
    idx = (np.sqrt(simu['Cov_cc']) > 1.e-5)
    simu = simu[idx]
    #idxb = (np.sqrt(simu['Cov_cc']) <= 0.04)
    return simu

def Plot_LC_sncosmo(table, telescope):
    print('What will be plotted', table)
    prefix = 'LSST::'
    print(table.dtype)
    for band in 'ugrizy':
        name_filter = prefix+band
        if telescope.airmass > 0:
                bandpass = sncosmo.Bandpass(
                telescope.atmosphere[band].wavelen,
                telescope.atmosphere[band].sb,
                name=name_filter,
                wave_unit=u.nm)
        else:
            bandpass = sncosmo.Bandpass(
                telescope.system[band].wavelen,
                telescope.system[band].sb,
                name=name_filter,
                wave_unit=u.nm)
            # print('registering',name_filter)
        sncosmo.registry.register(bandpass, force=True)

    source=sncosmo.get_source('salt2-extended',version='1.0')
    dust = sncosmo.OD94Dust()
    model = sncosmo.Model(source=source)
    print('metadata',table.meta)
    model.set(z = table.meta['z'],
              c = table.meta['Color'],
              t0 = table.meta['DayMax'],
              x1 = table.meta['X1'])
    z = table.meta['z']
    tab = table[['flux','fluxerr','band','zp','zpsys','time']]
    res, fitted_model = sncosmo.fit_lc(tab,model,['t0', 'x0', 'x1', 'c'],bounds={'z':(z-0.001, z+0.001)})
    print(res)
    print('jjj',fitted_model)
    sncosmo.plot_lc(data=tab, model=fitted_model, errors = res['errors'])
    plt.draw()
    plt.pause(20.)
    plt.close()
    
def Plot_Summary(fieldids,X1=0.0,Color=0.0,Plot = True):

    tot_label = []
    fontsize = 20
    if Plot:
        fig,ax = plt.subplots(ncols=1, nrows=1)
        fig.suptitle('(X1,Color) = ('+str(X1)+','+str(Color)+')',fontsize=fontsize)
   
    r = []
    for fieldid in fieldids:
        simua = Get('Output_Fit',fieldid,X1,Color)
        """
        simub = Get('Output_Fit_airmass',fieldid,X1,Color)
        print(simua.dtype)
        print(simua[['z','SNID','SNR_z','SNR_i']])
        print(simub[['z','SNID','SNR_z','SNR_i']])
        """
        fa = interpolate.interp1d(np.sqrt(simua['Cov_cc']), simua['z'])
        idx = np.sqrt(simua['Cov_cc'])<0.04
        r.append((fieldid,X1,Color,np.max(simua[idx]['z']),fa(0.04)))
        print(fieldid,X1,Color,np.max(simua[idx]['z']),fa(0.04))
        if Plot:
            tot_label.append(ax.errorbar(simua['z'],np.sqrt(simua['Cov_cc']),color = mycolors[fieldid],ls=myls[(X1,Color)],label=cadence[fieldid]))
              
    if Plot:          
        zr = np.arange(0.,1.3,0.1)
        plt.plot(zr, [0.04]*len(zr),color=colors['darkblue'],ls='-')
        ax.set_xlabel('z',fontsize=20)
        ax.set_ylabel('$\sigma_C$',fontsize=20)
        plt.yticks(size=20)
        plt.xticks(size=20)
        labs = [l.get_label() for l in tot_label]
        ax.legend(tot_label, labs, ncol=2,loc='best',prop={'size':18},frameon=False)
        if X1 == 0.0:
            ax.set_xlim([0.,1.2])
            ax.set_xticks(np.arange(0.0,1.2,0.1))
        else:
            ax.set_xlim([0.,0.9])
            ax.set_xticks(np.arange(0.0,0.9,0.1))
        ax.set_ylim([0.,0.06])
        
        #plt.grid()
        plt.gca().xaxis.grid(True)
        plt.show()

    return np.rec.fromrecords(r, names=['fieldid','X1','Color','zlim','zlim_interp'])

def Plot_NSN_New(fieldids,X1,Color,Plot=True):
    
    tab = Plot_Summary(fieldids,X1,Color, Plot = False)
    sn_rate = SN_Rate(rate='Perrett')
    
    zmin = 0.01
    fontsize = 20
    survey_area = 9.6 #deg2
    tot_label=[]
    r = []
    if Plot:
        fig,ax = plt.subplots(ncols=1, nrows=1)
        fig.suptitle('(X1,Color) = ('+str(X1)+','+str(Color)+')',fontsize=fontsize)
    for fieldid in fieldids:
        idx = tab['fieldid'] == fieldid
        sela = tab[idx]
        duration = season_length[fieldid]*30 #in days
        idxb = (sela['X1'] == X1)&(sela['Color'] == Color)
        val = sela[idxb]
        zmax_nointerp = val['zlim']
        zmax = np.asscalar(val['zlim_interp'])  
        X1 = np.asscalar(val['X1'])
        Color = np.asscalar(val['Color'])
        zz, rate, err_rate, nsn, err_nsn = sn_rate(
                zmin=zmin, zmax=zmax,
                duration=duration,
                 survey_area = survey_area,
                account_for_edges=True)
        r.append((fieldid,X1,Color,zmax,np.cumsum(nsn)[-1],Nfields[fieldid]))
        if Plot:
            tot_label.append(ax.errorbar([zmax],[np.cumsum(nsn)[-1]],marker = markers[fieldid], color=mycolors[fieldid], label=cadence[fieldid],markersize = 15,markerfacecolor='None',ls='None',markeredgewidth=3))

    if Plot:
        labs = [l.get_label() for l in tot_label]
        ax.legend(tot_label, labs, ncol=2,loc='best',prop={'size':fontsize},frameon=True,framealpha=1.0)
        plt.grid()
        plt.yticks(size=20)
        plt.xticks(size=20)
        ax.set_xlabel('$z_{lim}$',fontsize=20)
        ax.set_ylabel('$N_{SN}(z < z_{lim})$ per year',fontsize=20)
        plt.show()

    return np.rec.fromrecords(r,names=['fieldid','X1','Color','zlim','NSN','NFields'])

def Plot_Summary_NSN(fieldids,X1,Color):
     
    nsn_tot =Plot_NSN_New(fieldids,X1,Color, Plot=False)
    rb = []
    fontsize = 20
    for (X1,Color) in np.unique(nsn_tot[['X1','Color']]):
        idx = (np.abs(nsn_tot['X1']-X1)<1.e-5)&(np.abs(nsn_tot['Color']-Color)<1.e-5)
        sel = nsn_tot[idx]
        for key, val in all_names.items():
            NSN = 0
            zvals = []
            for iig in val:
                idxb = sel['fieldid'] == iig
                selb = sel[idxb]
               
                if len(selb) > 0:
                    NSN += selb['NSN']*selb['NFields']
                    zvals+=[selb['zlim']*len(selb['NFields'])]
            print(key, X1,Color,NSN*10,np.mean(zvals),np.median(zvals))
            rb.append((key, X1,Color,np.asscalar(NSN)*10,np.median(zvals)))

    tab = np.rec.fromrecords(rb,names=['cadence','X1','Color','NSN','zlim'])
    tot_label = []
    fig,ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle('(X1,Color) = ('+str(X1)+','+str(Color)+')',fontsize=fontsize)
    
    for cadence in np.unique(tab['cadence']):
        idx = tab['cadence'] == cadence
        sela = tab[idx]
        for val in sela:
            print('plotting',val['zlim'],val['NSN'],markers_c[cadence],mycolors_c[cadence])
            ax.plot([val['zlim']],[val['NSN']],marker=markers_c[cadence], color=mycolors_c[cadence],markersize = 15)
            if len(cadence) >= 10:
                ax.text((0.984+0.0015*(16-len(cadence)))*val['zlim'],1.01*val['NSN'],cadence,fontsize=18)
            else:
                ax.text(1.002*val['zlim'],val['NSN'],cadence,fontsize=18)
    plt.grid()
    ax.set_xlabel('$z_{lim}$',fontsize=20)
    ax.set_ylabel('$N_{SN}(z < z_{lim})$ [10 years]',fontsize=20)
    plt.yticks(size=20)
    plt.xticks(size=20)
    if X1 == 0.0:
        ax.set_xlim([1.0,1.08])
        #ax.set_ylim([4000,22000])
    plt.show()
  
            
#fieldids = [100,101,102,103,106,107,110,111]
#fieldids = [100,101,102,103,110,111,112,113,114,115]
"""
fieldids = [100,101,102,103,112,113,114,115]
x1_colors = [(0.0,0.0),(-2.0,0.2)]
mycolors = dict(zip(fieldids,['r','k','b','g','m','c',colors['orange'],colors['darkgrey'],colors['darkgreen'],colors['fuchsia']]))
markers = dict(zip(fieldids,['s','o','D','<','*','h','X','8','s','o']))
cadence = dict(zip(fieldids,['feature-like','cca-hackathon','WP_2_days','WP_1_day','Interleaved_2_days','Interleaved_1_day','Deep_2_days','Deep_1_day']))
"""

fieldids = [100,101,104,102,103,112,113,114,115,116,117]
cadence_sum=['feature-like','cca_hackathon','WP','WP_Interleaved','Deep','Deep_Interleaved']
x1_colors = [(0.0,0.0),(-2.0,0.2)]
#mycolors = dict(zip(fieldids,['r','k','b','g','m','c',colors['orange'],colors['darkgrey'],colors['darkgreen'],colors['darkgrey'],colors['darkgreen']]))
#markers = dict(zip(fieldids,['s','o','D','<','*','h','X','8','s','o','D']))
mycolors = dict(zip(fieldids,['r','k','k','b','b','g','g','m','m',colors['orange'],colors['orange']]))
markers = dict(zip(fieldids,['s','o','o','D','D','<','<','h','h','*','*']))
markers_c = dict(zip(cadence_sum,['s','o','D','<','*','h','>']))
mycolors_c = dict(zip(cadence_sum,['r','k','b','g','m',colors['orange']]))                        
cadence = dict(zip(fieldids,['feature-like','cca_hackathon_2_days','cca_hackathon_1_day','WP_2_days','WP_1_day','WP_Interleaved_2_days','WP_Interleaved_1_day','Deep_2_days','Deep_1_day','Deep_Interleaved_2_days','Deep_Interleaved_1_day']))
season_length=dict(zip(fieldids,[6.5,6.0,3.3,6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5]))
Nfields = dict(zip(fieldids,[4,4,1,4,1,4,1,4,1,4,1]))
myls = dict(zip(x1_colors,('-','-')))
input_dir = 'Output_Fit'
all_names = dict(zip(['feature-like','cca_hackathon','WP','WP_Interleaved','Deep','Deep_Interleaved'],[(100,0),(101,104),(102,103),(112,113),(114,115),(116,117)]))
colors_band=dict(zip('ugrizy',['b','c','g','y','r','m']))

fieldids_1=[104,103,113,115,117]
fieldids_2 = [100,101,102,112,114,116]

X1 = 0.0
Color = 0.0

#Plot_Summary(fieldids_1,X1=X1,Color=Color)
Plot_Summary(fieldids_2,X1=X1,Color=Color)
"""
Plot_NSN_New(fieldids_1,X1=X1,Color=Color)
Plot_NSN_New(fieldids_2,X1=X1,Color=Color)
"""
Plot_Summary_NSN(fieldids, X1=X1,Color=Color)

#Plot('X0')
#Print_Simu()
#Plot_SNR(fieldid=[112,115],X1=0.0,Color=0.0)
#Plot_SNR_band()
#Print_LC(fieldid = 103, X1=-2.0, Color=0.2)
#Plot_LC(fieldid = 113, X1=0.0, Color=0.0)
#Plot_Nmeas(fieldid = 100, X1=0.0, Color=0.0)
#Print_ExpTime()
#Print_X0(fieldid=103,X1=-2.0,Color=0.2)
#Print_X0(fieldid=103,X1=0.0,Color=0.0)
