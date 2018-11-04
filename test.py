import h5py
from astropy.table import Table

dirfich ='../Output_Simu/colossus_2665/WFD'
simu_name = dirfich+'/Simu_WFD_3.hdf5'
lc_name = dirfich+'/LC_WFD_3.hdf5'
f = h5py.File(simu_name, 'r')
print(f.keys())
for i, key in enumerate(f.keys()):
    simu = Table.read(simu_name, path=key)

print(simu.dtype)
for val in simu:
    print(val['id_hdf5'],val['n_lc_points'])
    lc =  Table.read(lc_name, path='lc_'+str(val['id_hdf5']))
    
