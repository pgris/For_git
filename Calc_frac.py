import numpy as np

filelist = 'List_cadences.txt'

config = np.loadtxt(filelist,dtype={'names': ('name','Nu','Nother','duration','cadence','Nfields'),'formats': ('U15','i4','i4','f8','i4','i4')})

print(config)

Nyears = 10
Nref = 2769048
Nref_new = 8*3025*3600
cadences = ['feature-like','cca_hackathon','WP','Interleaved','Nicolas']

for cad in cadences:
    Nvisits = 0
    for val in config:
        if val['name'].count(cad):
            Nvisits+=val['Nfields']*(val['Nu']+val['Nother']*30*val['duration']/val['cadence'])

    Nvisits*=Nyears
    print(cad,100.*np.round(Nvisits/Nref,3),100.*np.round(Nvisits*30/Nref_new,3))

    
