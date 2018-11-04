import os

def new_param(input_file, fieldid, X1, Color,zmax):
    with open(input_file, 'r') as file :
        filedata = file.read()

    filedata = filedata.replace('fieldid',str(fieldid))
    filedata = filedata.replace('x1val',str(X1))
    filedata = filedata.replace('colorval',str(Color))
    filedata = filedata.replace('zmax',str(zmax))
    
    new_file = 'param_fakes_'+str(fieldid)+'_'+str(X1)+'_'+str(Color)+'.yaml'
    
    #print(filedata)
    with open(new_file, 'w') as file :
        file.write(filedata)

    return new_file


input_file = 'param_fakes_gen.yaml'

#for fieldid in [100,101,102,103]:
for fieldid in [100,101,104,102,103,112,113,114,115,116,117]:
#for fieldid in [112,113,114,115]:
    zmax = 1.3
    for (X1,Color) in [(0.0,0.0),(-2.0,0.2)]:
        if X1 == -2.0:
            zmax = 0.9
        cmd = 'python SN_Catalog_Simulations/SN_Simulation/run_pipeline.py '+new_param(input_file,fieldid,X1,Color,zmax)
        os.system(cmd)
