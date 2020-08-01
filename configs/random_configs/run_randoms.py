#!/usr/bin/python

import numpy as np
import sys
import subprocess as sub

def create_config(replacefile, searchlist, replacelist):
    
    config_files = np.array([])

    f = open(replacefile,'r')
    filedata = f.read()
    f.close()

    shape = np.shape(replacelist)
    if len(shape) == 1:
        Nr = 1
    else:
        Nr = shape[1]
    
    for r in range(Nr):

        newdata = filedata
        fileout = '%s_%s'%(replacefile, r+1)
        
        print()
        print('For %s:'%fileout)
        
        for s in range(len(searchlist)):
            search = '%s'%searchlist[s]
            try:
                replace = '%s'%replacelist[s, r]
            except:
                replace = '%s'%replacelist[s]
            
            print('     - replace %s with %s'%(search, replace))
        
            newdata = newdata.replace(search, replace)
        
        f = open(fileout, 'w')
        f.write(newdata)
        f.close()
    
        config_files = np.append(config_files, [fileout])
        
    return config_files


# Defining the config file(s)

# Run the random ESD profiles
replacefile = '/data/users/brouwer/Projects/EG_project/configs/random_configs/ggl_mdm_randoms.config'
findlist = np.array(['randnum'])
replacelist = np.array([np.arange(50)])
config_files = create_config(replacefile, findlist, replacelist)


# Running the created config files (one by one)
config_files = np.reshape(config_files, np.size(config_files))
print()
print('All config files:', config_files)

for c in config_files:
    print('Now running:', c)
    runname ='kids_ggl -c %s --esd'%c
    sub.run(runname.split(' '))
