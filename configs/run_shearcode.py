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
"""
replacefile = '/data2/brouwer/shearprofile/KiDS-GGL/brouwer/configs_margot/ggl_Ned.config'
findlist = ['@', '710']
replacelist = np.array([np.arange(6)+1, np.array([1724, 2634, 2201, 1577, 3379, 710])])
config_files = create_config(replacefile, findlist, replacelist)

# Test different riso values
replacefile = '/data/users/brouwer/Projects/EG_project/configs/ggl_k1000_replace.config'
findlist = np.array(['riso'])
replacelist = np.array([[3., 4.5, 6.]])
config_files = create_config(replacefile, findlist, replacelist)

# Test different fiso values
replacefile = '/data/users/brouwer/Projects/EG_project/configs/ggl_mdm_mice.config'
findlist = np.array(['fiso'])
replacelist = np.array([['0p0', '0p1', '0p2']])
config_files = create_config(replacefile, findlist, replacelist)

# Test different fiso values - Msat
replacefile = '/data/users/brouwer/Projects/EG_project/configs/ggl_k1000_replace.config'
findlist = np.array(['fiso'])
replacelist = np.array([[0., 0.1, 0.5, 1.]])
config_files = create_config(replacefile, findlist, replacelist)

replacefile = '/data/users/brouwer/Projects/EG_project/configs/ggl_mice_test.config'
findlist = np.array(['percvalue', 'distmin'])
replacelist = np.array([['0p3', '0p2', '0p1'], [3,4.5]])
config_files = create_config(replacefile, findlist, replacelist)
"""
# Run the three stellar mass limits
#replacefile = '/data/users/brouwer/Projects/EG_project/configs/ggl_k1000_replace.config'
replacefile = '/data/users/brouwer/Projects/EG_project/configs/ggl_mdm_k1000.config'
findlist = np.array(['masslim'])
replacelist = np.array([['GL', 'min', 'max']])
config_files = create_config(replacefile, findlist, replacelist)
"""
"""


# Running the created config files (one by one)
config_files = np.reshape(config_files, np.size(config_files))
print()
print('All config files:', config_files)

for c in config_files:
    print('Now running:', c)
    runname ='kids_ggl -c %s --esd'%c
    sub.run(runname.split(' '))
    
    
