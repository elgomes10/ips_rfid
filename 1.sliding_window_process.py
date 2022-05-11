# -*- coding: utf-8 -*-

import pandas as pd
import warnings

import numpy as np

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

dbase = pd.read_csv('observations_autoparts.csv')


newdbase = dbase[0:0]
first_window_size = 10
window_size = 10

tags = np.unique(dbase.loc[:,'TagID'].values) 
    
for TagID in tags:

    dbase_TagID = dbase.query("TagID == @TagID")
    dbase_TagID = dbase_TagID.reset_index(drop=True)
    
    #Fill the first window
    window = dbase_TagID.iloc[0:first_window_size, 0:8].values
    vmean = np.around(np.mean(window,axis=0),4)
    vmin = np.min(window,axis=0)
    vmax = np.max(window,axis=0)
    vstd = np.around(np.std(window,axis=0),4)
    
    dbase_TagID.loc[0:first_window_size-1,['avg_rssi_antenna1','avg_rssi_antenna2','avg_rssi_antenna3','avg_rssi_antenna4','avg_rc_antenna1','avg_rc_antenna2','avg_rc_antenna3','avg_rc_antenna4']] = vmean 
    dbase_TagID.loc[0:first_window_size-1,['min_rssi_antenna1','min_rssi_antenna2','min_rssi_antenna3','min_rssi_antenna4','min_rc_antenna1','min_rc_antenna2','min_rc_antenna3','min_rc_antenna4']] = vmin
    dbase_TagID.loc[0:first_window_size-1,['max_rssi_antenna1','max_rssi_antenna2','max_rssi_antenna3','max_rssi_antenna4','max_rc_antenna1','max_rc_antenna2','max_rc_antenna3','max_rc_antenna4']] = vmax
    dbase_TagID.loc[0:first_window_size-1,['stddev_rssi_antenna1','stddev_rssi_antenna2','stddev_rssi_antenna3','stddev_rssi_antenna4','stddev_rc_antenna1','stddev_rc_antenna2','stddev_rc_antenna3','stddev_rc_antenna4']] = vstd
    
    #Fill the other windows
    for line in range(first_window_size-1,len(dbase_TagID)):
        if line < window_size:
            beg = 0
            end = line+1
        else:
            beg = line - window_size +1
            end = line+1
    
        window = dbase_TagID.iloc[beg:end, 0:8].values
        
        vmean = np.around(np.mean(window,axis=0),4)
        vmin = np.min(window,axis=0)
        vmax = np.max(window,axis=0)
        vstd = np.around(np.std(window,axis=0),4)
        
        dbase_TagID.loc[line,['avg_rssi_antenna1','avg_rssi_antenna2','avg_rssi_antenna3','avg_rssi_antenna4','avg_rc_antenna1','avg_rc_antenna2','avg_rc_antenna3','avg_rc_antenna4']] = vmean 
        dbase_TagID.loc[line,['min_rssi_antenna1','min_rssi_antenna2','min_rssi_antenna3','min_rssi_antenna4','min_rc_antenna1','min_rc_antenna2','min_rc_antenna3','min_rc_antenna4']] = vmin
        dbase_TagID.loc[line,['max_rssi_antenna1','max_rssi_antenna2','max_rssi_antenna3','max_rssi_antenna4','max_rc_antenna1','max_rc_antenna2','max_rc_antenna3','max_rc_antenna4']] = vmax
        dbase_TagID.loc[line,['stddev_rssi_antenna1','stddev_rssi_antenna2','stddev_rssi_antenna3','stddev_rssi_antenna4','stddev_rc_antenna1','stddev_rc_antenna2','stddev_rc_antenna3','stddev_rc_antenna4']] = vstd
     
    newdbase = newdbase.append(dbase_TagID) 
    
#column reorder  

column_to_reorder = newdbase.pop('TagID')
newdbase.insert(len(newdbase.columns), 'TagID', column_to_reorder) 
  
column_to_reorder = newdbase.pop('read')
newdbase.insert(len(newdbase.columns), 'read', column_to_reorder)    

column_to_reorder = newdbase.pop('true_x')
newdbase.insert(len(newdbase.columns), 'true_x', column_to_reorder) 

column_to_reorder = newdbase.pop('true_y')
newdbase.insert(len(newdbase.columns), 'true_y', column_to_reorder) 

#Sort and save file
newdbase = newdbase.sort_values(by=['read','TagID'])
newdbase = newdbase.reset_index(drop=True)
newdbase.to_csv('dbase_sliding_window_'+str(first_window_size)+'-'+str(window_size)+'.csv',index = False)

    