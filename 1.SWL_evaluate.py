# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

dbase = pd.read_csv('dbase_sliding_window_10-10.csv')

reference_tags = [1,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,129,137,145,153,161,169,177,185,193,201,209,217,225,233,241,249,257,265,273,281,289,297,305,313,321,329,337,345,353,361,369,377,385,393] #diagonal mesh 
landmarc = dbase.query("TagID == @reference_tags")

num_features = 40
          
land_predictors = landmarc.iloc[:, :num_features].values  
land_x = landmarc.loc[:,'true_x'].values # x   
land_y = landmarc.loc[:,'true_y'].values # y
regressor_X = RandomForestRegressor(n_estimators=1000,random_state=0)    
regressor_X.fit(land_predictors, land_x) 

regressor_Y = RandomForestRegressor(n_estimators=1000,random_state=0)    
regressor_Y.fit(land_predictors, land_y) 

tags = np.unique(dbase.loc[:,'TagID'].values) 

for i in tags:   
    test_tag = [i] 
    test = dbase.query("TagID == @test_tag")

    test_predictors = test.iloc[:, :num_features].values   
    test_x = test.loc[:,'true_x'].values # x
    test_y = test.loc[:,'true_y'].values # y


    predict_x = regressor_X.predict(test_predictors)
    predict_y = regressor_Y.predict(test_predictors)

        
    mae_x = mean_absolute_error(predict_x, test_x)
    mae_y = mean_absolute_error(predict_y, test_y)
    
    
    print(str(i)+'\t'+str(mae_x).replace('.', ',')+'\t'+str(mae_y).replace('.', ',')+'\t'+str((mae_x+mae_y)/2).replace('.', ','))