import pandas as pd
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


#Offline phase
dbase = pd.read_csv('dbase_sliding_window_10-10.csv')

reference_tags = [6,16,23,25,32,37,43,49,51,58,62,64,71,88,99,100,103,106,115,118,125,132,137,147,153,160,161,166,182,191,196,200,204,210,222,234,240,246,248,249,251,259,260,268,269,288,299,300,303,309,314,322,326,330,345,350,353,360,362,368,375,386,393,398] # the best GA fitness

regions = []
regions.append([6,16,23,25,32,37,43,49])
regions.append([51,58,62,64,71,88,99,100])
regions.append([103,106,115,118,125,132,137,147])
regions.append([153,160,161,166,182,191,196,200])
regions.append([204,210,222,234,240,246,248,249])
regions.append([251,259,260,268,269,288,299,300])
regions.append([303,309,314,322,326,330,345,350])
regions.append([353,360,362,368,375,386,393,398])


landmarc = dbase.query("TagID == @reference_tags")
num_features = 40
land_predictors = landmarc.iloc[:, :num_features].values 

#Set regions in dataframe
counter = 0
for region in regions:
    landmarc.loc[landmarc['TagID'] == region,'region'] = counter
    counter = counter+1
    
#Train the classifier to predict the region    
vclass = landmarc.loc[:, 'region'].values
classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0)
classifier.fit(land_predictors, vclass)

#Train the regressors for each region and axis
unique_regions = np.unique(landmarc.loc[:,'region'].values)
regressor_X = []
regressor_Y = []
counter = 0
for r in unique_regions: 
    landmarc_regressor = landmarc.query("region == @r")    
    land_predictors_regressor = landmarc_regressor.iloc[:, :num_features].values 
    
    land_x = landmarc_regressor.loc[:, 'true_x'].values # x    
    regressor_X.append(RandomForestRegressor(n_estimators=1000,random_state=0))    
    regressor_X[counter].fit(land_predictors_regressor, land_x)
    
    land_y = landmarc_regressor.loc[:, 'true_y'].values # y    
    regressor_Y.append(RandomForestRegressor(n_estimators=1000,random_state=0))    
    regressor_Y[counter].fit(land_predictors_regressor, land_y)
    
    counter = counter+1
    
tags = np.unique(dbase.loc[:,'TagID'].values) 


#Online phase
for i in tags:    
    test_tag = [i] 
    test = dbase.query("TagID == @test_tag")

    test_predictors = test.iloc[:, :num_features].values    
    test_x = test.loc[:, 'true_x'].values # x
    test_y = test.loc[:, 'true_y'].values # y

    all_class_predictions = classifier.predict(test_predictors)
    
    counter = 0
    predictions_x = []
    predictions_y = []

    
    for k in all_class_predictions:
        predict_x = regressor_X[k].predict(test_predictors[counter].reshape(1,-1))
        predictions_x.append(predict_x)
        
        predict_y = regressor_Y[k].predict(test_predictors[counter].reshape(1,-1))
        predictions_y.append(predict_y)   
                  
        counter = counter+1

    
    mae_x = mean_absolute_error(predictions_x, test_x)
    mae_y = mean_absolute_error(predictions_y, test_y)
    
    
    print(str(i)+'\t'+str(mae_x).replace('.', ',')+'\t'+str(mae_y).replace('.', ',')+'\t'+str((mae_x+mae_y)/2).replace('.', ','))
    
























