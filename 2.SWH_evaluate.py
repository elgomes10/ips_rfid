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

reference_tags = [1,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121,129,137,145,153,161,169,177,185,193,201,209,217,225,233,241,249,257,265,273,281,289,297,305,313,321,329,337,345,353,361,369,377,385,393] #mesh diagonal 

regions = []
regions.append([1,9,17,25,33,41,49])
regions.append([57,65,73,81,89,97])
regions.append([105,113,121,129,137,145])
regions.append([153,161,169,177,185,193])
regions.append([201,209,217,225,233,241,249])
regions.append([257,265,273,281,289,297])
regions.append([305,313,321,329,337,345])
regions.append([353,361,369,377,385,393])

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
    
























