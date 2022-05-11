# -*- coding: utf-8 -*-

import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

import pygad


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
dbase = pd.read_csv('dbase_sliding_window_10-10.csv')
num_features = 40



def fitness_function(solution, solution_idx):
    reference_tags = list(np.sort(solution))
    landmarc = dbase.query("TagID == @reference_tags")   
    land_predictors = landmarc.iloc[:, :num_features].values 
    
    
    #Set regions in dataframe
    regions = []
    arr = np.sort(solution)
    for i in range(50,401,50):
        newarr = arr[(arr > i-50) & (arr <= i)]
        regions.append(list(newarr))
        
        
    counter = 0
    penalty = False
    for region in regions:
        if len(region>0):
            landmarc.loc[landmarc['TagID'].isin(region),'region'] = counter
            counter = counter+1
        else:
            penalty = True
    
    #Punish solutions with regions without reference tags    
    if penalty:
        mean_error = 100
    else:
        #Train the classifier to predict the region 
        landmarc['region'] = landmarc['region'].astype(int)
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
            mean_error = (mae_x+mae_y)/2
    
    return 1/mean_error


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(fitness=(1/max(ga_instance.best_solutions_fitness))))


num_generations = 100
num_parents_mating = 3

sol_per_pop = 6
num_genes = 64

init_range_low = 1
init_range_high = 400

gene_space = range(1,401)

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       sol_per_pop=sol_per_pop,
                       gene_type=int,
                       gene_space = gene_space,
                       allow_duplicate_genes=False,
                       stop_criteria='reach_1',
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=callback_generation)

ga_instance.run()
ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

ga_instance.save('ga_instance.pygad')

solution = list(solution)
print(solution)




