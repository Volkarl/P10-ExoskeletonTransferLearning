# Imports
import tensorflow as tf
print(tf.test.is_gpu_available())
from hyperopt import STATUS_OK
from functools import partial

from config_classes import hyperparameter_list, configuration
import optimizer_component as opt
import data_management_temp as data
#import data_management_component as data
#import cnn_baseline as cnn
#import evaluation_component as eval

def objective(config: configuration, hyplist: hyperparameters, hyperparameter_dict): 
    loss, exec_time = run_all(config, hyplist, hyperparameter_dict)
    return { 'loss': loss, 'status': STATUS_OK, 'exec_time': exec_time }

config = configuration(1,1,1,1,1,1,1,1,1) # TODO FIX ME
hyplist = hyperparameter_list()
partial_objective = partial(objective, config = config, hyplist = hyplist)
# This is basically function currying. Defines our objective function with the config_dict parameter already present
opt.perform_hyperopt(partial_objective, hyplist.space, 100)


def run_all(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    # Config_dict is set by this file. Specific hyperparams are given by hyperopt
    # Access individual hyperparams by using hyperparameters[hyplist.HYPERPARAM_NAME]


    if(use_cnn): cnn(config, hyplist, hyperparameter_dict)
    else: adaboost(config, hyplist, hyperparameter_dict)


    # One sheet at a time!

    #for sheet, path in config.dataset_sheet_titles, config.dataset_file_paths:
    #    train, val = data.process_sheet(sheet, path, config, hyplist, hyperparameter_dict)
    # what about when we dont want val data?

    # maybe have for each person, and then five sessions per person instead?
    # For adaboost or boosting we need one weaklearner per person, which means that we need train/val/test for each person
    # But then how does that work for a single CNN? 


    # There are two options: 
    # 1. Each person is split into train/val/test
    # 2. Each person is split into train/val, except the last people that are purely test
    # Make lambda function for this

    # for person in people
    #x,y = dataslicing(person, 0.9)
    #z,_ = dataslicing(person, 0)

    

#    if(singlesplit):
 #       people_data = [dataslicing(person, config.val_percent, config.test_percent) for person in people] 
        # the dataslicing itself returns a named tuple class data(x, y, z), which is train val test
        # each element in peopledata has three elements

#    else:
#        people_data = [dataslicing(person, config.val_percent, config.test_percent) for person in people[:-1]]
        # Here, each people_data item holds only two values
#        test_data = slicePerson(people[-1])
        # THIS IS STILL FUCKED, BECAUSE I DONT WANT TO KEEP IT ALL IN MEMORY




 #   data.process_data(config, hyplist, hyperparameter_dict) 
    # TODO: Fix these. Rewrite them in new temp files, simplify and remove unecessary stuff like my plotting etc.
    cnn.run_cnn(hyperparameters, hyplist) 
    eval.evaluate_results()
    return loss, exec_time

def adaboost():
    for sheet in sheets:
        train,val,test = processsheet(sheet, 0, 0.8, 0.9, 1)
        weaklearner = model.compile()
        weaklearner.fit(train, val)
        weaklearner.eval(test)
        return weaklearner
        # This might not make sense entirely with how you want to split up datasets for making individual weaklearners

def cnn():
    cnn = model.compile()
    for sheet in sheets[:-1]:
        train,val,_ = processsheet(sheet, 0, 0.9, 1, 1)
        cnn.fit(train, val)
    _,_,test = processsheet(sheet[-1], 0, 0, 0, 1)
    cnn.eval(test) # make into a evaluation function that does stuff like save execution time in a file!
