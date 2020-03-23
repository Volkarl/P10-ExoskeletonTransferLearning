# Imports
import tensorflow as tf
print(tf.test.is_gpu_available())
from hyperopt import STATUS_OK, STATUS_FAIL
from functools import partial
from os import chdir
from os.path import exists

from config_classes import hyperparameter_list, configuration
import optimizer_component as opt
import data_manager_component as data
import cnn_component as cnn

def objective(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    try:
        loss, training_time = run_all(config, hyplist, hyperparameter_dict)
        return { "loss": loss, 
                 "training_time": training_time,
                 "status": STATUS_OK }
    except Exception as e:
        print(str(e))
        return { "status": STATUS_FAIL,
                 "exception": str(e) }

def run_all(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    # Config_dict is set by this file. Specific hyperparams are given by hyperopt
    # Access individual hyperparams by using hyperparameters[hyplist.HYPERPARAM_NAME]


    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution


    # TODO: Remove train and validation slices. Make some kind of object to contain batched data and its information
    # Or maybe just figure out how to get data out of my batched objects, but I'm not sure if this even works
    _, _, _, shape, _, _ = data.process_sheet(config.dataset_file_paths[0], config.dataset_sheet_titles[0], config.cnn_datasplit, config, hyplist, hyperparameter_dict)
    model = cnn.compile_model_cnn(shape, config, hyplist, hyperparameter_dict)
    
    i, training_time = 0, 0
    skip_datasets = 16 # should be 1 normally

    for path, sheet in zip(config.dataset_file_paths[:-skip_datasets], config.dataset_sheet_titles[:-1]): # All sheets except the last
        print(f"DATASET {i} of {len(config.dataset_file_paths)}")
        i = i + 1
        train, val, _, _, train_slices, val_slices = data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
        if(i == 0): training_time = cnn.fit_model_cnn(model, train, val, train_slices, val_slices, True, config, hyplist, hyperparameter_dict)
        else: _ = cnn.fit_model_cnn(model, train, val, train_slices, val_slices, False, config, hyplist, hyperparameter_dict)
    _, _, test, _, _, _ = data.process_sheet(config.dataset_file_paths[-1], config.dataset_sheet_titles[-1], config.cnn_testsplit, config, hyplist, hyperparameter_dict)
    (loss) = cnn.evaluate_model_cnn(model, test) # make into a evaluation function that does stuff like save execution time in a file!

    # TODO REMOVE THE MODEL FROM MEMORY WHEN IM DONE USING IT


    return loss, training_time
    # TODO: At some point put this into a CNN-only function.


    #if(use_cnn): cnn(config, hyplist, hyperparameter_dict)
    #else: adaboost(config, hyplist, hyperparameter_dict)
    # One sheet at a time!

    # There are two options: 
    # 1. Each person is split into train/val/test
    # 2. Each person is split into train/val, except the last people that are purely test
    # Make lambda function for this


#def adaboost_run():
#    for sheet in sheets:
#        train,val,test = processsheet(sheet, config.ada_datasplit)
#        weaklearner = model.compile()
#        weaklearner.fit(train, val)
#        weaklearner.eval(test)
#        return weaklearner
        # This might not make sense entirely with how you want to split up datasets for making individual weaklearners

# def cnn_run(config: configuration, hyplist: hyperparameter_list, hyp_dict):



config = configuration()
hyplist = hyperparameter_list()
partial_objective = partial(objective, config, hyplist)
# This is basically function currying. Defines our objective function with the config_dict parameter already present
opt.perform_hyperopt(partial_objective, hyplist.space(), 100)
