# Imports
import tensorflow as tf
print(tf.test.is_gpu_available())
from hyperopt import STATUS_OK, STATUS_FAIL
from functools import partial
from os import chdir
from os.path import exists
from tensorflow.keras.backend import clear_session
from numpy import mean

from config_classes import hyperparameter_list, configuration
import optimizer_component as opt
import data_manager_component as data
import cnn_component as cnn
from ensemble import Model_Ensemble_CNN

def objective(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    try:
        loss, training_time = run_cnn(config, hyplist, hyperparameter_dict)
        #loss, training_time = run_ensemble(config, hyplist, hyperparameter_dict)
        #loss, training_time = run_all(config, hyplist, hyperparameter_dict)
        return { "loss": loss, 
                 "training_time": training_time, # TODO Training time doesn't seem to work atm
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
    skip_datasets = 1 # should be 1 normally

    for path, sheet in zip(config.dataset_file_paths[:-skip_datasets], config.dataset_sheet_titles[:-skip_datasets]): # All sheets except the last
        print(f"DATASET {i} of {len(config.dataset_file_paths)}")
        i = i + 1
        train, val, _, _, train_slices, val_slices = data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
        if(i == 0): training_time = cnn.fit_model_cnn(model, train, val, train_slices, val_slices, True, config, hyplist, hyperparameter_dict)
        else: _ = cnn.fit_model_cnn(model, train, val, train_slices, val_slices, False, config, hyplist, hyperparameter_dict)
    _, _, test, _, _, _ = data.process_sheet(config.dataset_file_paths[-1], config.dataset_sheet_titles[-1], config.cnn_testsplit, config, hyplist, hyperparameter_dict)
    (loss) = cnn.evaluate_model_cnn(model, test) # make into a evaluation function that does stuff like save execution time in a file!

    # Cleanup
    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory
    # TODO: Maybe gc.collect as well?

    return loss, training_time
    # TODO: At some point put this into a CNN-only function.

    # There are two options: 
    # 1. Each person is split into train/val/test
    # 2. Each person is split into train/val, except the last people that are purely test
    # Make lambda function for this

def fit_cnn_with_person(idx, length, person, model, use_timer):
    print(f"DATASET {idx} of {length}")
    model.fit(person.train, person.val, person.train_slices, person.val_slices, use_timer)

def run_cnn(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution

    skip_amount = 15 # should be 0
    test_people_num = -1 # last person is test
    dataset_amount = len(config.dataset_file_paths) - skip_amount + test_people_num
    train_iterable = zip(config.dataset_file_paths[skip_amount:test_people_num], config.dataset_sheet_titles[skip_amount:test_people_num])
    test_iterable = zip(config.dataset_file_paths[test_people_num:], config.dataset_sheet_titles[test_people_num:])

    first_path, first_sheet = next(train_iterable)
    person = data.process_sheet(first_path, first_sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
    model = cnn.Model_CNN(person.datashape, config, hyplist, hyperparameter_dict)
    fit_cnn_with_person(1, dataset_amount, person, model, True)

    for idx, (path, sheet) in enumerate(train_iterable):
        person = data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) # atm 1 person = 1 sheet. Will probably change to 1 p = 5 sheets
        fit_cnn_with_person(idx + 1, dataset_amount, person, model, False)
    
    losslst, train_time = [], model.train_time
    for path, sheet in test_iterable:
        person = data.process_sheet(path, sheet, config.cnn_testsplit, config, hyplist, hyperparameter_dict) # atm 1 person = 1 sheet. Will probably change to 1 p = 5 sheets
        losslst.append(model.evaluate(person.test))

    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory

    return mean(losslst), train_time

def run_ensemble(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution
    # TODO do the todos mentioned above

    test_people_num = -1 # last person is test
    train_iterable = zip(config.dataset_file_paths[:test_people_num], config.dataset_sheet_titles[:test_people_num])
    test_iterable = zip(config.dataset_file_paths[test_people_num:], config.dataset_sheet_titles[test_people_num:])
    train_people = [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in train_iterable]
    test_people = [data.process_sheet(path, sheet, config.cnn_testsplit, config, hyplist, hyperparameter_dict) for path, sheet in test_iterable]

    model = Model_Ensemble_CNN(train_people, test_people, config, hyplist, hyperparameter_dict)
    model.fit()
    loss = model.evaluate()
    # TODO: What the fuck jonathan. Jeg vil jo gerne træne 1 sheet ad gangen, hvad er der galt med dig
    # TODO: Måske prøv at bruge YIELD. Så behøver min kode ikke ligne lort

    training_time = -1

    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory
    return loss, training_time

    # TODO: FIX FIRST MY NON-ENSEMBLE METHOD. THEN TRY ENSEMBLE AGAIN (but only have it work on five or so datasheets)


#def adaboost_run():
#    for sheet in sheets:
#        train,val,test = processsheet(sheet, config.ada_datasplit)
#        weaklearner = model.compile()
#        weaklearner.fit(train, val)
#        weaklearner.eval(test)
#        return weaklearner
        # This might not make sense entirely with how you want to split up datasets for making individual weaklearners




config = configuration()
hyplist = hyperparameter_list()
partial_objective = partial(objective, config, hyplist)
# This is basically function currying. Defines our objective function with the config_dict parameter already present
opt.perform_hyperopt(partial_objective, hyplist.space(), 100)
