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
        #loss = run_cnn(config, hyplist, hyperparameter_dict)
        loss = run_ensemble(config, hyplist, hyperparameter_dict)
        return { "loss": loss, 
                 "status": STATUS_OK }
    except Exception as e:
        print(str(e))
        return { "status": STATUS_FAIL,
                 "exception": str(e) }

def fit_cnn_with_person(idx, length, person, model):
    print(f"DATASET {idx} of {length}")
    model.fit(person.train, person.val, person.train_slices, person.val_slices)

def run_cnn(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    # TODO Need to fix this one to work with multiple people with multiple sessions, such that I dont just assume that one person = 1 sheet

    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution

    skip_amount = 15 # should be 0
    test_people_num = -1 # last person is test
    dataset_amount = len(config.dataset_file_paths) - skip_amount + test_people_num
    train_iterator = zip(config.dataset_file_paths[skip_amount:test_people_num], config.dataset_sheet_titles[skip_amount:test_people_num])
    test_iterator = zip(config.dataset_file_paths[test_people_num:], config.dataset_sheet_titles[test_people_num:])

    first_path, first_sheet = next(train_iterator)
    person = data.process_sheet(first_path, first_sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
    model = cnn.Model_CNN(person.datashape, config, hyplist, hyperparameter_dict)
    fit_cnn_with_person(1, dataset_amount, person, model)

    for idx, (path, sheet) in enumerate(train_iterator):
        person = data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) # atm 1 person = 1 sheet. Will probably change to 1 p = 5 sheets
        fit_cnn_with_person(idx + 1, dataset_amount, person, model)
    
    loss_lst = []
    for path, sheet in test_iterator:
        person = data.process_sheet(path, sheet, config.cnn_testsplit, config, hyplist, hyperparameter_dict) # atm 1 person = 1 sheet. Will probably change to 1 p = 5 sheets
        loss_lst.append(model.evaluate(person.test))

    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory
    return mean(loss_lst)

def find_datashape(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # We load the first sheet as a test-run to see which datashape it ends up with
    first_path, first_sheet = config.dataset_file_paths[0], config.dataset_sheet_titles[0]
    person = data.process_sheet(first_path, first_sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
    return person.datashape

def run_ensemble(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution

    train_ppl_amount = 3
    train_spp = 5 # Train_Sheets_Per_Person
    train_sheets = train_ppl_amount * train_spp
    test_ppl_amount = 1
    test_spp = 5 # Test_Sheets_Per_Person 
    test_sheets = test_ppl_amount * test_spp

    train_people_files = [zip(config.dataset_file_paths[i:i+train_spp], config.dataset_sheet_titles[i:i+train_spp]) for i in range(0, train_sheets, train_spp)]
    test_people_files = [zip(config.dataset_file_paths[i:i+test_spp], config.dataset_sheet_titles[i:i+test_spp]) for i in range(train_sheets, train_sheets + test_sheets, test_spp)]
    # Note that this grabs the test sheets from right after our train sheets, not necessarily the last sheets
    # TODO Maybe put these person-tuple-calculations into the configuration class
    # TODO: GetPeopleIterator

    model = Model_Ensemble_CNN(find_datashape(config, hyplist, hyperparameter_dict), train_ppl_amount, config, hyplist, hyperparameter_dict)

    for idx, person in enumerate(train_people_files):
        print(f"PERSON {idx + 1} of {train_ppl_amount}")
        sessions = [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in person]
        model.fit(idx, sessions)

    loss = 0
    for idx, person in enumerate(test_people_files):
        sessions = [data.process_sheet(path, sheet, config.cnn_testsplit, config, hyplist, hyperparameter_dict) for path, sheet in person]
        loss = model.evaluate(sessions)

    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory
    # TODO: For cleanup maybe gc.collect as well?
    return loss

do_param_optimization = False

config = configuration()
hyplist = hyperparameter_list()

if do_param_optimization: 
    partial_objective = partial(objective, config, hyplist)
    # This is basically function currying. Defines our objective function with the config_dict parameter already present
    opt.perform_hyperopt(partial_objective, hyplist.space(), 100)

else: 
    objective(config, hyplist, hyplist.best_arguments())