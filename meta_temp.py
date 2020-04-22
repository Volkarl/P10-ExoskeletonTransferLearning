# Imports
import tensorflow as tf
print(tf.test.is_gpu_available())
from hyperopt import STATUS_OK, STATUS_FAIL
from functools import partial
from os import chdir
from os.path import exists
from tensorflow.keras.backend import clear_session
from Fixed_AdaBoostR2 import AdaBoostR2 
import numpy as np

from config_classes import hyperparameter_list, configuration
import optimizer_component as opt
import data_manager_component as data
import cnn_component as cnn
from ensemble import Model_Ensemble_CNN

def objective(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    try:
        #loss = run_cnn(config, hyplist, hyperparameter_dict)
        #loss = run_ensemble(config, hyplist, hyperparameter_dict)
        loss = run_AdaBoostR2(config, hyplist, hyperparameter_dict)
        return { "loss": loss, 
                 "status": STATUS_OK }
    except Exception as e:
        print(str(e))
        return { "status": STATUS_FAIL,
                 "exception": str(e) }

def fit_cnn_with_person(idx, length, person, model):
    print(f"DATASET {idx} of {length}")
    model.fit(person.train, person.val, person.train_slices, person.val_slices)

def setup_windows_linux_pathing():
    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution

def run_cnn(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    # TODO Need to fix this one to work with multiple people with multiple sessions, such that I dont just assume that one person = 1 sheet

    setup_windows_linux_pathing()

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
    return np.mean(loss_lst)

def find_datashape(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # We load the first sheet as a test-run to see which datashape it ends up with
    first_path, first_sheet = config.dataset_file_paths[0], config.dataset_sheet_titles[0]
    person = data.process_sheet(first_path, first_sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
    return person.datashape

def run_ensemble(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    setup_windows_linux_pathing()

    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()
    model = Model_Ensemble_CNN(find_datashape(config, hyplist, hyperparameter_dict), config.train_ppl_amount, config, hyplist, hyperparameter_dict)

    for idx, person in enumerate(train_ppl_file_iter):
        print(f"PERSON {idx + 1} of {config.train_ppl_amount}")
        sessions = [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in person]
        model.fit(idx, sessions)

    loss = 0
    for idx, person in enumerate(test_ppl_file_iter):
        sessions = [data.process_sheet(path, sheet, config.cnn_testsplit, config, hyplist, hyperparameter_dict) for path, sheet in person]
        loss = model.evaluate(sessions)

    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory
    # TODO: For cleanup maybe gc.collect as well?
    return loss

def flatten_split_sessions(sessions):
    sliced_X, sliced_Y = [], []
    for session in sessions: # Flatten the outer lists
        # Seperate the sensor values (x) from the ground truth values (y)
        sliced_X.extend(session.x_train)
        sliced_X.extend(session.x_val)
        sliced_Y.extend(session.y_train)
        sliced_Y.extend(session.y_val)

        # TODO: BE AWARE THAT WE CURRENTLY IGNORE BATCHSIZE, EPOCHS, SHUFFLEBUFFERSIZE
    return np.array(sliced_X), np.array(sliced_Y) # make into numpy arrays, such that we have a shape property

def unpack_sessions(person_iterator, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    sessions = []
    for person in person_iterator:
        for session in [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in person]:
            sessions.append(session)
    return sessions

def run_AdaBoostR2(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    setup_windows_linux_pathing()
    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

    #TODO It currently fails in STEP_1, try fix
    #TODO Then also remember to check to see how it works with multiple people rather than just one

    # Possible TODO: Should we fully yeet batchdata and add batch_size and epochs into fit function instead? 
    # We could perhaps replicate shuffle_buffer as just a manual shuffling around a moving point? Then we would still have that hyperparameter



    # TRAINING
    sessions = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sliced_X, sliced_Y = flatten_split_sessions(sessions)
    
    #base_estimator = cnn.Model_CNN(find_datashape(config, hyplist, hyperparameter_dict), config, hyplist, hyperparameter_dict)
    ds = find_datashape(config, hyplist, hyperparameter_dict)
    create_base_estimator_fn = lambda: cnn.Model_CNN(ds, config, hyplist, hyperparameter_dict)

    len_source = (len(sliced_X) // 3) * 2 # TODO: For now, 66% of data is source, rest is target
    ada_model = AdaBoostR2(create_base_estimator_fn, [len_source, len(sliced_X) - len_source], 2)
    ada_model.fit(sliced_X, sliced_Y)

    del sliced_X, sliced_Y, sessions # Remove from memory

    # TESTING
    sessions = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sliced_X, sliced_Y = flatten_split_sessions(sessions)
    predictions = ada_model.predict(sliced_X)
    return ada_model.evaluate(predictions, sliced_Y)

def run_TwoStageTrAdaBoost(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    setup_windows_linux_pathing()
    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

#    source_sessions = [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in train_ppl_file_iter]
 #   target_sessions = [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in test_ppl_file_iter]


    #for idx, person in enumerate(train_ppl_file_iter):
    #    print(f"PERSON {idx + 1} of {config.train_ppl_amount}")
    #    sessions = [data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict) for path, sheet in person]
    #   model.fit(idx, sessions)
    # TODO dont quite know how we're going to do train, test, validation splits right now

do_param_optimization = False

config = configuration()
hyplist = hyperparameter_list()

if do_param_optimization: 
    partial_objective = partial(objective, config, hyplist)
    # This is basically function currying. Defines our objective function with the config_dict parameter already present
    opt.perform_hyperopt(partial_objective, hyplist.space(), 100)

else: 
    objective(config, hyplist, hyplist.best_arguments())