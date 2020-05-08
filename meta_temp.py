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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from TwoStageTrAdaBoost import TwoStageTrAdaBoostR2
from plotting_experiments import plotstuff, make_simple_comparison_plot, weights_across_time
from Fixed_TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 as ExoAda
from config_classes import hyperparameter_list, configuration
import optimizer_component as opt
import data_manager_component as data
import cnn_component as cnn
from ensemble import Model_Ensemble_CNN

def objective(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    try:
        setup_windows_linux_pathing()
        #loss = run_plotting_experiments(config, hyplist, hyperparameter_dict)
        loss = run_Baseline6(config, hyplist, hyperparameter_dict)
        #loss = run_cnn(config, hyplist, hyperparameter_dict)
        #loss = run_ensemble(config, hyplist, hyperparameter_dict)
        #loss = run_AdaBoostR2(config, hyplist, hyperparameter_dict)
        #loss = run_wrapper(config, hyplist, hyperparameter_dict)
        return { "loss": loss, 
                 "status": STATUS_OK }
    except Exception as e:
        print(str(e))
        return { "status": STATUS_FAIL,
                 "exception": str(e) }

def run_wrapper(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    #Create and wrap base estimator
    create_base_estimator_fn = lambda: cnn.compile_model_cnn(find_datashape(config, hyplist, hyperparameter_dict), config, hyplist, hyperparameter_dict)
    wrapped_estimator = KerasRegressor(create_base_estimator_fn)
    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

    #Train
    sessions_source = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions_novel_person = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions_target = sessions_novel_person[0:-1] # Leave the last session for the test set
    
    sliced_X_train_source, sliced_Y_train_source = flatten_split_sessions(sessions_source)
    sliced_X_train_tar, sliced_Y_train_tar = flatten_split_sessions(sessions_target)
    sliced_X_train, sliced_Y_train = [], []
    sliced_X_train.extend(sliced_X_train_source)
    sliced_X_train.extend(sliced_X_train_tar)
    sliced_Y_train.extend(sliced_Y_train_source)
    sliced_Y_train.extend(sliced_Y_train_tar)
    sliced_X_train = np.array(sliced_X_train)
    sliced_Y_train = np.array(sliced_Y_train)
    sliced_Y_train = np.concatenate(sliced_Y_train, axis=0)

    regressor = TwoStageTrAdaBoostR2(wrapped_estimator, [len(sliced_X_train_source), len(sliced_X_train_tar)], 2, 2, 2) # TODO: 2,2,2 are temp values
    regressor.fit(sliced_X_train, sliced_Y_train)

    #Test
    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions_novel_person[-1:]) # Test only on the last session from the target person
    sliced_Y_test = np.concatenate(sliced_Y_test, axis=0)
    prediction = regressor.predict(sliced_X_test)

    # Plot groundtruth vs prediction
    make_simple_comparison_plot(sliced_Y_test, "target_test", prediction, "TwoStageTrAdaBoostR2", "x", "y", "Two-stage Transfer Learning Boosted Decision Tree Regression")
    return mean_absolute_error(sliced_Y_test, prediction)

def fit_cnn_with_person(idx, length, person, model):
    print(f"DATASET {idx} of {length}")
    model.fit(person.train, person.val, person.train_slices, person.val_slices)

def setup_windows_linux_pathing():
    gitdir = "P10-ExoskeletonTransferLearning"
    if(exists(gitdir)): chdir(gitdir) # Change dir unless we're already inside it. Necessary for linux v windows execution

def run_cnn(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
    # TODO Need to fix this one to work with multiple people with multiple sessions, such that I dont just assume that one person = 1 sheet

    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()
    #sessions_source = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    #sessions_novel_person = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    #sessions_target = sessions_novel_person[0:-1] # Leave the last session for the test set
    # TODO should probably change such that we test on only one sheet of the last person, to achieve parity with the other baselines

    model = cnn.Model_CNN(find_datashape(config, hyplist, hyperparameter_dict), config, hyplist, hyperparameter_dict)
    for person in train_ppl_file_iter:
        for (path, sheet) in person:
            session = data.process_sheet(path, sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
            model.fit(session.train, session.val, session.train_slices, session.val_slices)
    
    sessions_novel_person = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    session_train = sessions_novel_person[0:-1]
    session_test = sessions_novel_person[-1:]
    for session in session_train:
        model.fit(session.train, session.val, session.train_slices, session.val_slices)
    
    loss_lst = []
    for session in session_test:
        loss_lst.append(model.evaluate(session.test))

    #loss_lst = []
    #for person in test_ppl_file_iter:
    #    for (path, sheet) in person:
    #        session = data.process_sheet(path, sheet, config.cnn_testsplit, config, hyplist, hyperparameter_dict)
    #        loss_lst.append(model.evaluate(session.test))

    del model # Remove all references from the model, such that the garbage collector claims it
    clear_session() # Clear the keras backend dataflow graph, as to not fill up memory
    return np.mean(loss_lst)

def find_datashape(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # We load the first sheet as a test-run to see which datashape it ends up with
    first_path, first_sheet = config.dataset_file_paths[0], config.dataset_sheet_titles[0]
    person = data.process_sheet(first_path, first_sheet, config.cnn_datasplit, config, hyplist, hyperparameter_dict)
    return person.datashape

def run_ensemble(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict): 
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
    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()
    # Possible TODO: Should we fully yeet batchdata and add batch_size and epochs into fit function instead? 
    # We could perhaps replicate shuffle_buffer as just a manual shuffling around a moving point? Then we would still have that hyperparameter



    # TRAINING
    sessions = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sliced_X, sliced_Y = flatten_split_sessions(sessions)
    
    #base_estimator = cnn.Model_CNN(find_datashape(config, hyplist, hyperparameter_dict), config, hyplist, hyperparameter_dict)
    ds = find_datashape(config, hyplist, hyperparameter_dict)
    create_base_estimator_fn = lambda: cnn.Model_CNN(ds, config, hyplist, hyperparameter_dict)

    len_source = (len(sliced_X) // 3) * 2 # TODO: For now, 66% of data is source, rest is target
    ada_model = AdaBoostR2(create_base_estimator_fn, [len_source, len(sliced_X) - len_source])
    ada_model.fit(sliced_X, sliced_Y)

    del sliced_X, sliced_Y, sessions # Remove from memory

    # TESTING
    sessions = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sliced_X, sliced_Y = flatten_split_sessions(sessions)
    predictions = ada_model.predict(sliced_X)
    return ada_model.evaluate(predictions, sliced_Y)

def run_Baseline1(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # CNN on C only
    # Train set: 4/5th of person C
    # Test set: 1/5th of person C
    _, test_ppl_file_iter = config.get_people_iterators()

    sessions = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sliced_X_train, sliced_Y_train = flatten_split_sessions(sessions[:4])
    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions[4:])

    ds = find_datashape(config, hyplist, hyperparameter_dict)
    model = cnn.Model_CNN(ds, config, hyplist, hyperparameter_dict)
    model.fit_ada(sliced_X_train, sliced_Y_train) ### TODO: INCLUDE BATCHSIZE EPOCHS AND SHUFFLEBUFFLE
    return model.evaluate_nonbatched(sliced_X_test, sliced_Y_test)

def run_Baseline2(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # CNN on all data
    # Train set: person A, B and 4/5ths of C
    # Test set: 1/5th of person C

    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

    sessions = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions.extend(unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict))
    sliced_X_train, sliced_Y_train = flatten_split_sessions(sessions[:-1])
    
    ds = find_datashape(config, hyplist, hyperparameter_dict)
    model = cnn.Model_CNN(ds, config, hyplist, hyperparameter_dict)
    model.fit_ada(sliced_X_train, sliced_Y_train) ### TODO: INCLUDE BATCHSIZE EPOCHS AND SHUFFLEBUFFLE

    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions[-1:])
    return model.evaluate_nonbatched(sliced_X_test, sliced_Y_test)

def run_Baseline3(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # NOTE: IS THIS ONE EVEN NEEDED?
    # Ensemble model of CNN
    # Train set: CNN A = person A, CNN B = person B and CNN C = 4/5ths of C
    # Test set: 1/5th of person C

    return 0

def run_Baseline4(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # NOTE: IS THIS ONE EVEN NEEDED?
    # CNN with layer fine-tuning
    # Train set: 9/10th of person A, 9/10th of person B -> Then fine tune last layers on 4/5ths of person C
    # Test set: 1/10th of person A, 1/10th of person B -> After fine tuning of the last layers, test on 1/5ths of person C

    return 0

def make_2d_array(dataset):
    # Reshape 3D-array (slices x observations x samples) into 2D-array (slices x samples-samples-samples-samples-...)
    slices, observations, samples = dataset.shape
    return dataset.reshape((slices, observations*samples))

def run_Baseline5(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # Two-Stage AdaBoost.R2 out-of-the-box (using regression decision trees)
    # Train set: Source is person A + person B. Target is 4/5th of person C.
    # Test set: 1/5th of person C

    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

    sessions_source = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions_novel_person = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions_target = sessions_novel_person[:-1] # Leave the last session for the test set
    sessions_test = sessions_novel_person[-1:]

    sliced_X_source, sliced_Y_source = flatten_split_sessions(sessions_source)
    sliced_X_target, sliced_Y_target = flatten_split_sessions(sessions_target)
    sliced_X_train, sliced_Y_train = [], []
    sliced_X_train.extend(sliced_X_source)
    sliced_X_train.extend(sliced_X_target)
    sliced_Y_train.extend(sliced_Y_source)
    sliced_Y_train.extend(sliced_Y_target)
    sliced_X_train = np.array(sliced_X_train)
    sliced_Y_train = np.array(sliced_Y_train)
    sliced_Y_train = np.concatenate(sliced_Y_train, axis=0) # Flatten inner lists that contain one element each
    sliced_X_train = make_2d_array(sliced_X_train) # Reshape to work with our DecisionTreeRegressor

    # Create default tradaboost estimator
    regressor = TwoStageTrAdaBoostR2(sample_size=[len(sliced_X_source), len(sliced_X_target)], n_estimators=2, steps=2, fold=2) # TODO: 2,2,2 are temp values
    regressor.fit(sliced_X_train, sliced_Y_train)
    # TODO PUT ESTIMATORS, STEPS AND FOLDS INTO SOME CLASS WHERE HYPEROPT CAN OPTIMIZE IT? Nope, its not feasible. 


    # Evaluate
    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions_test) # Test only on the last session from the target person
    sliced_Y_test = np.concatenate(sliced_Y_test, axis=0) # Flatten inner lists that contain one element each
    sliced_X_test = make_2d_array(sliced_X_test) # Reshape to work with our DecisionTreeRegressor
    prediction = regressor.predict(sliced_X_test)
    return mean_absolute_error(sliced_Y_test, prediction)

def run_Baseline6(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # Exo-Ada
    # Train set: Source is person A + person B. Target is 4/5th of person C.
    # Test set: 1/5th of person C
    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

    sessions_source = unpack_sessions(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions_novel_person = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    sessions_target = sessions_novel_person[:-1] # Leave the last session for the test set
    sessions_test = sessions_novel_person[-1:]

    sliced_X_source_A, sliced_Y_source_A = flatten_split_sessions(sessions_source[:5])
    sliced_X_source_B, sliced_Y_source_B = flatten_split_sessions(sessions_source[5:])
    sliced_X_target_C, sliced_Y_target_C = flatten_split_sessions(sessions_target)

    sliced_X_train, sliced_Y_train = [], []
    for lst in [sliced_X_source_A, sliced_X_source_B, sliced_X_target_C] : sliced_X_train.extend(lst)
    for lst in [sliced_Y_source_A, sliced_Y_source_B, sliced_Y_target_C] : sliced_Y_train.extend(lst)
    sliced_X_train = np.array(sliced_X_train)
    sliced_Y_train = np.array(sliced_Y_train)
    sliced_Y_train = np.concatenate(sliced_Y_train, axis=0) # Flatten inner lists that contain one element each

    # Create Exo-Ada
    ds = find_datashape(config, hyplist, hyperparameter_dict)
    create_base_estimator_fn = lambda: cnn.Model_CNN(ds, config, hyplist, hyperparameter_dict)
    regressor = ExoAda(create_base_estimator_fn, sample_size=[len(sliced_X_source_A) + len(sliced_X_source_B), len(sliced_X_target_C)], n_estimators=5, steps=5, fold=2) # TODO: 2,2,2 are temp values
    
    # Initializing weights such that each dataset has a percentage to 1 / n_samples
    sample_weights = np.empty(len(sliced_X_train), dtype=np.float64)
    len_A, len_B, len_C = len(sliced_X_source_A), len(sliced_X_source_B), len(sliced_X_target_C)
    weight_per_dataset = 1. / 3
    sample_weights[:len_A] = weight_per_dataset / len_A
    sample_weights[len_A:len_A+len_B] = weight_per_dataset / len_B
    sample_weights[len_A+len_B:len_A+len_B+len_C] = weight_per_dataset / len_C

    regressor.fit(sliced_X_train, sliced_Y_train, sample_weights)

    # Plot sample_weights for the datasets across time
    weights_across_time(regressor.sample_weights_, len(sliced_X_source_A), len(sliced_X_source_B), len(sliced_X_target_C))

    # Evaluate
    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions_test) # Test only on the last session from the target person
    sliced_Y_test = np.concatenate(sliced_Y_test, axis=0) # Flatten inner lists that contain one element each
    prediction = regressor.predict(sliced_X_test)
    return mean_absolute_error(sliced_Y_test, prediction)

def run_plotting_experiments(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    plotstuff(config, hyplist, hyperparameter_dict, flatten_split_sessions)
    return 0

do_param_optimization = False

config = configuration()
hyplist = hyperparameter_list()

if do_param_optimization: 
    partial_objective = partial(objective, config, hyplist)
    # This is basically function currying. Defines our objective function with the config_dict parameter already present
    opt.perform_hyperopt(partial_objective, hyplist.space(), 100)

else: 
    objective(config, hyplist, hyplist.best_arguments())