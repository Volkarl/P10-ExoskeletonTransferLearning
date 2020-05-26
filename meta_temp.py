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
from plotting import plot_dataset_comparison, make_simple_comparison_plot, weights_across_time, stacked_histogram
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
        
        #loss_lst_4 = []
        #loss_lst_3 = []
        #loss_lst_2 = []
        #loss_lst_1 = []
        #for _ in range(3): 
        #    loss_lst_4.append(run_Baseline4(config, hyplist, hyperparameter_dict, 4))
        #for _ in range(3): 
        #    loss_lst_3.append(run_Baseline4(config, hyplist, hyperparameter_dict, 3))
        #for _ in range(3): 
        #    loss_lst_2.append(run_Baseline4(config, hyplist, hyperparameter_dict, 2))
        #for _ in range(3): 
        #    loss_lst_1.append(run_Baseline4(config, hyplist, hyperparameter_dict, 1))
        #print(f"Loss_Lst_4 {loss_lst_4}, Mean {np.mean(loss_lst_4)}")
        #print(f"Loss_Lst_3 {loss_lst_3}, Mean {np.mean(loss_lst_3)}")
        #print(f"Loss_Lst_2 {loss_lst_2}, Mean {np.mean(loss_lst_2)}")
        #print(f"Loss_Lst_1 {loss_lst_1}, Mean {np.mean(loss_lst_1)}")
        
        print(loss)
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
    sessions_novel_person = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict) # TODO UNPACK LAST FIFTH OF THIS ONE WITHOUT SHUFFLING
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

def flatten_split_sessions(sessions, keep_dataset_percent = 1.0):
    sliced_X, sliced_Y = [], []
    for session in sessions: # Flatten the outer lists
        stop_idx = int(len(session.x) * keep_dataset_percent)
        sliced_X.extend(session.x[:stop_idx])
        sliced_Y.extend(session.y[:stop_idx])
    return np.array(sliced_X), np.array(sliced_Y) # make into numpy arrays, such that we have a shape property

def unpack_sessions(files, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, allow_shuffle):
    return [data.process_sheet(sheet, config, hyplist, hyperparameter_dict, allow_shuffle) for sheet in files]

def run_AdaBoostR2(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()

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
    sessions = unpack_sessions(test_ppl_file_iter, config, hyplist, hyperparameter_dict, False)
    sliced_X, sliced_Y = flatten_split_sessions(sessions)
    predictions = ada_model.predict(sliced_X)
    return ada_model.evaluate(predictions, sliced_Y)

def run_Baseline1(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, test_sheets = 4):
    # CNN on C only
    # Train set: 4/5th of person C
    # Test set: 1/5th of person C
    people = config.get_people_iterator()
    train_files = people[2][:test_sheets]
    test_files = people[2][4:]

    train_sessions = unpack_sessions(train_files, config, hyplist, hyperparameter_dict, True)
    sliced_X_train, sliced_Y_train = flatten_split_sessions(train_sessions)
    test_sessions = unpack_sessions(test_files, config, hyplist, hyperparameter_dict, False)
    sliced_X_test, sliced_Y_test = flatten_split_sessions(test_sessions)

    model = cnn.Model_CNN(train_sessions[0].datashape, config, hyplist, hyperparameter_dict)
    model.fit_ada(sliced_X_train, sliced_Y_train)
    return model.evaluate_nonbatched(sliced_X_test, sliced_Y_test)

def run_Baseline2(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, test_sheets = 4):
    # CNN on all data
    # Train set: person A, B and 4/5ths of C
    # Test set: 1/5th of person C

    people = config.get_people_iterator()
    A_files = people[0][:5]
    B_files = people[1][:5]
    C_files = people[2][:test_sheets] #[2][:4]
    test_files = people[2][4:]

    sessions_A = unpack_sessions(A_files, config, hyplist, hyperparameter_dict, True)
    sessions_B = unpack_sessions(B_files, config, hyplist, hyperparameter_dict, True)
    sessions_C = unpack_sessions(C_files, config, hyplist, hyperparameter_dict, True)
    sliced_X_A, sliced_Y_A = flatten_split_sessions(sessions_A)
    sliced_X_B, sliced_Y_B = flatten_split_sessions(sessions_B, 0.6)
    sliced_X_C, sliced_Y_C = flatten_split_sessions(sessions_C)

    sliced_X_train, sliced_Y_train = [], []
    for lst in [sliced_X_A, sliced_X_B, sliced_X_C] : sliced_X_train.extend(lst)
    for lst in [sliced_Y_A, sliced_Y_B, sliced_Y_C] : sliced_Y_train.extend(lst)
    sliced_X_train, sliced_Y_train = np.array(sliced_X_train), np.array(sliced_Y_train)

    model = cnn.Model_CNN(sessions_A[0].datashape, config, hyplist, hyperparameter_dict)
    model.fit_ada(sliced_X_train, sliced_Y_train)

    test_sessions = unpack_sessions(test_files, config, hyplist, hyperparameter_dict, False)
    sliced_X_test, sliced_Y_test = flatten_split_sessions(test_sessions)
    return model.evaluate_nonbatched(sliced_X_test, sliced_Y_test)

def run_Baseline3(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # NOTE: Not needed
    # CNN with layer fine-tuning
    # Train set: 9/10th of person A, 9/10th of person B -> Then fine tune last layers on 4/5ths of person C
    # Test set: 1/10th of person A, 1/10th of person B -> After fine tuning of the last layers, test on 1/5ths of person C

    return 0

def run_Baseline4(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, test_sheets = 4):
    # Ensemble model of CNN
    # Train set: CNN A = person A, CNN B = person B and CNN C = 4/5ths of C
    # Test set: 1/5th of person C

    people = config.get_people_iterator()
    A_files = people[0][:5]
    B_files = people[1][:5]
    C_files = people[2][:test_sheets]
    test_files = people[2][4:]

    sessions_A = unpack_sessions(A_files, config, hyplist, hyperparameter_dict, True)
    sessions_B = unpack_sessions(B_files, config, hyplist, hyperparameter_dict, True)
    sessions_C = unpack_sessions(C_files, config, hyplist, hyperparameter_dict, True)
    sliced_X_A, sliced_Y_A = flatten_split_sessions(sessions_A)
    sliced_X_B, sliced_Y_B = flatten_split_sessions(sessions_B, 0.6)
    sliced_X_C, sliced_Y_C = flatten_split_sessions(sessions_C)

    model_A = cnn.Model_CNN(sessions_A[0].datashape, config, hyplist, hyperparameter_dict)
    model_B = cnn.Model_CNN(sessions_B[0].datashape, config, hyplist, hyperparameter_dict)
    model_C = cnn.Model_CNN(sessions_C[0].datashape, config, hyplist, hyperparameter_dict)
    
    model_A.fit_ada(sliced_X_A, sliced_Y_A)
    model_B.fit_ada(sliced_X_B, sliced_Y_B)
    model_C.fit_ada(sliced_X_C, sliced_Y_C)

    ensemble = Model_Ensemble_CNN([model_A, model_B, model_C])

    test_sessions = unpack_sessions(test_files, config, hyplist, hyperparameter_dict, False)
    sliced_X_test, sliced_Y_test = flatten_split_sessions(test_sessions)

    prediction = ensemble.predict(sliced_X_test, sliced_Y_test, False)
    make_simple_comparison_plot(sliced_Y_test, "Person C Test Set", prediction, "Ensemble CNN", "x", "y", "Ensemble CNN", False, "ens")
    return mean_absolute_error(sliced_Y_test, prediction)

def make_2d_array(dataset):
    # Reshape 3D-array (slices x observations x samples) into 2D-array (slices x samples-samples-samples-samples-...)
    slices, observations, samples = dataset.shape
    return dataset.reshape((slices, observations*samples))

def run_Baseline5(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # Two-Stage AdaBoost.R2 out-of-the-box (using regression decision trees)
    # Train set: Source is person A + person B. Target is 4/5th of person C.
    # Test set: 1/5th of person C
    people = config.get_people_iterator()
    source_A_files = people[0][:5]
    source_B_files = people[1][:5]
    target_C_files = people[2][:4]
    test_files = people[2][4:]

    sessions_A_source = unpack_sessions(source_A_files, config, hyplist, hyperparameter_dict, True)
    sessions_B_source = unpack_sessions(source_B_files, config, hyplist, hyperparameter_dict, True)
    sessions_C_target = unpack_sessions(target_C_files, config, hyplist, hyperparameter_dict, True)
    sliced_X_source_A, sliced_Y_source_A = flatten_split_sessions(sessions_A_source)
    sliced_X_source_B, sliced_Y_source_B = flatten_split_sessions(sessions_B_source, 0.6)
    sliced_X_target_C, sliced_Y_target_C = flatten_split_sessions(sessions_C_target)

    sliced_X_train, sliced_Y_train = [], []
    for lst in [sliced_X_source_A, sliced_X_source_B, sliced_X_target_C] : sliced_X_train.extend(lst)
    for lst in [sliced_Y_source_A, sliced_Y_source_B, sliced_Y_target_C] : sliced_Y_train.extend(lst)
    sliced_X_train = np.array(sliced_X_train)
    sliced_X_train = make_2d_array(sliced_X_train) # Reshape to work with our DecisionTreeRegressor
    sliced_Y_train = np.array(sliced_Y_train)
    sliced_Y_train = np.concatenate(sliced_Y_train, axis=0) # Flatten inner lists that contain one element each

    # Create default tradaboost estimator
    e, s, f = 2, 7, 2
    print(f"You're about to run with arguments ({e}, {s}, {f}), which equals fitting {(e*s*f) + (e*s) + s} base learners")
    regressor = TwoStageTrAdaBoostR2(sample_size=[len(sliced_X_source_A) + len(sliced_X_source_B), len(sliced_X_target_C)], n_estimators=e, steps=s, fold=f)
    regressor.fit(sliced_X_train, sliced_Y_train)

    # Plot sample_weights for the datasets across time
    weights_across_time(regressor.sample_weights_, len(sliced_X_source_A), len(sliced_X_source_B), len(sliced_X_target_C), True, "baseline5")

    errors, idx, bew, ew = regressor.get_estimator_info()
    print(f"Errors {errors}")
    print(f"Best idx {idx}")
    print(f"Weights of best estimator {bew}")
    stacked_histogram(np.array(ew), np.array(errors), do_savefig=True, savename="baseline5")

    # Evaluate
    sessions_test = unpack_sessions(test_files, config, hyplist, hyperparameter_dict, False)
    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions_test)
    sliced_X_test = make_2d_array(sliced_X_test) # Reshape to work with our DecisionTreeRegressor
    sliced_Y_test = np.concatenate(sliced_Y_test, axis=0) # Flatten inner lists that contain one element each
    prediction = regressor.predict(sliced_X_test)

    make_simple_comparison_plot(sliced_Y_test, "Person C Test Set", prediction, "2-Stage TrAdaBoost", "x", "y", "2-Stage TrAdaBoost", True, "baseline5")
    return mean_absolute_error(sliced_Y_test, prediction)

def run_Baseline6(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    # Exo-Ada
    # Train set: Source is person A + person B. Target is 4/5th of person C.
    # Test set: 1/5th of person C

    people = config.get_people_iterator()
    source_A_files = people[0][:5]
    source_B_files = people[1][:5]
    target_C_files = people[2][:4]
    test_files = people[2][4:]

    sessions_A_source = unpack_sessions(source_A_files, config, hyplist, hyperparameter_dict, True)
    sessions_B_source = unpack_sessions(source_B_files, config, hyplist, hyperparameter_dict, True)
    sessions_C_target = unpack_sessions(target_C_files, config, hyplist, hyperparameter_dict, True)
    sliced_X_source_A, sliced_Y_source_A = flatten_split_sessions(sessions_A_source)
    sliced_X_source_B, sliced_Y_source_B = flatten_split_sessions(sessions_B_source, 0.6)
    sliced_X_target_C, sliced_Y_target_C = flatten_split_sessions(sessions_C_target)

    sliced_X_train, sliced_Y_train = [], []
    for lst in [sliced_X_source_A, sliced_X_source_B, sliced_X_target_C] : sliced_X_train.extend(lst)
    for lst in [sliced_Y_source_A, sliced_Y_source_B, sliced_Y_target_C] : sliced_Y_train.extend(lst)
    sliced_X_train = np.array(sliced_X_train)
    sliced_Y_train = np.array(sliced_Y_train)
    sliced_Y_train = np.concatenate(sliced_Y_train, axis=0) # Flatten inner lists that contain one element each

    # Create Exo-Ada
    e, s, f, ss = 100, 3, 2, 0 #2,2,2,0#2, 10, 4, 0
    print(f"You're about to run with arguments ({e}, {s}, {f}, {ss}), which equals fitting {(e*s*f) + (e*s) + s + (e*ss*f)} base learners")

    #create_base_estimator_fn = lambda: cnn.Model_DTR(4)
    create_base_estimator_fn = lambda: cnn.Model_CNN(sessions_A_source[0].datashape, config, hyplist, hyperparameter_dict)
    regressor = ExoAda(create_base_estimator_fn, sample_size=[len(sliced_X_source_A) + len(sliced_X_source_B), len(sliced_X_target_C)], n_estimators=e, steps=s, fold=f, start_steps=ss)
    
    # Initializing weights such that each dataset has a percentage to 1 / n_samples
    sample_weights = np.empty(len(sliced_X_train), dtype=np.float64)
    len_A, len_B, len_C = len(sliced_X_source_A), len(sliced_X_source_B), len(sliced_X_target_C)
    weight_per_dataset = 1. / 3
    sample_weights[:len_A] = weight_per_dataset / len_A
    sample_weights[len_A:len_A+len_B] = weight_per_dataset / len_B
    sample_weights[len_A+len_B:len_A+len_B+len_C] = weight_per_dataset / len_C

    regressor.fit(sliced_X_train, sliced_Y_train, sample_weights, weight_per_dataset)

    # Plot sample_weights for the datasets across time
    weights_across_time(regressor.sample_weights_, len(sliced_X_source_A), len(sliced_X_source_B), len(sliced_X_target_C), True, "baseline6")

    errors, idx, bew, ew = regressor.get_estimator_info()
    print(f"Errors {errors}")
    print(f"Best idx {idx}")
    print(f"Weights of best estimator {bew}")
    stacked_histogram(np.array(ew), np.array(errors), do_savefig=True, savename="baseline6")

    # Evaluate
    sessions_test = unpack_sessions(test_files, config, hyplist, hyperparameter_dict, False)
    sliced_X_test, sliced_Y_test = flatten_split_sessions(sessions_test) # Test only on the last session from the target person
    sliced_Y_test = np.concatenate(sliced_Y_test, axis=0) # Flatten inner lists that contain one element each
    prediction = regressor.predict(sliced_X_test)
    
    make_simple_comparison_plot(sliced_Y_test, "Person C Test Set", prediction, "Exo-Ada", "x", "y", "Exo-Ada", True, "baseline6")
    return mean_absolute_error(sliced_Y_test, prediction)

def run_plotting_experiments(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    plot_dataset_comparison(config, hyplist, hyperparameter_dict, flatten_split_sessions)
    return 0


do_param_optimization = False

config = configuration()
hyplist = hyperparameter_list()

if do_param_optimization: 
    partial_objective = partial(objective, config, hyplist)
    # This is basically function currying. Defines our objective function with the config_dict parameter already present
    opt.perform_hyperopt(partial_objective, hyplist.space(), 500)

else: 
    objective(config, hyplist, hyplist.best_arguments())