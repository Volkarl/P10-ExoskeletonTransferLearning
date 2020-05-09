#import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (20,10)
from config_classes import hyperparameter_list, configuration
import numpy as np
from data_manager_component import process_sheet_no_slice

def make_simple_comparison_plot(y1, y1_name, y2, y2_name, x_axis_name, y_axis_name, title):
    plt.figure()
    plt.plot(y1, c="b", label=y1_name, linewidth=0.5)
    plt.plot(y2, c="r", label=y2_name, linewidth=2)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.legend()
    plt.show()

def stacked_histogram(stacked_hist_values, errors, colors = ['b','g', 'r', 'c', 'm', 'y', 'k', 'w', 'purple', 'crimson']):
    colors = colors[:stacked_hist_values.shape[1]] # Shortens the color array, if it's longer than how many values we have to stack for each column of our histogram
    X = range(stacked_hist_values.shape[0])
    accuracies = [1 - er for er in errors]

    # plt.hist(X, stacked_hist_values, color=colors, density=True, histtype='bar', stacked=True) # , normed=1, alpha=0.5, 
    # We use the manual way of stacking bar plots to create a histogram, because I couldn't get the conventional way working (see previous line)
    bottom_vals = np.zeros(stacked_hist_values.shape[0])
    w_totals = np.sum(stacked_hist_values, axis = 1) # Find the sum for each of the ensemble models

    for stack in range(stacked_hist_values.shape[1]):
        weights = stacked_hist_values[:,stack]
        weights = [(w / w_totals[idx]) * accuracies[idx] for idx, w in enumerate(weights)] # Calculate the percentage of how much each estimator/weaklearner contributes to the accuracy
        plt.bar(X, weights, color = colors[stack], alpha = 0.5, bottom = bottom_vals)
        for idx, w in enumerate(weights):
            bottom_vals[idx] += w

    plt.xlabel("Ensemble Models")
    plt.ylabel("WeakLearner Weight Distribution") # So basically this is how much each weaklearner contributes to the accuracy
    plt.show()
    print("stop")

# ar = np.array([[2,0.5,1.8,0.2],[0.2,2,0.2,1.5],[0.5,2,1.5,2],[2,0.7,0.2,1.5],[1.5,2,0.2,2]])
# er = np.array([0.2,0.22,0.23,0.21,0.24])
# stacked_histogram(ar, er)

def weights_across_time(sample_weights_across_steps, len_A, len_B, len_C):
    y_A, y_B, y_C = [], [], []
    for sample_weights in sample_weights_across_steps:
        sw_A, sw_B, sw_C = sample_weights[:len_A], sample_weights[len_A:len_A+len_B], sample_weights[len_A+len_B:]
        y_A.append(np.sum(sw_A))
        y_B.append(np.sum(sw_B))
        y_C.append(np.sum(sw_C))

    plt.figure()
    plt.plot(y_A, c="b", label=f"Dataset A (samples {len_A}, sum {np.sum(y_A)})", linewidth=2)
    plt.plot(y_B, c="r", label=f"Dataset B (samples {len_B}, sum {np.sum(y_B)})", linewidth=2)
    plt.plot(y_C, c="y", label=f"Dataset C (samples {len_C}, sum {np.sum(y_C)})", linewidth=2)
    plt.xlabel("Boosting Step")
    plt.ylabel("Total Dataset Weight")
    plt.title("Distribution of Sample Weights Over Boosting Iterations")
    plt.legend()
    plt.show()
    print("BREAKPOINT HERE")

def unpack_sessions_no_slice(person_iterator, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    session_features, session_truths = [], []
    for person in person_iterator:
        for path, sheet in person:
            features, truths = process_sheet_no_slice(path, sheet, config, hyplist, hyperparameter_dict)
            session_features.append(features.values)
            session_truths.append(truths.values)
    return session_features, session_truths

def plot_dataset_comparison(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, flatten_split_sessions):
    hyperparameter_dict[hyplist.use_ref_points] = False
    hyperparameter_dict[hyplist.smoothing] = 50

    train_ppl_file_iter, test_ppl_file_iter = config.get_people_iterators()
    session_features_train, session_truths_train = unpack_sessions_no_slice(train_ppl_file_iter, config, hyplist, hyperparameter_dict)
    features_person1, truths_person1 = session_features_train[0], session_truths_train[0]
    features_person2, truths_person2 = session_features_train[5], session_truths_train[5]
    session_features_test, session_truths_test = unpack_sessions_no_slice(test_ppl_file_iter, config, hyplist, hyperparameter_dict)
    features_person3, truths_person3 = session_features_test[0], session_truths_test[0]

    reduce_size = lambda lst, divide: lst[0:(len(lst) // divide)] # Reduce our 60ish second recording to 10
    X_1, Y_1 = reduce_size(features_person1, 4), reduce_size(truths_person1, 4)
    X_2, Y_2 = reduce_size(features_person2, 4), reduce_size(truths_person2, 4)
    X_3, Y_3 = reduce_size(features_person3, 6), reduce_size(truths_person3, 6)

    make_mean = lambda lst: [np.mean(obs) for obs in lst]
    X_1 = make_mean(X_1)
    X_2 = make_mean(X_2)
    X_3 = make_mean(X_3)

    #make_first_sensor = lambda lst: [obs[0] for obs in lst]
    #X_1 = make_first_sensor(X_1)
    #X_2 = make_first_sensor(X_2)
    #X_3 = make_first_sensor(X_3)

    plt.figure()
    plt.plot(X_1, c="b", label="Emil", linewidth=0.5)
    plt.plot(X_2, c="y", label="Jona", linewidth=0.5)
    plt.plot(X_3, c="r", label="Palle", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Sensor Value Means Between People")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Y_1, c="b", label="Emil", linewidth=0.5)
    plt.plot(Y_2, c="y", label="Jona", linewidth=0.5)
    plt.plot(Y_3, c="r", label="Palle", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Angle Values Between People")
    plt.legend()
    plt.show()

    print("end")