import matplotlib.pyplot as plt
from config_classes import hyperparameter_list, configuration
import numpy as np
from data_manager_component import process_sheet_no_slice

def make_simple_comparison_plot(y1, y1_name, y2, y2_name, x_axis_name, y_axis_name, title = None, do_savefig = False, savename = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y1, c="b", label=y1_name, linewidth=0.5)
    ax.plot(y2, c="r", label=y2_name, linewidth=2)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    if title != None: ax.set_title(title)
    ax.legend()
    if do_savefig: fig.savefig(f'comp_{savename}.png')
    else: plt.show()

def stacked_histogram(stacked_hist_values, errors, colors = ['b','g', 'r', 'c', 'm', 'y', 'k', 'lime', 'purple', 'crimson'], do_savefig = False, savename = None):
    colors = colors[:stacked_hist_values.shape[1]] # Shortens the color array, if it's longer than how many values we have to stack for each column of our histogram
    X = range(stacked_hist_values.shape[0])
    accuracies = [1 - er for er in errors]

    # plt.hist(X, stacked_hist_values, color=colors, density=True, histtype='bar', stacked=True) # , normed=1, alpha=0.5, 
    # We use the manual way of stacking bar plots to create a histogram, because I couldn't get the conventional way working (see previous line)
    bottom_vals = np.zeros(stacked_hist_values.shape[0])
    w_totals = np.sum(stacked_hist_values, axis = 1) # Find the sum for each of the ensemble models

    fig = plt.figure()
    #fig.suptitle("Empty figure")
    #ax.set_title("Empty plot")
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Ensemble Models")
    ax.set_ylabel("WeakLearner Weight Distribution") # So basically this is how much each weaklearner contributes to the accuracy

    for stack in range(stacked_hist_values.shape[1]):
        weights = stacked_hist_values[:,stack]
        weights = [(w / w_totals[idx]) * accuracies[idx] for idx, w in enumerate(weights)] # Calculate the percentage of how much each estimator/weaklearner contributes to the accuracy
        ax.bar(X, weights, color = colors[stack], alpha = 0.5, bottom = bottom_vals)
        for idx, w in enumerate(weights):
            bottom_vals[idx] += w
    
    if do_savefig: fig.savefig(f'sh_{savename}.png')
    else: plt.show()
    print("BREAKPOINT HERE")

#ar = np.array([[2,0.5,1.8,0.2],[0.2,2,0.2,1.5],[0.5,2,1.5,2],[2,0.7,0.2,1.5],[1.5,2,0.2,2]])
#er = np.array([0.2,0.22,0.23,0.21,0.24])
#stacked_histogram(ar, er, do_savefig=True, savename="asfsa")

def weights_across_time(sample_weights_across_steps, len_A, len_B, len_C, do_savefig = False, savename = None):
    y_A, y_B, y_C = [], [], []
    for sample_weights in sample_weights_across_steps:
        sw_A, sw_B, sw_C = sample_weights[:len_A], sample_weights[len_A:len_A+len_B], sample_weights[len_A+len_B:]
        y_A.append(np.sum(sw_A))
        y_B.append(np.sum(sw_B))
        y_C.append(np.sum(sw_C))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y_A, c="b", label=f"Source A (samples {len_A}, sum {'%.4f' % np.sum(y_A)})", linewidth=2)
    ax.plot(y_B, c="r", label=f"Source B (samples {len_B}, sum {'%.4f' % np.sum(y_B)})", linewidth=2)
    ax.plot(y_C, c="y", label=f"Target C (samples {len_C}, sum {'%.4f' % np.sum(y_C)})", linewidth=2)
    ax.set_xlabel("Boosting Step")
    ax.set_ylabel("Total Dataset Weight")
    ax.set_title("Distribution of Sample Weights Between Datasets")
    ax.legend()

    if do_savefig: fig.savefig(f'wat_{savename}.png')
    else: plt.show()
    print("BREAKPOINT HERE")

def unpack_sessions_no_slice(files, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    return [process_sheet_no_slice(sheet, config, hyplist, hyperparameter_dict) for sheet in files]

def plot_dataset_comparison(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, flatten_split_sessions):
    hyperparameter_dict[hyplist.use_ref_points] = False

    people = config.get_people_iterator()
    A_files = people[0][:1]
    B_files = people[1][:1]
    C_files = people[2][:1]

    a = unpack_sessions_no_slice(A_files, config, hyplist, hyperparameter_dict)
    sliced_X_A, sliced_Y_A = flatten_split_sessions(a)
    sliced_X_B, sliced_Y_B = flatten_split_sessions(unpack_sessions_no_slice(B_files, config, hyplist, hyperparameter_dict))
    sliced_X_C, sliced_Y_C = flatten_split_sessions(unpack_sessions_no_slice(C_files, config, hyplist, hyperparameter_dict))

    X_1, Y_1 = sliced_X_A[:500], sliced_Y_A[:500]
    X_2, Y_2 = sliced_X_B[:500], sliced_Y_B[:500]
    X_3, Y_3 = sliced_X_C[:500], sliced_Y_C[:500]

    make_mean = lambda lst: [np.mean(obs) for obs in lst]
    X_1 = make_mean(X_1)
    X_2 = make_mean(X_2)
    X_3 = make_mean(X_3)

    #make_first_sensor = lambda lst: [obs[0] for obs in lst]
    #X_1 = make_first_sensor(X_1)
    #X_2 = make_first_sensor(X_2)
    #X_3 = make_first_sensor(X_3)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)

    ax1.plot(Y_1, c="b", label="Person A", linewidth=1)
    ax1.plot(Y_2, c="y", label="Person B", linewidth=1)
    ax1.plot(Y_3, c="r", label="Person C", linewidth=1)
    ax1.set_xlabel("Time (0.01 s)")
    ax1.set_ylabel("Radians")
    ax1.set_title("Elbow Angle")
    
    fig.legend(loc="lower center", ncol=3)
    
    ax2.plot(X_1, c="b", label="Person A", linewidth=1)
    ax2.plot(X_2, c="y", label="Person B", linewidth=1)
    ax2.plot(X_3, c="r", label="Person C", linewidth=1)
    ax2.set_xlabel("Time (0.01 s)")
    ax2.set_ylabel("Value")
    ax2.set_title("Sensor Value Means")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    plt.show()

    print("end")