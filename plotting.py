import matplotlib.pyplot as plt
import numpy as np
import pickle

from data_manager_component import process_sheet_no_slice
from config_classes import hyperparameter_list, configuration

def read_saved_data(savename):
    comp_data = pickle.load(open(f"comp_data_{savename}.p", "rb"))
    hist_data = pickle.load(open(f"hist_data_{savename}.p", "rb"))
    weights_data = pickle.load(open(f"weights_data_{savename}.p", "rb"))
    print("Stop")

    #weights_across_time(weights_data, 7624, 8960, 7373, False, savename)
    
    #pad_lst = lambda lst, n: [el for el in lst for _ in range(n)] 
    #test, exo = pad_lst(comp_data["Person C Test Set"], 30)[35000:], pad_lst(comp_data["Exo-Ada"], 30)[35000:]
    #make_simple_comparison_plot(test, "Test Session", exo, "Exo-Ada", "Time (1 ms)", "Radians", "Elbow Angle Estimation", False, savename)

# read_saved_data("baseline6")

def make_simple_comparison_plot(y1, y1_name, y2, y2_name, x_axis_name, y_axis_name, title = None, do_savefig = False, savename = None):
    pickle.dump( { y1_name : y1, y2_name : y2 } , open(f"comp_data_{savename}.p", "wb"))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(y1, c="b", label=y1_name, linewidth=0.5)
    ax.plot(y2, c="r", label=y2_name, linewidth=1)
    #ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    if title != None: ax.set_title(title)
    ax.legend()
    #ax.legend(loc="upper right")

    if do_savefig: fig.savefig(f'comp_{savename}.png')
    else: plt.show()

def plot_multiple_comparisons(y_lists, labels, colors, xlabel, ylabel, title):
    plt.figure()

    for i in range(len(y_lists)):
        plt.plot(y_lists[i], label=labels[i], color=colors[i], linewidth=1)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def stacked_histogram(stacked_hist_values, errors, colors = ['b','g', 'r', 'c', 'm', 'y', 'k', 'lime', 'purple', 'crimson'], do_savefig = False, savename = None):
    pickle.dump( { "stacked_hist_values" : stacked_hist_values, "errors" : errors } , open(f"hist_data_{savename}.p", "wb"))
    
    cols = []
    for i in range(stacked_hist_values.shape[1]):
        cols.append(colors[i % len(colors)])
    colors = cols
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
    pickle.dump(sample_weights_across_steps, open(f"weights_data_{savename}.p", "wb"))
    
    y_A, y_B, y_C = [], [], []
    for sample_weights in sample_weights_across_steps:
        sw_A, sw_B, sw_C = sample_weights[:len_A], sample_weights[len_A:len_A+len_B], sample_weights[len_A+len_B:]
        y_A.append(np.sum(sw_A))
        y_B.append(np.sum(sw_B))
        y_C.append(np.sum(sw_C))

    #fig = plt.figure()
    #ax = fig.add_subplot(2,2,1)
    #ax.plot(y_A, c="b", label=f"Source A", linewidth=1)
    #ax.plot(y_B, c="r", label=f"Source B", linewidth=1)
    #ax.plot(y_C, c="y", label=f"Target C", linewidth=1)
    #ax.set_xlabel("Boosting Step")
    #ax.set_ylabel("Total Dataset Weight")
    #ax.set_title("Sample Weight Distributions")
    #ax.legend()

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
    config.granularity = 1

    people = config.get_people_iterator()
    A_files = people[0][:1]
    B_files = people[1][:1]
    C_files = people[2][:1]

    a = unpack_sessions_no_slice(A_files, config, hyplist, hyperparameter_dict)
    sliced_X_A, sliced_Y_A = flatten_split_sessions(a)
    sliced_X_B, sliced_Y_B = flatten_split_sessions(unpack_sessions_no_slice(B_files, config, hyplist, hyperparameter_dict))
    sliced_X_C, sliced_Y_C = flatten_split_sessions(unpack_sessions_no_slice(C_files, config, hyplist, hyperparameter_dict))

    X_1, Y_1 = sliced_X_A[:15000], sliced_Y_A[:15000]
    X_2, Y_2 = sliced_X_B[:15000], sliced_Y_B[:15000]
    X_3, Y_3 = sliced_X_C[:15000], sliced_Y_C[:15000]

    #pad_lst = lambda lst, n: [el for el in lst for _ in range(n)] 
    #X_1, Y_1 = pad_lst(X_1, 30), pad_lst(Y_1, 30)
    #X_2, Y_2 = pad_lst(X_2, 30), pad_lst(Y_2, 30)
    #X_3, Y_3 = pad_lst(X_3, 30), pad_lst(Y_3, 30)

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
    ax1.set_xlabel("Time (1 ms)")
    ax1.set_ylabel("Radians")
    ax1.set_title("Elbow Angle")
    
    fig.legend(loc="lower center", ncol=3)
    
    ax2.plot(X_1, c="b", label="Person A", linewidth=1)
    ax2.plot(X_2, c="y", label="Person B", linewidth=1)
    ax2.plot(X_3, c="r", label="Person C", linewidth=1)
    ax2.set_xlabel("Time (1 ms)")
    ax2.set_ylabel("Value")
    ax2.set_title("Sensor Value Means")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    plt.show()
    print("end")

def plot_ablation_study(): # is run with 5/10/3
    x = [ 1, 2, 3, 4 ]
    # labels = [ "Exo-Ada", "Exo-Ada w/o BaseCNN", "Exo-Ada w/o Multi-Domain", "2-Stage TrAdaBoost" ] 
    labels = [ "Exo-Ada", "E-Regr", "E-Single", "2-Stage" ] 
    b6 = [ 0.14948451317179137, 0.1568888250433344, 0.15388176396547415, 0.1481531452799892 ]
    text = [ 0.149, 0.157, 0.154, 0.148 ]

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.bar(x, b6, color = ["m", "r", "b", "c"], alpha = 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE")
    ax.set_title("Ablation Study")

    for index, value in enumerate(b6):
        plt.text(x=index + 1, y=value - 0.02, s=text[index], ha='center') #Change font size with: fontdict=dict(fontsize=20)

    plt.show()
    print("end")

# plot_ablation_study()

def plot_estimator_reduc_accuracy_comparison():
    x = [ 5, 10, 15, 20, 25 ]
    b5 = [ 0.1555330507508396, 0.14826824963938826, 0.14427301085729652, 0.14062949540356182, 0.15158738720870604 ]
    b6 = [ 0.14948451317179137, 0.15511536357815509, 0.1499810981641296, 0.14686213174025406, 0.1522692119494696 ]

    plt.figure()
    plt.plot(x, b5, "cs--", markerfacecolor='none', label="2-Stage TrAdaBoost", linewidth=1)
    plt.plot(x, b6, "m*-", markerfacecolor='none', label="Exo-Ada", linewidth=1)
    plt.xlabel("N Estimators")
    plt.ylabel("MAE")
    plt.xticks(x)
    plt.title("Error Over Amount of Estimators")
    plt.legend(frameon=False, markerfirst=False)

    plt.show()
    print("end")

# plot_estimator_reduc_accuracy_comparison()

def plot_target_accuracy_comparison():
    x = [1, 2, 3, 4]
    b1 = [ 0.46204124166973287, 0.5145204290302875, 0.348631253080425, 0.16115500314396838 ]
    b2 = [ 0.18753941475931857, 0.17503625891662256, 0.17703442328322402, 0.16213747466457573 ]
    b4 = [ 0.23851535677450605, 0.2701332927192523, 0.2126576669858217, 0.18190372315629796 ]
    b5 = [ 0.323813768059448, 0.15258097422374503, 0.15371555521185734, 0.15158738720870604 ]
    b6 = [ 0.18876090789241787, 0.1740548223383503, 0.17671658026879764, 0.14948451317179137 ]

    plt.figure()
    plt.plot(x, b6, "m*-", markerfacecolor='none', label="Exo-Ada", linewidth=1.5)
    plt.plot(x, b5, "cs--", markerfacecolor='none', label="2-Stage TrAdaBoost", linewidth=1.5)
    plt.plot(x, b4, "r>-.", markerfacecolor='none', label="Ensemble", linewidth=1.5)
    plt.plot(x, b2, "g+--", markerfacecolor='none', label="CNN_Big", linewidth=1.5)
    plt.plot(x, b1, "bx:", markerfacecolor='none', label="CNN_Small", linewidth=1.5)
    plt.xlabel("Person C Sessions")
    plt.ylabel("MAE")
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.title("Robustness to Size of the Target Dataset")
    plt.legend(frameon=False, markerfirst=False)

    plt.show()
    print("end")

# plot_target_accuracy_comparison()
