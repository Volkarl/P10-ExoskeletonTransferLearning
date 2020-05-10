import pandas as pd
import numpy as np
import tensorflow as tf
from config_classes import hyperparameter_list, configuration

class batched_data:
    def __init__(self, datashape, x, y, slices):
        self.datashape = datashape
        self.x = x
        self.y = y
        self.slices = slices

# Data layout in the xlsx files
columns_data = ['1' ,'2', '3', '4', '5', '6', '7', '8', 'N/A_1', 'N/A_2', 'angle', 'time', 'session']
columns_features_considered = columns_data[:8]
column_ground_truth = columns_data[10]
# Each timestep represents 1 millisecond, 0.001 second. 

def process_sheet_no_slice(sheet_title, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    raw_data = load_dataset(sheet_title, config.granularity)
    _, features, ground_truth = split_data(raw_data, config.granularity, hyperparameter_dict[hyplist.smoothing])
    return features, ground_truth

def process_sheet(sheet_title, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, allow_shuffle = True):
    raw_data = load_dataset(sheet_title, config.granularity)
    indexes, features, ground_truth = split_data(raw_data, config.granularity, hyperparameter_dict[hyplist.smoothing])
    if(hyperparameter_dict[hyplist.use_ref_points]): 
        features = calc_ref_features(features, hyperparameter_dict[hyplist.ref_point1], hyperparameter_dict[hyplist.ref_point2])

    x, y = slice_data(indexes, features, ground_truth, hyperparameter_dict[hyplist.dilation_group][hyplist.past_history], config)
    # Data is now sliced into past_history slices

    sbs = hyperparameter_dict[hyplist.shuffle_buffer_size]
    if sbs != 0 and allow_shuffle: # We allow turning off shuffling for either the test set, or for plotting purposes
        x = shuffle_buffer_manual(x, sbs)
        y = shuffle_buffer_manual(y, sbs)

    return batched_data(x.shape[-2:], x, y, len(x))

def shuffle_buffer_manual(dataset, shuffle_buffer_size):
    buffer = dataset[:shuffle_buffer_size] # fill up buffer
    dataset = dataset[shuffle_buffer_size:] # remove those elements in buffer from the dataset

    shuffled_dataset = []
    for i in range(len(dataset)):
        rand_index = np.random.randint(0, shuffle_buffer_size)
        shuffled_dataset.append(buffer[rand_index])
        buffer[rand_index] = dataset[i]
    shuffled_dataset.extend(buffer)
    return np.array(shuffled_dataset)

def slice_data(indexes, features, ground_truth, past_history, config: configuration):
    ft, sssw, gran = config.future_target, config.step_size_sliding_window, config.granularity
    dataset = features.values
    x, y = multivariate_data(dataset, ground_truth.values, 0, None, past_history, ft, sssw, gran)
    return x, y

def multivariate_data(dataset_features, dataset_ground_truth, start_index, end_index, history_size, target_size, step, granularity):
    # Create array of all sliding windows of the data
    data, labels = [], []
    start_index = start_index + history_size 
    if end_index is None: 
        end_index = len(dataset_features)
    for i in range(start_index, end_index - target_size): # start 100, end 790. 
        indices = range(i-history_size, i, step) # range(0, 100) step size of 1          --- our sliding window
        data.append(dataset_features[indices]) # append new array that contains all values within our sliding window
        labels.append(dataset_ground_truth[i+target_size])
    return np.array(data), np.array(labels)

def calc_ref_features(features, ref_point1, ref_point2):
    relative_features1 = [subtract_refvalue(obs, obs[ref_point1]) for obs in features.values]
    relative_features2 = [subtract_refvalue(obs, obs[ref_point2]) for obs in features.values]
    return pd.DataFrame([relative_features1[i] + relative_features2[i] for i in range(0, len(features))])
    # TODO: Make a new method for this, one which removes the two null reference points, and makes it percentage based instead

def subtract_refvalue(obs, ref_value):
    return [val - ref_value for val in obs]

def load_dataset(sheet_title, granularity):
    sheet_data = pd.read_csv(f"Datasets/{sheet_title}.csv")
    sheet_data.columns = columns_data
    return sheet_data

def split_data(raw_data, granularity, smoothing):
    indexes = range(0, len(raw_data), 1)[::granularity] # Each timestep is a millisecond
    features = raw_data[columns_features_considered][::granularity].ewm(span=smoothing).mean() 
    # NOTE: Be aware that this type of smoothing may not be applicable in an online application (where we dont have values before and after)
    ground_truth = pd.DataFrame(raw_data[column_ground_truth][::granularity]).ewm(span=smoothing).mean()
    return indexes, features, ground_truth
