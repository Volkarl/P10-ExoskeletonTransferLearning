import pandas as pd
import numpy as np
import tensorflow as tf
from config_classes import hyperparameter_list, configuration

class batched_data:
    def __init__(self, datashape, batched_train_data, batched_val_data, batched_test_data, train_slices, val_slices, test_slices,    x_train, y_train, x_val, y_val, x_test, y_test):
        self.datashape = datashape
        self.train = batched_train_data
        self.val = batched_val_data
        self.test = batched_test_data
        self.train_slices = train_slices
        self.val_slices = val_slices
        self.test_slices = test_slices

        self.x_train = x_train # TODO THESE VALUES ARE A TEMPORARY SOLUTION, BECAUSE I DONT KNOW IF I SHOULD REMOVE BATCH_DATA OR NOT
        self.y_train = y_train 
        self.x_val = x_val 
        self.y_val = y_val 
        self.x_test = x_test 
        self.y_test = y_test

# Data layout in the xlsx files
columns_data = ['1' ,'2', '3', '4', '5', '6', '7', '8', 'N/A_1', 'N/A_2', 'angle', 'time', 'session']
columns_features_considered = columns_data[:8]
column_ground_truth = columns_data[10]
# Each timestep represents 1 millisecond, 0.001 second. 

def process_sheet_no_slice(sheet_path, sheet_title, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    raw_data = load_dataset(sheet_path, sheet_title, config.granularity)
    _, features, ground_truth = split_data(raw_data, config.granularity, hyperparameter_dict[hyplist.smoothing])
    return features, ground_truth

def process_sheet(sheet_path, sheet_title, datasplit, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    raw_data = load_dataset(sheet_path, sheet_title, config.granularity)
    indexes, features, ground_truth = split_data(raw_data, config.granularity, hyperparameter_dict[hyplist.smoothing])
    if(hyperparameter_dict[hyplist.use_ref_points]): 
        features = calc_ref_features(features, hyperparameter_dict[hyplist.ref_point1], hyperparameter_dict[hyplist.ref_point2])
    x_train, y_train, x_val, y_val, x_test, y_test = slice_data(indexes, features, ground_truth, datasplit, hyperparameter_dict[hyplist.dilation_group][hyplist.past_history], config)
    # Data is now sliced into past_history slices

    datashape = x_train.shape[-2:]
    batch_train, batch_val, batch_test = batch_data(x_train, y_train, x_val, y_val, x_test, y_test, config.batch_size, 
                                                    config.epochs, hyperparameter_dict[hyplist.shuffle_buffer_size])

    #return batched_data(datashape, batch_train, batch_val, batch_test, len(x_train), len(x_val), len(x_test))
    return batched_data(datashape, batch_train, batch_val, batch_test, len(x_train), len(x_val), len(x_test), x_train, y_train, x_val, y_val, x_test, y_test)


def batch_data(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, epochs, shuffle_buffer_size):
    batched_train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle_buffer_size != 0:
        batched_train_data = batched_train_data.shuffle(shuffle_buffer_size)
    batched_train_data = batched_train_data.batch(batch_size).repeat(epochs)

    batched_val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)#.repeat(epochs)
    batched_test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return batched_train_data, batched_val_data, batched_test_data

#def shuffle_buffer_manual(dataset, buffer_size):
#    buffer = dataset[0:buffer_size]
#    shuffled_dataset = []
#    for i in len(dataset) - buffer_size:
#        rand_index = rand(buffer)
#        shuffled_dataset.append(buffer[rand_index])
#        buffer.remove(rand_index)
#        buffer.append(dataset[i])

def slice_data(indexes, features, ground_truth, datasplit, past_history, config: configuration):
    (train_start, val_start, test_start) = datasplit
    ft, sssw, gran = config.future_target, config.step_size_sliding_window, config.granularity

    dataset = features.values
    x = lambda a: int(len(dataset) * a)
    train_start_num, val_start_num, test_start_num = x(train_start), x(val_start), x(test_start)

    x_train, y_train = multivariate_data(dataset, ground_truth.values, train_start_num, val_start_num, past_history, ft, sssw, gran)
    x_val, y_val = multivariate_data(dataset, ground_truth.values, val_start_num, test_start_num, past_history, ft, sssw, gran)
    x_test, y_test = multivariate_data(dataset, ground_truth.values, test_start_num, None, past_history, ft, sssw, gran)
    return x_train, y_train, x_val, y_val, x_test, y_test

# Create array of all sliding windows of the data
def multivariate_data(dataset_features, dataset_ground_truth, start_index, end_index, history_size,
                      target_size, step, granularity):
    data, labels = [], []
    start_index = start_index + history_size 
    if end_index is None: 
        end_index = len(dataset_features)
    for i in range(start_index, end_index - target_size): # start 100, end 790. 
        indices = range(i-history_size, i, step) # range(0, 100) step size of 1          --- our sliding window
        data.append(dataset_features[indices]) # append new array that contains all values within our sliding window
        labels.append(dataset_ground_truth[i+target_size])
    return np.array(data), np.array(labels)

#def calc_ref_features2(features, ref_point1, ref_point2):
#    relative_features1 = [subtract_refvalue(obs, obs[ref_point1]) for obs in features.values]
#    relative_features2 = [subtract_refvalue(obs, obs[ref_point2]) for obs in features.values]
#    return pd.DataFrame([relative_features1[i] + relative_features2[i] for i in range(0, len(features))])
#def subtract_refvalue2(obs, ref_value):
#    return [val - ref_value for idx, val in enumerate(obs)]

def calc_ref_features(features, ref_point1, ref_point2):
    relative_features1 = [subtract_refvalue(obs, obs[ref_point1]) for obs in features.values]
    relative_features2 = [subtract_refvalue(obs, obs[ref_point2]) for obs in features.values]
    return pd.DataFrame([relative_features1[i] + relative_features2[i] for i in range(0, len(features))])
    # TODO: Make a new method for this, one which removes the two null reference points, and makes it percentage based instead

def subtract_refvalue(obs, ref_value):
    return [val - ref_value for val in obs]

def load_dataset(sheet_path, sheet_title, granularity):
    # sheet_data = pd.read_csv(f"Datasets/{sheet_path}_{sheet_title}_raw_data.csv") # Our old datasets had this weird naming scheme
    sheet_data = pd.read_csv(f"Datasets/{sheet_title}.csv")
    sheet_data.columns = columns_data
    return sheet_data

def split_data(raw_data, granularity, smoothing):
    indexes = range(0, len(raw_data), 1)[::granularity] # Each timestep is a millisecond
    features = raw_data[columns_features_considered][::granularity].ewm(span=smoothing).mean() 
    # TODO: Be aware that smoothing may not be applicable in an online application (where we dont have values before and after)
    ground_truth = pd.DataFrame(raw_data[column_ground_truth][::granularity]).ewm(span=smoothing).mean()
    return indexes, features, ground_truth
