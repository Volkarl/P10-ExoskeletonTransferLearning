import pandas as pd
import tensorflow as tf

def slice_dataset(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, MEAN, VAL_PERCENT, PAST_HISTORY, 
                  FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, message, USE_REF_POINTS, REF_POINT1, REF_POINT2):
    print(f"Working on dataset: {DATASET_FILE_PATH} {DATASET_SHEET_TITLE} {message}")
    raw_data = load_dataset(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY)
    indexes, features, ground_truth = split_data(raw_data, GRANULARITY, SMOOTHING, MEAN)
    plot_dataset(features, ground_truth, indexes)
    if(USE_REF_POINTS):
        relative_features1 = [subtract_refvalue(obs, obs[REF_POINT1]) for obs in features.values]
        relative_features2 = [subtract_refvalue(obs, obs[REF_POINT2]) for obs in features.values]
        features = pd.DataFrame([relative_features1[i] + relative_features2[i] for i in range(0, len(features))])
        plot_dataset(features, ground_truth, indexes)
    x_train, y_train, x_val, y_val = slice_data(indexes, features, ground_truth, VAL_PERCENT, PAST_HISTORY, 
                                                FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, GRANULARITY)
    return x_train, y_train, x_val, y_val

def subtract_refvalue(obs, ref_value):
    return [val - ref_value for val in obs]

def slice_dataset_plain(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, MEAN, VAL_PERCENT, PAST_HISTORY, 
                  FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, message):
    print(f"Working on dataset: {DATASET_FILE_PATH} {DATASET_SHEET_TITLE} {message}")
    raw_data = load_dataset(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY)
    indexes, features, ground_truth = split_data(raw_data, GRANULARITY, SMOOTHING, MEAN)
    # plot_dataset(features, ground_truth, indexes)
    
    dataset = features.values
    observations = len(dataset)
    val_split = int(observations * (1 - VAL_PERCENT))

    x_train, y_train, x_val, y_val = features[:val_split], ground_truth[:val_split], features[val_split:], ground_truth[val_split:]
    return x_train, y_train, x_val, y_val

def process_multiple_sheets(start, end, DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, MEAN, 
                            VAL_PERCENT, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, 
                            REF_POINT1, REF_POINT2, SET_NAME):
    x_train_combine, y_train_combine, x_val_combine, y_val_combine = [], [], [], []
    datashape = (0,0)
    for i in range(start, end):
        x_train, y_train, x_val, y_val = slice_dataset(DATASET_FILE_PATH[i], DATASET_SHEET_TITLE[i], GRANULARITY, 
                                                       SMOOTHING, MEAN, VAL_PERCENT, PAST_HISTORY, FUTURE_TARGET, 
                                                       STEP_SIZE_SLIDING_WINDOW, f"{SET_NAME} number {i}", 
                                                       USE_REF_POINTS, REF_POINT1, REF_POINT2)
        datashape = x_train.shape[-2:]
        print(datashape)
        
        print(x_train.shape, x_val.shape)
        x_train_combine.extend(x_train)
        y_train_combine.extend(y_train)
        x_val_combine.extend(x_val)
        y_val_combine.extend(y_val)
        print(len(x_train_combine), len(x_val_combine))
    return x_train_combine, y_train_combine, x_val_combine, y_val_combine, datashape


def process_data(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, VAL_PERCENT, 
                 PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, BATCH_SIZE, EPOCHS, 
                 SHUFFLE_BUFFER_SIZE, MEAN, USE_REF_POINTS, REF_POINT1, REF_POINT2):
    sheet_num = len(DATASET_FILE_PATH) - 1
    
    x_train_all, y_train_all, x_val_all, y_val_all, _ = process_multiple_sheets(0, sheet_num - 1, DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, MEAN, VAL_PERCENT, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, REF_POINT1, REF_POINT2, "TRAIN SET")
    x_test_all, y_test_all, _, _, _ = process_multiple_sheets(sheet_num - 1, sheet_num, DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, MEAN, 0, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, REF_POINT1, REF_POINT2, "TEST SET")
    x_plot, y_plot, _, _, datashape = process_multiple_sheets(sheet_num - 1, sheet_num, DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, MEAN, 0, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, REF_POINT1, REF_POINT2, "PLOTTING SET")
    # The last one is used only for plotting, such that it doesn't take forever

    batch_train_data, batch_val_data, batch_test_data, batch_plot_data = batch_data(x_train_all, y_train_all, x_val_all, y_val_all, 
                                                                   x_test_all, y_test_all, x_plot, y_plot, 
                                                                   BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER_SIZE)
    test_data_indexes = range(0, len(x_plot) * GRANULARITY, GRANULARITY) # Necessary for plotting
    return { "batch_train_data": batch_train_data, "batch_val_data": batch_val_data, "batch_test_data": batch_test_data}, y_plot, len(x_train_all), len(x_val_all), test_data_indexes, datashape

# Definitions

# Data layout in the xlsx files
columns_data = ['1' ,'2', '3', '4', '5', '6', '7', '8', 'N/A_1', 'N/A_2', 'angle', 'time', 'session']
columns_features_considered = columns_data[:8]
column_ground_truth = columns_data[10]
# Note that we ignore the 'time' column. That makes our data slightly imprecise as there are tiny, 
# TINY differences in time intervals in the real data (not worth modeling). Each timestep represents 
# 1 millisecond, 0.001 second. 

def load_dataset(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY):
    # Read sheet 1 (table of contents), find index of entry with correct title, then load the corresponding excel sheet
    table_of_contents = pd.read_excel(DATASET_FILE_PATH, sheet_name=0, header=None)
    sheet_index = table_of_contents[table_of_contents[0] == f"{DATASET_SHEET_TITLE}_raw_data"][0].index[0]
    sheet_data = pd.read_excel(DATASET_FILE_PATH, sheet_name=sheet_index + 1, header=None)
    sheet_data.columns = columns_data
    return sheet_data

def mean_observations(features, indexes):
    features_len = len(features)
    observations_len = len(features.iloc[0])
    df = pd.DataFrame([(sum(features.iloc[i]) / observations_len) for i in range(0, features_len)])
    df.index = indexes
    return df

def split_data(raw_data, GRANULARITY, SMOOTHING, MEAN):
    indexes = range(0, len(raw_data), 1)[::GRANULARITY] # Each timestep is a millisecond
    features = raw_data[columns_features_considered][::GRANULARITY].ewm(span=SMOOTHING).mean()
    if(MEAN): features = mean_observations(features, indexes)
    ground_truth = pd.DataFrame(raw_data[column_ground_truth][::GRANULARITY]).ewm(span=SMOOTHING).mean()
    return indexes, features, ground_truth

def plot_dataset(features, ground_truth, indexes):
    features.plot(subplots=True) 
    plt.show()
    ground_truth.plot()
    plt.show()
    plt.plot(indexes, features) # Show all sensors together
    plt.show()

# Create array of all sliding windows of the data
def multivariate_data(dataset_features, dataset_ground_truth, start_index, end_index, history_size,
                      target_size, step, granularity, single_step=False, print_index=False):
    data, labels = [], []
    start_index = start_index + history_size 
    if end_index is None:
        end_index = len(dataset_features) - target_size 
    if print_index: print("start")
    for i in range(start_index, end_index): # start 100, end 790. 
        if print_index: print("A", i,)
        indices = range(i-history_size, i, step) # range(0, 100) step size of 1          --- our sliding window
        data.append(dataset_features[indices]) # append new array that contains all values within our sliding window
        if single_step:
            labels.append(dataset_ground_truth[i+target_size])
        else:
            labels.append(dataset_ground_truth[i:i+target_size])
    return np.array(data), np.array(labels)


def slice_data(indexes, features, ground_truth, VAL_PERCENT, PAST_HISTORY, FUTURE_TARGET, 
               STEP_SIZE_SLIDING_WINDOW, GRANULARITY):
    dataset = features.values
    observations = len(dataset)
    val_split = int(observations * (1 - VAL_PERCENT))
        
    x_train, y_train = multivariate_data(dataset, ground_truth.values, 0,
                                         val_split, PAST_HISTORY, FUTURE_TARGET, 
                                         STEP_SIZE_SLIDING_WINDOW, GRANULARITY, single_step = False, 
                                         print_index = False)
    x_val, y_val = multivariate_data(dataset, ground_truth.values, val_split, 
                                         None, PAST_HISTORY, FUTURE_TARGET, 
                                         STEP_SIZE_SLIDING_WINDOW, GRANULARITY, single_step=False, 
                                         print_index = False)
    
    return x_train, y_train, x_val, y_val

def batch_data(x_train, y_train, x_val, y_val, x_test, y_test, x_plot, y_plot, BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER_SIZE):
    batched_train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if SHUFFLE_BUFFER_SIZE == 0:
        batched_train_data = batched_train_data.batch(BATCH_SIZE).repeat(EPOCHS)
    else:
        batched_train_data = batched_train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat(EPOCHS)

    batched_val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).repeat(EPOCHS)
    batched_test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1) # batch size of 1, no repeat
    batched_plot_data = tf.data.Dataset.from_tensor_slices((x_plot, y_plot)).batch(1)
    return batched_train_data, batched_val_data, batched_test_data, batched_plot_data


# In[ ]:





# In[ ]:




