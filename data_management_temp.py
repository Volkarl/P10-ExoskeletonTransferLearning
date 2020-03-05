import pandas as pd
from config_classes import hyperparameter_list, configuration

# Data layout in the xlsx files
columns_data = ['1' ,'2', '3', '4', '5', '6', '7', '8', 'N/A_1', 'N/A_2', 'angle', 'time', 'session']
columns_features_considered = columns_data[:8]
column_ground_truth = columns_data[10]
# Each timestep represents 1 millisecond, 0.001 second. 

def process_sheet(sheet_path, sheet_title, datasplit, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    raw_data = load_dataset(sheet_path, sheet_title, config.granularity)
    indexes, features, ground_truth = split_data(raw_data, config.granularity, config.smoothing)
    if(hyperparameter_dict[hyplist.use_ref_points]): 
        features = calc_ref_features(features, hyperparameter_dict(hyplist.ref_point1), hyperparameter_dict(hyplist.ref_point2))



    # TODO FIX THE ONE BELOW
    x_train, y_train, x_val, y_val, x_test, y_test = slice_data(indexes, features, ground_truth, hyperparameter_dict[hyplist.past_history], 
                                                FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, GRANULARITY)
    return x_train, y_train, x_val, y_val, x_test, y_test


def slice_data(indexes, features, ground_truth, VAL_PERCENT, PAST_HISTORY, FUTURE_TARGET, 
               STEP_SIZE_SLIDING_WINDOW, GRANULARITY):
    (train_start, train_end, val_end, test_end) = datasplit



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



def calc_ref_features(features, ref_point1, ref_point2):
    relative_features1 = [subtract_refvalue(obs, obs[ref_point1]) for obs in features.values]
    relative_features2 = [subtract_refvalue(obs, obs[ref_point2]) for obs in features.values]
    return pd.DataFrame([relative_features1[i] + relative_features2[i] for i in range(0, len(features))])

def subtract_refvalue(obs, ref_value):
    return [val - ref_value for val in obs]

def load_dataset(sheet_path, sheet_title, granularity):
    # Read sheet 1 (table of contents), find index of entry with correct title, then load the corresponding excel sheet
    table_of_contents = pd.read_excel(sheet_path, sheet_name=0, header=None)
    sheet_index = table_of_contents[table_of_contents[0] == f"{sheet_title}_raw_data"][0].index[0]
    sheet_data = pd.read_excel(sheet_path, sheet_name=sheet_index + 1, header=None)
    sheet_data.columns = columns_data
    return sheet_data

def split_data(raw_data, granularity, smoothing):
    indexes = range(0, len(raw_data), 1)[::granularity] # Each timestep is a millisecond
    features = raw_data[columns_features_considered][::granularity].ewm(span=smoothing).mean() 
    # TODO: Be aware that smoothing may not be applicable in an online application (where we dont have values before and after)
    ground_truth = pd.DataFrame(raw_data[column_ground_truth][::granularity]).ewm(span=smoothing).mean()
    return indexes, features, ground_truth
