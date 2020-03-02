from config_classes import hyperparameter_list, configuration

def process_data(config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    paths, titles = config.dataset_file_paths, config.dataset_sheet_titles
    sheet_num = len(paths) - 1
    
    x_train_all, y_train_all, x_val_all, y_val_all, _ = process_multiple_sheets(0, sheet_num - 1, paths, titles, GRANULARITY, SMOOTHING, MEAN, VAL_PERCENT, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, REF_POINT1, REF_POINT2, "TRAIN SET")
    x_test_all, y_test_all, _, _, _ = process_multiple_sheets(sheet_num - 1, sheet_num, paths, titles, GRANULARITY, SMOOTHING, MEAN, 0, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, REF_POINT1, REF_POINT2, "TEST SET")
    x_plot, y_plot, _, _, datashape = process_multiple_sheets(sheet_num - 1, sheet_num, paths, titles, GRANULARITY, SMOOTHING, MEAN, 0, PAST_HISTORY, FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, USE_REF_POINTS, REF_POINT1, REF_POINT2, "PLOTTING SET")
    # The last one is used only for plotting, such that it doesn't take forever

    batch_train_data, batch_val_data, batch_test_data, batch_plot_data = batch_data(x_train_all, y_train_all, x_val_all, y_val_all, 
                                                                   x_test_all, y_test_all, x_plot, y_plot, 
                                                                   BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER_SIZE)
    test_data_indexes = range(0, len(x_plot) * GRANULARITY, GRANULARITY) # Necessary for plotting
    return { "batch_train_data": batch_train_data, "batch_val_data": batch_val_data, "batch_test_data": batch_test_data}, y_plot, len(x_train_all), len(x_val_all), test_data_indexes, datashape


# Use Fron_generator functions? 
# Maybe just transform one sheet at a time, and then feed it to model.fit, then the next one. 

# Make a generator function, ala https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# It generates one batch at a time!




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
