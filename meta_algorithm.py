# Imports
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from timeit import default_timer as timer

import data_management_component as data
import cnn_baseline as cnn
import evaluation_component as eval

print(tf.test.is_gpu_available())
plt.rcParams["figure.figsize"] = (20,10)

# Parameters 
# ATTEMPT_NAME="LSTM_BASELINE"
# DATASET_FILE_PATH= ["Datasets\\uniformdata.xlsx", "Datasets\\uniformdata.xlsx", "Datasets\\uniformdata.xlsx", "Datasets\\uniformdata.xlsx", "Datasets\\uniformdata.xlsx"]
# DATASET_SHEET_TITLE= ["data_uniforma", "data_uniformb", "data_uniformc", "data_uniformd", "data_uniforme"]
# GRANULARITY=10
# STEP_SIZE_SLIDING_WINDOW=5
# PAST_HISTORY=20
# FUTURE_TARGET=1
# VAL_PERCENT=0.1
# EPOCHS=5
# BATCH_SIZE=10
# SMOOTHING=50
# SHUFFLE_BUFFER_SIZE=100
# MEAN=False
# USE_REF_POINTS=False
# REF_POINT1=0
# REF_POINT2=7


# TODO Pass objects to this function instead of this shit
def run_all(DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, STEP_SIZE_SLIDING_WINDOW, PAST_HISTORY, 
            FUTURE_TARGET, VAL_PERCENT, EPOCHS, BATCH_SIZE, SMOOTHING, ATTEMPT_NAME, SHUFFLE_BUFFER_SIZE, MEAN,
            USE_REF_POINTS, REF_POINT1, REF_POINT2, KERNEL_SIZE, FILTERS, PADDING): 
    batched_train_data, batched_val_data, batch_test, test_ground_truth, train_slices, val_slices, test_data_indexes, data_shape = data.process_data(
        DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, VAL_PERCENT, PAST_HISTORY, 
        FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER_SIZE, MEAN, 
        USE_REF_POINTS, REF_POINT1, REF_POINT2)
    model, training_history, training_time = cnn.run_cnn(data_shape, batched_train_data, batched_val_data, train_slices, 
             val_slices, BATCH_SIZE, EPOCHS, FUTURE_TARGET, KERNEL_SIZE, FILTERS, PADDING, MIN_DELTA, PATIENCE) 
    eval.evaluate_results(model, training_history, test_ground_truth, batch_test, test_data_indexes, training_time)

# Load components

DATASET_FILE_PATH=["Datasets\\SingleSessionData.xlsx","Datasets\\SingleSessionData.xlsx","Datasets\\SingleSessionData.xlsx","Datasets\\SingleSessionData.xlsx", "Datasets\\SingleSessionData.xlsx"]#,"Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionFour.xlsx","Datasets\\SessionToSessionFour.xlsx","Datasets\\SessionToSessionFour.xlsx","Datasets\\SessionToSessionFour.xlsx"]
DATASET_SHEET_TITLE=["data_Uniform","data_NonUniform","data_NonUniformWithPause","data_NonUniformTwo", "data_UniformWithPause"]#,"data_Uniform","data_NonUniform","data_NonUniformWithPause","data_UniformWithPause","data_Uniform","data_UniformTwo","data_NonUniform","data_NonUniformTwo","data_Uniform","data_UniformTwo","data_NonUniform","data_NonUniformTwo","data_NonUniform","data_NonUniformTwo","data_Uniform","data_UniformTwo"]
GRANULARITY=10
STEP_SIZE_SLIDING_WINDOW=1
PAST_HISTORY=100
FUTURE_TARGET=1
VAL_PERCENT=0.15
EPOCHS=100 # Max epochs, in case early stopping doesn't take effect
MIN_DELTA=0.0001
PATIENCE=20
BATCH_SIZE=500
SMOOTHING=50
SHUFFLE_BUFFER_SIZE=0
MEAN=False
USE_REF_POINTS=True
REF_POINT1=0
REF_POINT2=7

plt.rcParams["figure.figsize"] = (20,10)


#ATTEMPT_NAME="CNN_REF_BiTri"
#KERNEL_SIZE=10
#FILTERS=32
#PADDING="valid"

#batch_train, batch_val, batch_test, batch_plot, test_ground_truth, train_slices, val_slices, test_data_indexes, data_shape = process_data(
#    DATASET_FILE_PATH, DATASET_SHEET_TITLE, GRANULARITY, SMOOTHING, VAL_PERCENT, PAST_HISTORY, 
#    FUTURE_TARGET, STEP_SIZE_SLIDING_WINDOW, BATCH_SIZE, EPOCHS, SHUFFLE_BUFFER_SIZE, MEAN, 
#    USE_REF_POINTS, REF_POINT1, REF_POINT2)
#model, training_history, training_time = run_cnn(data_shape, batch_train, batch_val, train_slices, val_slices, BATCH_SIZE, EPOCHS, FUTURE_TARGET, KERNEL_SIZE, FILTERS, PADDING, MIN_DELTA, PATIENCE) 
#evaluate_results(model, training_history, test_ground_truth, batch_test, batch_plot, test_data_indexes, training_time)
