# Imports
import tensorflow as tf
print(tf.test.is_gpu_available())
from hyperopt import STATUS_OK
from functools import partial

from config_classes import hyperparameters, configuration
import optimizer_component as opt
import data_management_component as data
#import cnn_baseline as cnn
#import evaluation_component as eval

def objective(config_dict, hyperparameters, hyplist: hyperparams): 
    loss, exec_time = run_all(config_dict, hyperparameters, hyplist)
    return { 'loss': loss, 'status': STATUS_OK, 'exec_time': exec_time }

config = configuration(1,1,1,1,1,1,1,1,1) # TODO FIX ME
hyperparams = hyperparameters()
partial_objective = partial(objective, config_dict = config, hyplist = hyperparams)
# This is basically function currying. Defines our objective function with the config_dict parameter already present
opt.perform_hyperopt(partial_objective, hyperparams.space, 100)


def run_all(config_dict, hyperparameters, hyplist: hyperparams): 
    # Config_dict is set by this file. Specific hyperparams are given by hyperopt

    # Access individual hyperparams by using hyperparameters[hyplist.HYPERPARAM_NAME]
    data.process_data(config_dict, hyperparameters, hyplist) 
    # TODO: Fix these. Rewrite them in new temp files, simplify and remove unecessary stuff like my plotting etc.
    cnn.run_cnn(hyperparameters, hyplist) 
    eval.evaluate_results()
    return loss, exec_time
