from hyperopt import hp, STATUS_OK

class hyperparameter_list:
    def __init__(self):
        self.past_history = "past_history"
        self.smoothing = "smoothing"
        self.shuffle_buffer_size = "shuffle_buffer_size"
        self.kernel_size = "kernel_size"
        self.filters = "filters"
        self.padding = "padding"
        self.optimizer = "optimizer"
        self.use_ref_points = "use_ref_points"
        self.ref_point1 = "ref_point1"
        self.ref_point2 = "ref_point2"
        self.use_dilation = "use_dilation"
        self.dilation_rate = "dilation_rate"
        self.layer_amount = "layer_amount"

    #@staticmethod
    def space(self):
        return {
          self.kernel_size: hp.uniform(self.kernel_size, 2, 100),
          self.filters: hp.uniform(self.filters, 2, 100),
          self.optimizer: hp.choice(self.optimizer, ["adadelta", "adam", "rmsprop"])
        # past_history, 
        # smoothing, 
        # shuffle_buffer_size, 
        # kernel_size, 
        # filters, 
        # padding, 
        # optimizer, 
        # use_ref_points, 
        # ref_point1, 
        # ref_point2, 
        # use_dilation, 
        # dilation_rate
        }

class configuration:
    def __init__(self): 
        self.dataset_file_paths = ["Datasets\\SingleSessionData.xlsx","Datasets\\SingleSessionData.xlsx","Datasets\\SingleSessionData.xlsx","Datasets\\SingleSessionData.xlsx", "Datasets\\SingleSessionData.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionOne.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionTwo.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionThree.xlsx","Datasets\\SessionToSessionFour.xlsx","Datasets\\SessionToSessionFour.xlsx","Datasets\\SessionToSessionFour.xlsx","Datasets\\SessionToSessionFour.xlsx"]
        self.dataset_sheet_titles = ["data_Uniform","data_NonUniform","data_NonUniformWithPause","data_NonUniformTwo", "data_UniformWithPause","data_Uniform","data_NonUniform","data_NonUniformWithPause","data_UniformWithPause","data_Uniform","data_UniformTwo","data_NonUniform","data_NonUniformTwo","data_Uniform","data_UniformTwo","data_NonUniform","data_NonUniformTwo","data_NonUniform","data_NonUniformTwo","data_Uniform","data_UniformTwo"]
        self.attempt_name = "attempt_name"
        self.granularity = 1
        self.step_size_sliding_window = 1
        self.future_target = 1
        self.epochs = 5
        self.batch_size = 500
        self.min_delta = 0.0001
        self.patience = 20
        self.ada_datasplit = (0, 0.8, 0.9, 1)
        self.cnn_datasplit = (0, 0.9, 1, 1)
        self.cnn_testsplit = (0, 0, 0, 1)
