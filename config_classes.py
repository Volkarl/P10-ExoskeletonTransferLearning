from hyperopt import hp

class hyperparameter_list:
    def __init__(self):
        self.past_history = "past_history"
        self.smoothing = "smoothing"
        self.shuffle_buffer_size = "shuffle_buffer_size"
        self.kernel_size = "kernel_size"
        self.filters = "filters"
        self.optimizer = "optimizer"
        self.use_ref_points = "use_ref_points"
        self.ref_point1 = "ref_point1"
        self.ref_point2 = "ref_point2"
        self.use_dilation = "use_dilation"
        self.layer_amount = "layer_amount"

    #@staticmethod
    def space(self):
        return {

            self.past_history: 1 + hp.randint(self.past_history, 300),
            self.smoothing: hp.choice(self.smoothing, [1, 25, 50, 75, 100]),
            self.shuffle_buffer_size: 1 + hp.randint(self.shuffle_buffer_size, 100),
            self.kernel_size: 2 + hp.randint(self.kernel_size, 100),
            self.filters: 1 + hp.randint(self.filters, 100),
            self.optimizer: hp.choice(self.optimizer, ["adadelta", "adam", "rmsprop"]),
            self.use_ref_points: hp.choice(self.use_ref_points, [True, False]),
            self.ref_point1: hp.choice(self.ref_point1, [0, 1, 2, 3]),
            self.ref_point2: hp.choice(self.ref_point2, [4, 5, 6, 7]),
            self.use_dilation: hp.choice(self.use_dilation, [True, False]),
            self.layer_amount: 1 + hp.randint(self.layer_amount, 10) # Values distributed in interval: (0, 10]
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
        self.padding = "valid"
        self.kernel_initializer = 'uniform'
        self.activation= 'relu'
        self.dilation_rate = 2
        self.ada_datasplit = (0, 0.8, 0.9)
        self.cnn_datasplit = (0, 0.9, 1)
        self.cnn_testsplit = (0, 0, 0)
