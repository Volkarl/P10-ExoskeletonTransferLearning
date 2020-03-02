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

    @staticmethod
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
    def __init__(self, dataset_file_paths, dataset_sheet_titles, attempt_name, granularity, step_size_sliding_window, 
                 future_target, val_percent, epochs, batch_size): 
        self.dataset_file_paths = dataset_file_paths
        self.dataset_sheet_titles = dataset_sheet_titles
        self.attempt_name = attempt_name
        self.granularity = granularity
        self.step_size_sliding_window = step_size_sliding_window
        self.future_target = future_target
        self.val_percent = val_percent
        self.epochs = epochs
        self.batch_size = batch_size
