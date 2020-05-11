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
        self.dilation_group = "dilation_group"

    #@staticmethod
    def space(self):
        return {
            self.kernel_size: hp.choice(self.kernel_size, [2, 3, 4]), # This needs to be a small value, otherwise it will cause dimensionality errors unless past_history divided by layer size is sufficiently large
            self.smoothing: hp.choice(self.smoothing, [1, 25, 50, 75, 100]),
            self.shuffle_buffer_size: 1 + hp.randint(self.shuffle_buffer_size, 100),
            self.filters: 1 + hp.randint(self.filters, 100),
            self.optimizer: hp.choice(self.optimizer, ["adadelta", "adam", "rmsprop"]),
            self.use_ref_points: hp.choice(self.use_ref_points, [False, True]),
            self.ref_point1: hp.choice(self.ref_point1, [0, 1, 2, 3]),
            self.ref_point2: hp.choice(self.ref_point2, [4, 5, 6, 7]), # TODO: Make ref points a choice as well
            self.dilation_group: hp.choice(self.dilation_group, [{ 
                    self.use_dilation: False,
                    self.layer_amount: 1 + hp.randint("dilation_layer_false", 9),
                    self.past_history: 20 + hp.randint("dilation_past_false", 280)
                    # TODO: Add kernel size here as well
                },{
                    self.use_dilation: True, # We use dilation: therefore require a high pasthistory and low layeramount
                    self.layer_amount: 2 + hp.randint("dilation_layer_true", 4),
                    self.past_history: 150 + hp.randint("dilation_past_true", 150)
                }])
        }

# TODO: Add pooling layers

    def best_arguments(self): # This is just hard coded to whatever hyperopt found
        return {
            self.kernel_size: 2,
            self.smoothing: 25,
            self.shuffle_buffer_size: 1 + 17,
            self.filters: 1 + 22,
            self.optimizer: "adam",
            self.use_ref_points: True,
            self.ref_point1: 2,
            self.ref_point2: 6,
            self.dilation_group: {
                self.use_dilation: True,
                self.layer_amount: 2 + 4,
                self.past_history: 150 + 141
            }
        }

class configuration:
    def __init__(self): 
        self.dataset_sheet_titles = ["EmilA", "EmilB", "EmilC", "EmilD", "EmilE", "PalleA", "PalleB", "PalleC", "PalleD", "PalleE", "JonaA", "JonaB", "JonaC", "JonaD", "JonaE"]
        self.attempt_name = "attempt_name"
        self.granularity = 30
        self.step_size_sliding_window = 1
        self.future_target = 1
        self.epochs = 10
        self.batch_size = 500
        self.min_delta = 0.0001
        self.patience = 10
        self.padding = "valid"
        self.kernel_initializer = 'uniform'
        self.activation= 'relu'
        self.dilation_rate = 2
        self.ppl_amount = 3
        self.sheets_per_person = 5 

    def get_people_iterator(self):
        spp = self.sheets_per_person
        sheets = self.ppl_amount * spp

        people_files = [self.dataset_sheet_titles[i:i+spp] for i in range(0, sheets, spp)]
        # Note that this grabs the test sheets from right after our train sheets, not necessarily the last sheets
        return people_files
