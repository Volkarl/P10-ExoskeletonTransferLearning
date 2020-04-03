from numpy import mean
from config_classes import hyperparameter_list, configuration
from data_manager_component import batched_data
from cnn_component import Model_CNN

class Model_Ensemble_CNN:
    def __init__(self, datashape, person_amount, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
        self._models = [Model_CNN(datashape, config, hyplist, hyperparameter_dict) for i in range(person_amount)]
        # It may be faster to perform deepcopy here ? !
        self.train_time = 0
    
    def fit(self, use_timer, model_idx, train_person_sessions: [batched_data]):
        train_time_lst = []
        for s_idx, session in enumerate(train_person_sessions):
            print(f"Training MODEL {model_idx + 1} with SESSION {s_idx + 1}")
            self._models[model_idx].fit(session.train, session.val, session.train_slices, session.val_slices, use_timer)
            train_time_lst.append(self._models[model_idx].train_time)
        if use_timer: self.train_time = mean(train_time_lst)

    def evaluate(self, test_person_sessions: [batched_data]):
        loss_lst = []
        for model in self._models:
            for session in test_person_sessions:
                loss_lst.append(model.evaluate(session.test))
        loss = mean(loss_lst)
        print("ENSEMBLE MODEL MEAN LOSS: " + loss)
        return loss