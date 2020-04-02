from numpy import mean
from config_classes import hyperparameter_list, configuration
from data_manager_component import batched_data
from cnn_component import Model_CNN

class Model_Ensemble_CNN:
    def __init__(self, train_people: [batched_data], test_people: [batched_data], config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
        self._models = [Model_CNN(train_people[0].datashape, config, hyplist, hyperparameter_dict) for person in train_people]
        # It may be faster to perform deepcopy here ? !
        self._config = config
        self._hyplist = hyplist
        self._hyperparameter_dict = hyperparameter_dict
        self._train_people = train_people
        self._test_people = test_people
    
    def fit(self):
        for person, model in zip(self._train_people, self._models):
            model.fit(person.train, person.val, person.train_slices, person.val_slices, False)

    def evaluate(self):
        lst = []
        for model in self._models:
            for person in self._test_people:
                lst.append(model.evaluate(person.test))
        return mean(lst)