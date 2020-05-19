from numpy import mean
from config_classes import hyperparameter_list, configuration
from data_manager_component import batched_data
from cnn_component import Model_CNN
from plotting import plot_multiple_comparisons

class Model_Ensemble_CNN:
    def __init__(self, models):
        self._models = models
    
    def predict(self, X, Y, show_plot = False):
        predictions_all = [model.predict(X) for model in self._models]
        predictions_mean = mean(predictions_all, axis=0)

        if show_plot:
            predictions_all.append(predictions_mean)
            predictions_all.append(Y)
            plot_multiple_comparisons(predictions_all, ["CNN1", "CNN2", "CNN3", "Mean", "Person C Test Set"], ["m", "g", "r", "k", "b"], "x", "y", "Ensemble Model")
        return predictions_mean