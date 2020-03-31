from config_classes import hyperparameter_list, configuration
from cnn_component import compile_model_cnn, fit_model_cnn, evaluate_model_cnn

def compile_model_ensemble(shape, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    models = []
    for i in range(3):
        model = compile_model_cnn(shape, config, hyplist, hyperparameter_dict)
        model = cnn.fit_model_cnn(model, train, val, train_slices, val_slices, True, config, hyplist, hyperparameter_dict)
        models.append(model)

        #TODO: Make proper class structure with my CNN and my Ensemble models, so that I don't have to duplicate code so they can build op top of each other.
        # Ensemble is an aggregate of multiple CNNs