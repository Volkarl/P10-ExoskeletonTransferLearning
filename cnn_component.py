from config_classes import hyperparameter_list, configuration
import tensorflow as tf

optimizer_dict = {'adadelta': tf.keras.optimizers.Adadelta(), 'adam': tf.keras.optimizers.Adam(), 'rmsprop': tf.keras.optimizers.RMSprop()}

def compile_model_cnn(data_shape, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    model = tf.keras.models.Sequential()
    ks = hyperparameter_dict[hyplist.dilation_group][hyplist.kernel_size]

    model.add(tf.keras.layers.InputLayer(input_shape=data_shape))
    for i in range(hyperparameter_dict[hyplist.dilation_group][hyplist.layer_amount]):
        if(hyperparameter_dict[hyplist.dilation_group][hyplist.use_dilation]):
            model.add(tf.keras.layers.Conv1D(filters=hyperparameter_dict[hyplist.filters], kernel_size=ks, padding=config.padding, kernel_initializer=config.kernel_initializer, activation=config.activation, dilation_rate=config.dilation_rate**i))
        else:
            model.add(tf.keras.layers.Conv1D(filters=hyperparameter_dict[hyplist.filters], kernel_size=ks, padding=config.padding, kernel_initializer=config.kernel_initializer, activation=config.activation))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(config.future_target)) 
    model.compile(optimizer=hyperparameter_dict[hyplist.optimizer], loss='mae')
    model.summary()
    return model

def evaluate_model_cnn(trained_model, batched_test_data):
    return trained_model.evaluate(batched_test_data, verbose=2)

class Model_CNN:
    def __init__(self, datashape, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
        self._model = compile_model_cnn(datashape, config, hyplist, hyperparameter_dict)
        self._config = config
        self._hyplist = hyplist
        self._hyperparameter_dict = hyperparameter_dict

    def fit_ada(self, x, y): # TODO NAME CHANGE TO fit_unbatched
        self._model.fit(x, y, batch_size=self._config.batch_size, epochs=self._config.epochs, verbose=0)

    def predict(self, x):
        return self._model.predict(x)

    def evaluate(self, batched_test_data):
        return evaluate_model_cnn(self._model, batched_test_data)

    def evaluate_nonbatched(self, x, y):
        return self._model.evaluate(x, y, verbose=2)