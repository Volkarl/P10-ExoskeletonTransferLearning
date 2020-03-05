from config_classes import hyperparameter_list, configuration
import tensorflow as tf

def compile_cnn(data_shape, batched_train_data, batched_val_data, train_slices, 
             val_slices, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    model = compile_model_cnn(data_shape, config.future_target, hyplist.kernel_size, hyplist.filters, hyplist.padding)
    return model

def fit_cnn(data_shape, batched_train_data, batched_val_data, train_slices, 
             val_slices, model, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    training_history, training_time = fit_model_cnn(model, batched_train_data, batched_val_data, train_slices, 
                                                val_slices, config.batch_size, config.epochs, config.min_delta, config.patience)
    return training_history, training_time

def compile_model_cnn(data_shape, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv1D(filters=hyplist.filters, kernel_size=hyplist.kernel_size, padding=hyplist.padding, input_shape=data_shape ,kernel_initializer= 'uniform', activation= 'relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(config.future_target)) 
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics=['mae', 'mape', 'mse'])
    model.summary()
    return model



# Use model.train's argument for class_weight and perhaps sample_weight