from config_classes import hyperparameter_list, configuration
import tensorflow as tf
from timeit import default_timer as timer

optimizer_dict = {'adadelta': tf.keras.optimizers.Adadelta(), 'adam': tf.keras.optimizers.Adam(), 'rmsprop': tf.keras.optimizers.RMSprop()}

def compile_model_cnn(data_shape, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    model = tf.keras.models.Sequential()
    ks = hyperparameter_dict[hyplist.kernel_size].item() # This is required because my searchspace doesn't pass proper, native int values, for god knows what reason

    model.add(tf.keras.layers.InputLayer(input_shape=data_shape))

    #if(hyperparameter_dict[hyplist.use_dilation]): model.add(tf.keras.layers.Conv1D(input_shape=data_shape, filters=hyperparameter_dict[hyplist.filters], kernel_size=ks, padding=config.padding, kernel_initializer=config.kernel_initializer, activation=config.activation, dilation_rate=i**config.dilation_rate))
    #else: model.add(tf.keras.layers.Conv1D(input_shape=data_shape, filters=hyperparameter_dict[hyplist.filters], kernel_size=ks, padding=config.padding, kernel_initializer=config.kernel_initializer, activation=config.activation))

    for i in range(hyperparameter_dict[hyplist.layer_amount]):
        if(hyperparameter_dict[hyplist.use_dilation]):
            model.add(tf.keras.layers.Conv1D(filters=hyperparameter_dict[hyplist.filters], kernel_size=ks, padding=config.padding, kernel_initializer=config.kernel_initializer, activation=config.activation, dilation_rate=config.dilation_rate**i))
        else:
            model.add(tf.keras.layers.Conv1D(filters=hyperparameter_dict[hyplist.filters], kernel_size=ks, padding=config.padding, kernel_initializer=config.kernel_initializer, activation=config.activation))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(config.future_target)) 
    model.compile(optimizer=hyperparameter_dict[hyplist.optimizer], loss='mae', metrics=['mae', 'mape', 'mse'])
    model.summary()
    return model

def fit_model_cnn(model, batched_train_data, batched_val_data, train_slices, val_slices, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict, STUFF):
    train_batches = train_slices // config.batch_size
    val_batches = val_slices // config.batch_size
    # Splits the dataset into batches of this size: we perform gradiant descent once per batch
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=config.min_delta, patience=config.patience, verbose=1, mode='min', baseline=None, restore_best_weights=True)
    
    start = timer()
    training_history = model.fit(#batched_train_data, 
                                 STUFF[0], STUFF[1],
                                 epochs=config.epochs, 
                                 steps_per_epoch=train_batches, #TODO: THIS IS CLEARLY NOT THE PROBLEM BECAUSE BUG PERSISTS. 
                                 # TODO: TRY NOT BATCHING MY DATASET, AND PASSING A BATCH SIZE INSTEAD?
                                 #validation_data=batched_val_data,
                                 validation_data=(STUFF[2], STUFF[3]),
                                 validation_steps=val_batches,
                                 batch_size=config.batch_size, #TODO
                                 callbacks=[es])

    #TODO: You know what, maybe just try making it a generator function anyhow if nothing else works? Meh that's another larger refactor.

    # TODO: STOP EN HALV. Vi fejler når den kalder model.validate, men hvorfor gør den først det EFTER den har været gennem alle 
    # epochs af træningsdata?? Eller er det kun ved den femte at den fucker? Den kan jo tydeligvis regne alle fem val losses ud.

    end = timer()
    return training_history, end - start # time in seconds


# Use model.train's argument for class_weight and perhaps sample_weight