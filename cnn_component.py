from config_classes import hyperparameter_list, configuration
import tensorflow as tf
from timeit import default_timer as timer

optimizer_dict = {'adadelta': tf.keras.optimizers.Adadelta(), 'adam': tf.keras.optimizers.Adam(), 'rmsprop': tf.keras.optimizers.RMSprop()}

def compile_model_cnn(data_shape, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    model = tf.keras.models.Sequential()
    ks = hyperparameter_dict[hyplist.kernel_size] # This is required because my searchspace doesn't pass proper, native int values, for god knows what reason

    # TODO Maybe try adding a max-pooling layer for dimensionality reduction (noise) and instead of granularity for downsampling

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

def fit_model_cnn(model, batched_train_data, batched_val_data, train_slices, val_slices, use_timer, config: configuration, hyplist: hyperparameter_list, hyperparameter_dict):
    train_batches = train_slices // config.batch_size
    val_batches = val_slices // config.batch_size
    # Splits the dataset into batches of this size: we perform gradiant descent once per batch
    #es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=config.min_delta, patience=config.patience, verbose=1, mode='min', baseline=None, restore_best_weights=True)
    
    start, end = 0, 0
    if(use_timer): start = timer()
    model.fit(batched_train_data, epochs=config.epochs, 
              steps_per_epoch=train_batches,
              validation_data=batched_val_data,
              validation_steps=val_batches,
              #callbacks=[es],
              verbose=0)
    if(use_timer): end = timer()
    return end - start # time in seconds


# TODO: ?? Use model.train's argument for class_weight and perhaps sample_weight


def evaluate_model_cnn(trained_model, batched_test_data):
    return trained_model.evaluate(batched_test_data, verbose=2)
