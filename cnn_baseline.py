
def run_cnn(data_shape, batched_train_data, batched_val_data, train_slices, 
             val_slices, BATCH_SIZE, EPOCHS, FUTURE_TARGET, KERNEL_SIZE, FILTERS, PADDING, MIN_DELTA, PATIENCE):
    model = compile_model_cnn(data_shape, FUTURE_TARGET, KERNEL_SIZE, FILTERS, PADDING)
    training_history, training_time = fit_model_cnn(model, batched_train_data, batched_val_data, train_slices, 
                                                val_slices, BATCH_SIZE, EPOCHS, MIN_DELTA, PATIENCE)
    return model, training_history, training_time

def compile_model_cnn(data_shape, FUTURE_TARGET, KERNEL_SIZE, FILTERS, PADDING):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, input_shape=data_shape,kernel_initializer= 'uniform', activation= 'relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(FUTURE_TARGET)) 
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae', metrics=['mae', 'mape', 'mse'])
    model.summary()
    return model

def fit_model_cnn(model, batched_train_data, batched_val_data, train_slices, val_slices, BATCH_SIZE, EPOCHS, MIN_DELTA, PATIENCE):
    train_batches = train_slices // BATCH_SIZE 
    val_batches = val_slices // BATCH_SIZE
    # Splits the dataset into batches of this size: we perform gradiant descent once per batch
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=PATIENCE, verbose=1, mode='min', baseline=None, restore_best_weights=True)
    
    start = timer()
    training_history = model.fit(batched_train_data, epochs=EPOCHS, 
                                 steps_per_epoch=train_batches,
                                 validation_data=batched_val_data,
                                 validation_steps=val_batches,
                                 callbacks=[es])
    end = timer()
    return training_history, end - start # time in seconds
