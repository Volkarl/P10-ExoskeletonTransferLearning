
def evaluate_results(trained_model, training_history, test_ground_truth, batched_test_data, batched_plot_data, test_data_indexes, training_time):
    execution_time = plot_results(training_history, trained_model, test_ground_truth, batched_plot_data, test_data_indexes)
    evaluate_model(trained_model, batched_test_data)
    print_time(training_time, execution_time)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.show()

def plot_all(trained_model, test_ground_truth, batched_plot_data, test_data_indexes):
    start = timer()
    predictions = [trained_model.predict(elem)[0] for elem in batched_plot_data]
    end = timer()
    plt.plot(test_data_indexes, predictions, 'r', label='Prediction')
    plt.plot(test_data_indexes, np.ndarray.flatten(np.array(test_ground_truth)), 'b', label='Ground Truth') 
    plt.legend(loc='upper left')
    obs = len(test_data_indexes)
    return f"time {end - start}, observations {obs}, time per observation {(end - start) / obs}"

def plot_results(training_history, trained_model, test_ground_truth, batched_plot_data, test_data_indexes):
    plot_train_history(training_history, 'Multi-Step Training and validation loss')
    execution_time = plot_all(trained_model, test_ground_truth, batched_plot_data, test_data_indexes)
    return execution_time

def evaluate_model(trained_model, batched_test_data):
    trained_model.evaluate(batched_test_data)

def print_time(training_time, execution_time):
    print(f"Total training time: {training_time} seconds")
    print(f"Execution time: {execution_time} seconds")
