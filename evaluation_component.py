
def evaluate_results(trained_model, batched_test_data, training_time):
    evaluate_model(trained_model, batched_test_data)
    return training_time

def evaluate_model(trained_model, batched_test_data):
    trained_model.evaluate(batched_test_data)

