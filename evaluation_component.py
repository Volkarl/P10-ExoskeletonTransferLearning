
def evaluate_model(trained_model, batched_test_data):
    return trained_model.evaluate(batched_test_data)
    # TODO: It probably prints into the console. This needs to change
    # TODO: Hopefully this returns loss.
