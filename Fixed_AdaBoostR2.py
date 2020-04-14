"""
Code is based upon github.com/jay15summer/Two-stage-TrAdaboost.R2, modified for usage with Keras
------------------------------------------------------------------------------------------------
Algorithm: AdaBoost.R2

Input: source+target datasets, weight vector, amount of adaboost iterations N (how many estimators we will try out), 
For each adaboost iteration, N:
    1. Call base learner on the entire (?something about bootstrap_idx that I dont understand) dataset with weight vector, obtain a trained model (also called a hypothesis)
    2. Use hypothesis to calculate error for each sample
    3. Calculate error for the current hypothesis 
    4. Calculate beta-value
    5. Update the entire weight vector according to the beta-value
Output: The class is now an ensemble learner. It outputs the weighted median of all its constituent hypotheses

Notes:
Step (5) is different when we use it for Two-StageTrAdaBoost.R2. Here, we never change the percentwise value of the source weights, though we do keep adjusting the target weights
"""

# TODO Wrap my CNN stuff with the SKLearn wrapper function, KerasRegressor 


class AdaBoostR2:
    def __init__(self,
                 base_estimator,
                 sample_size = None,
                 n_estimators = 50,
                 learning_rate = 1.,
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
