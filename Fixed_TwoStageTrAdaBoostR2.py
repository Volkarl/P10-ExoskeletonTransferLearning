"""
Code is based upon github.com/jay15summer/Two-stage-TrAdaboost.R2, modified for usage with Keras
------------------------------------------------------------------------------------------------
Algorithm: Two-stage TrAdaBoost.R2

Input: source+target datasets, initialized weight vector, amount of boosting steps S, adaboost iterations N
For each boosting step, S:
    1. Call AdaBoost.R2 with base learner, except we only allow it to change its target values. Obtain model
    2. Call base learner with source+target dataset and corresponding weights, find hypothesis for each sample
    3. Use hypothesis to calculate error for each sample
    4. Update weight vector
Output: the model with lowest error on target dataset

Notes:
We use adaboost to create an ensemble model in (1), one where each model that it contains has further tuned the values of the target weight vector (but the percentwise distribution between source and target never changes)
However, we use just the results from the base learner to determine how we should change our weight vector in (2)
Our weight vector updates in (4) are dependant on a β-value, which serves to increase the target weight percentage (as compared to source) as we go through our boosting steps S
"""

class TwoStageTrAdaBoostR2:
    def __init__(self,
                 base_estimator,
                 sample_size = None,
                 n_estimators = 50,
                 steps = 10,
                 fold = 5,
                 learning_rate = 1.,
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.steps = steps
        self.fold = fold
        self.learning_rate = learning_rate
        self.random_state = random_state

    def Step1(self):
        return 0

    def Step2(self):
        return 0
    
    def Step3(self):
        return 0
    
    def Step4(self):
        return 0


