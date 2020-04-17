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

import numpy as np
from copy import deepcopy


class AdaBoostR2:
    def __init__(self,
                 base_estimator,
                 sample_size,
                 n_estimators = 50,
                 learning_rate = 1.,
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
    def check_parameters(self, X):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

    def init_weights(self, sample_weights, X):
        if sample_weights is None:
            # Initialize weights to 1 / n_samples
            sample_weights = np.empty(X.shape[0], dtype=np.float64)
            sample_weights[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weights = sample_weights / sample_weights.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weights.sum() <= 0:
                raise ValueError("Attempting to fit with a non-positive weighted number of samples.")

    def clear_results(self):
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

    def step_1(self, estimator, X, y, sample_weights): 
        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement, ergo bootstrap sampling, ergo bagging
        cdf = np.cumsum(sample_weights)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx]) 
        # We only fit on the bootstrapped sample, however, later we predict on the entire sample - a way to avoid overfitting
        self.estimators_.append(estimator)  # add the fitted estimator

    def step_2(self, estimator, X, y):
        y_predict = estimator.predict(X)          ### TODO: FIX TO KERAS
        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max
        return error_vect
    
    def step_3(self, sample_weights, error_vect):
        # Calculate the average loss
        estimator_error = (sample_weights * error_vect).sum()
        return estimator_error

    def step_4(self, estimator_error):
        beta = estimator_error / (1. - estimator_error)
        if beta < 1e-308: beta = 1e-308 # avoid overflow of np.log(1. / beta)
        estimator_weight = self.learning_rate * np.log(1. / beta)

        return beta, estimator_weight

    def step_5(self, beta, sample_weights, boost_iteration, error_vect):
        # Boost weight using AdaBoost.R2 alg, except the weight of the source data, which is not allowed to change
        source_weight_sum = np.sum(sample_weights[:-self.sample_size[-1]]) / np.sum(sample_weights)
        target_weight_sum = np.sum(sample_weights[-self.sample_size[-1]:]) / np.sum(sample_weights)

        if not boost_iteration == self.n_estimators - 1:
            sample_weights[-self.sample_size[-1]:] *= np.power(beta, (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)
            # make the sum weight of the source data not changing
            source_weight_sum_new = np.sum(sample_weights[:-self.sample_size[-1]]) / np.sum(sample_weights)
            target_weight_sum_new = np.sum(sample_weights[-self.sample_size[-1]:]) / np.sum(sample_weights)
            if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                sample_weights[:-self.sample_size[-1]] = sample_weights[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
                sample_weights[-self.sample_size[-1]:] = sample_weights[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new

        return sample_weights

    def perform_boost(self, boost_iteration, X, y, sample_weights):
        estimator = deepcopy(self.base_estimator)
        self.step_1(estimator, X, y, sample_weights)
        error_vect = self.step_2(estimator, X, y)
        estimator_error = self.step_3(sample_weights, error_vect)

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weights, 1., 0.
        elif estimator_error >= 0.5:
            # Discard current estimator if error is larger than 0.5, so long as our current estimator is not the only one
            if len(self.estimators_) > 1: self.estimators_.pop(-1)
            return None, None, None

        beta, estimator_weight = self.step_4(estimator_error)
        sample_weights = self.step_5(beta, sample_weights, boost_iteration, error_vect)

        return sample_weights, estimator_weight, estimator_error

    def fit(self, X, y, sample_weight=None):
        self.check_parameters(X)
        self.init_weights(sample_weight, X)
        self.clear_results()

        for ada_iteration in range(self.n_estimators): # this for loop is sequential and does not support parallel(revison is needed if making parallel)
            sample_weight, estimator_weight, estimator_error = self.perform_boost(ada_iteration, X, y, sample_weight)
            # Early termination
            if sample_weight is None: # When estimator error gets too large, we terminate
                break

            self.estimator_weights_[ada_iteration] = estimator_weight
            self.estimator_errors_[ada_iteration] = estimator_error
            # Stop if error is zero, ergo we made a perfect learner. This will never happen unless we have a tiny or a super simple dataset
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break
            if ada_iteration < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self

    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([est.predict(X) for est in self.estimators_]).T

        # Sort the predictions, such that each sample in X, now has a sorted array of results, lowest to highest, which we use for selecting the median value
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]
