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
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from copy import deepcopy

import Fixed_AdaBoostR2 as Ada

class TwoStageTrAdaBoostR2:
    def __init__(self,
                 create_base_estimator_fn,
                 sample_size,
                 n_estimators = 50,
                 steps = 10,
                 fold = 5,
                 learning_rate = 1.,
                 random_state = np.random.mtrand._rand,
                 start_steps = 0):
        self.create_base_estimator_fn = create_base_estimator_fn
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.steps = steps
        self.fold = fold
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.start_steps = start_steps

    def Step1(self, sample_weights, X_source, y_source, X_target, y_target):
        # Call AdaBoostR2, except dont allow it to change its source values
        # We perform cross validation for our target values, to get more learning out of it 
        kf = KFold(n_splits = self.fold)
        error = []
        target_weight = sample_weights[-self.sample_size[-1]:]
        source_weight = sample_weights[:-self.sample_size[-1]]
        for train, test in kf.split(X_target):
            cv_sample_size = [self.sample_size[0], len(train)]
            model = Ada.AdaBoostR2(self.create_base_estimator_fn, cv_sample_size, self.n_estimators, self.learning_rate, self.random_state)
            X_train = np.concatenate((X_source, X_target[train]))
            y_train = np.concatenate((y_source, y_target[train]))
            X_test = X_target[test]
            y_test = y_target[test]
            # make sure the sum weight of the target data do not change with CV's split sampling
            target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])
            model.fit(X_train, y_train, sample_weights = np.concatenate((source_weight, target_weight_train)))
            y_predict = model.predict(X_test)
            error.append(mean_absolute_error(y_predict, y_test))
        return np.array(error).mean()

    def Step2(self, sample_weights, X, y, estimator):
        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weights)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit_ada(X[bootstrap_idx], y[bootstrap_idx])
        return estimator
    
    def Step3(self, estimator, X, Y):
        y_predict = estimator.predict(X)
        y_predict = np.concatenate(y_predict, axis=0) # Flatten inner list
        error_vect = np.array([np.abs(y - y_truth) for y, y_truth in zip(y_predict, Y)])
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max
        return error_vect
    
    def Step4(self, step, sample_weights, error_vect, is_start_step = False, start_step_theoretical_sum = 0):
        # Caclulate beta value, then update target vs source weight relationship accordingly
        beta = self._beta_binary_search(step, sample_weights, error_vect, 1e-30, is_start_step, start_step_theoretical_sum)

        print(f"Beta: {beta}")
        # At the first step, beta is tiny, and basically doesn't allow source weights to change. However, target is still allowed to change individual weights through AdaboostR2

        if not step == self.steps - 1:
            sample_weights[:-self.sample_size[-1]] *= np.power(beta, (error_vect[:-self.sample_size[-1]]) * self.learning_rate)
        return sample_weights

    def _beta_binary_search(self, istep, sample_weight, error_vect, stp, is_start_step = False, start_step_theoretical_sum = 0):
        # calculate the specified sum of weight for the target data
        n_target = self.sample_size[-1]
        n_source = np.array(self.sample_size).sum() - n_target

        if is_start_step: 
            theoretical_sum = start_step_theoretical_sum
        else:
            theoretical_sum = n_target/(n_source+n_target) + istep/(self.steps-1)*(1-n_target/(n_source+n_target)) 
            # The theoretical sum can be written as basically: percent_of_dataset_that_is_target + how_far_we_are_through_total_boosting_steps * percent_of_dataset_that_is_NOT_target
            # This value increases as we get further through out total boosting steps

            # for the last iteration step, beta is 0.
            if istep == self.steps - 1:
                beta = 0.
                return beta
    
        # binary search for beta
        L = 0.
        R = 1.
        beta = (L+R)/2
        sample_weight_ = deepcopy(sample_weight)
        sample_weight_[:-n_target] *= np.power(beta, (error_vect[:-n_target]) * self.learning_rate)
        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)

        while np.abs(updated_weight_sum - theoretical_sum) > 0.01:
            if updated_weight_sum < theoretical_sum:
                R = beta - stp
                if R > L:
                    beta = (L+R)/2
                    sample_weight_ = deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep+1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break

            elif updated_weight_sum > theoretical_sum:
                L = beta + stp
                if L < R:
                    beta = (L+R)/2
                    sample_weight_ = deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    print("At step:", istep+1)
                    print("Binary search's goal not meeted! Value is set to be the available best!")
                    print("Try reducing the search interval. Current stp interval:", stp)
                    break
        return beta

    def clear_results(self):
        # Clear any previous fit results
        self.models_ = []
        self.errors_ = []
        self.sample_weights_ = []

    def adjust_source_weights(self, step, sample_weights, X, y, X_source, y_source, X_target, y_target):
        # This function is to ensure that the algorithm gets some time to shuffle weights around in our multiple source domains and attempts to find which ones are most similar to target
        # Before we run the "real" two-stage tradaboost and starts scaling up the importance of the target domain, whilst scaling down the importance of the source domain
        model = Ada.AdaBoostR2(self.create_base_estimator_fn, self.sample_size, self.n_estimators, self.learning_rate, self.random_state)
        model.fit(X, y, sample_weights)
        self.sample_weights_.append(np.copy(sample_weights)) # We do add this one to the list, purely for debugging purposes
        self.models_.append(model) 
        error = self.Step1(sample_weights, X_source, y_source, X_target, y_target)
        self.errors_.append(error)
        # We dont append the model nor the error, we let traditional two-stage tradaboost handle it, since this function is just to ensure our start sample_weights are in a good place

        # We run our beta calculations with a somewhat arbitrary value, that increases over its total start steps
        sample_weights = self.perform_second_stage_boost(0, X, y, sample_weights, True, (0.2 + step) / (self.start_steps + 0.2))


    def fit(self, X, y, sample_weights=None):
        Ada.AdaBoostR2.check_parameters(X, self.learning_rate, self.sample_size)
        sample_weights = Ada.AdaBoostR2.init_weights(sample_weights, X)
        self.clear_results()

        X_source = X[:-self.sample_size[-1]]
        y_source = y[:-self.sample_size[-1]]
        X_target = X[-self.sample_size[-1]:]
        y_target = y[-self.sample_size[-1]:]

        for start_step in range(self.start_steps):
            # This function is added by us, to help performance in the case where you are using multiple source domains, rather than one - which is what twostagetradaboost 
            # was designed for. It should only ever be run if learning from multiple source domains to one target domain
            self.adjust_source_weights(start_step, sample_weights, X, y, X_source, y_source, X_target, y_target)

        for s in range(self.steps):
            model = Ada.AdaBoostR2(self.create_base_estimator_fn, self.sample_size, self.n_estimators, self.learning_rate, self.random_state)
            model.fit(X, y, sample_weights)
            self.sample_weights_.append(np.copy(sample_weights))
            self.models_.append(model)

            error = self.Step1(sample_weights, X_source, y_source, X_target, y_target)
            self.errors_.append(error)
            sample_weights = self.perform_second_stage_boost(s, X, y, sample_weights)

            if sample_weights is None:
                break
            if np.array(error).mean() == 0:
                break
            sample_weight_sum = np.sum(sample_weights)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break
            if s < self.steps - 1:
                # Normalize
                sample_weights /= sample_weight_sum
        return self

    def perform_second_stage_boost(self, step, X, y, sample_weights, is_start_step = False, start_step_theoretical_sum = 0):
        # This is where we are going to change the source weights
        estimator = self.create_base_estimator_fn()
        estimator = self.Step2(sample_weights, X, y, estimator)
        # Update the weight vector
        error_vect = self.Step3(estimator, X, y)
        sample_weights = self.Step4(step, sample_weights, error_vect, is_start_step, start_step_theoretical_sum)
        return sample_weights

    def predict(self, X):
        # Select the model with the least CV error, let it predict
        min_error_idx = np.array(self.errors_).argmin()
        predictions = self.models_[min_error_idx].predict(X)
        return predictions

    def evaluate(self, y_predictions, y_ground_truth):
        return mean_absolute_error(y_predictions, y_ground_truth)

    def get_estimator_info(self):
        min_error_idx = np.array(self.errors_).argmin()
        best_ensemble_weights = self.models_[min_error_idx].estimator_weights_
        ensemble_weights = [model.estimator_weights_ for model in self.models_]
        return self.errors_, min_error_idx, best_ensemble_weights, ensemble_weights
        