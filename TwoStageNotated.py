"""
TwoStageTrAdaBoostR2 algorithm

based on algorithm 3 in paper "Boosting for Regression Transfer".

"""

import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

################################################################################
## the second stage
################################################################################
class Stage2_TrAdaBoostR2: #The class for stage 2
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth=4),
                 sample_size = None,
                 n_estimators = 50,
                 learning_rate = 1.,
                 loss = 'linear',
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator #Variables are save as in the TwoStageTrAdaBoostR2 class
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None):
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        for iboost in range(self.n_estimators): # this for loop is sequential and does not support parallel(revison is needed if making parallel)
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._stage2_adaboostR2(
                    iboost,
                    X, y,
                    sample_weight)
            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self


    def _stage2_adaboostR2(self, iboost, X, y, sample_weight):

        estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        self.estimators_.append(estimator)  # add the fitted estimator

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # avoid overflow of np.log(1. / beta)
        if beta < 1e-308:
            beta = 1e-308
        estimator_weight = self.learning_rate * np.log(1. / beta)

        # Boost weight using AdaBoost.R2 alg except the weight of the source data
        # the weight of the source data are remained
        source_weight_sum= np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
        target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)

        if not iboost == self.n_estimators - 1:
            sample_weight[-self.sample_size[-1]:] *= np.power(
                    beta,
                    (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)
            # make the sum weight of the source data not changing
            source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
                sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new

        return sample_weight, estimator_weight, estimator_error


    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([
                est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]


################################################################################
## the whole two stages
################################################################################
class TwoStageTrAdaBoostR2: #The main class for the program
    def __init__(self, 
                 base_estimator = DecisionTreeRegressor(max_depth=4), #default base learner
                 sample_size = None, #default sample size
                 n_estimators = 50,
                 steps = 10,
                 fold = 5,
                 learning_rate = 1.,
                 loss = 'linear',
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator #Base_estimator is the base learner we use for our model
        self.sample_size = sample_size #Sample_size is how much data we have, [source_length, target_length]
        self.n_estimators = n_estimators #n_estimators is the max itterations of the adaboost.R2 algorithm (stage 2)
        self.steps = steps #steps is how many itterations we have of the main twostagetradaboost loop
        self.fold = fold #fold is the amount of k-fold cross validation we have
        self.learning_rate = learning_rate #learning_rate is multiplied on weights when we update them.. "Learning rate shrinks the contribution of each regressor by learning_rate. There is a trade-off between learning_rate and n_estimators."??? (learning_rate should be bewteen 0-1 and if it is lower then n_estimators should be higher to allow it to continuously improve, some suggest learning_rate of 0.1 or less)
        self.loss = loss #Which type of loss we use. Default is linear and should probably always be (Paper found it best).
        self.random_state = random_state #random_state is the seed for the random generator


    def fit(self, X, y, sample_weight=None): #The primary fit function called when using the program
        # Check parameters
        if self.learning_rate <= 0: #We multiply our weights with the learning_rate, so it has to be positive.
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64) #initialize an empty array with length of X
            sample_weight[:] = 1. / X.shape[0] #make each weight 1 divided by amount of data
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

            #TODO Try to change all these to just [sample_size[0]] or [sample_size[1]]
        X_source = X[:-self.sample_size[-1]] #X_source takes the amount of data from X corosponding to the first element of sample_size
        y_source = y[:-self.sample_size[-1]] #y_source takes the amount of data from y corosponding to the first element of sample_size
        X_target = X[-self.sample_size[-1]:] #X_target takes the amount of data from X corosponding to the second element of sample_size
        y_target = y[-self.sample_size[-1]:] #y_target takes the amount of data from y corosponding to the second element of sample_size

        self.models_ = [] #initialize our array of models we will use to keep our modles to find them in the end.
        self.errors_ = [] #Initialize array of errors to find the best model in the end.
        for istep in range(self.steps): #The main iterating loop of the twostagetradaboost algorithm.
            model = Stage2_TrAdaBoostR2(self.base_estimator, #Initialize our model with all relevant variables.
                                        sample_size = self.sample_size,
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, loss = self.loss,
                                        random_state = self.random_state)
            model.fit(X, y, sample_weight = sample_weight) #Fit the model using the adaboost.R2 fit function (stage 2)
            self.models_.append(model) #Append the fitted model into the array of models
            # cv training
            kf = KFold(n_splits = self.fold) #Provides the indices to later split the data set into train and test for cross validation
            error = [] #Array for the errors from the cross validation?
            target_weight = sample_weight[-self.sample_size[-1]:] #Sets target_weights to the last x weights from sample_weight, where x is the first index of sample_size
            source_weight = sample_weight[:-self.sample_size[-1]] #Sets source_weights to the first y weights from sample_weight, where y is the first index of sample_size
            for train, test in kf.split(X_target): #For loop performing cross validation where it keeps changing which k_fold is the test dataset until all has been test.
                sample_size = [self.sample_size[0], len(train)] #Sets the sample_size array within the cross validation loop
                model = Stage2_TrAdaBoostR2(self.base_estimator, #Initialize the model for cross validation with relevant variables
                                        sample_size = sample_size,
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, loss = self.loss,
                                        random_state = self.random_state)
                X_train = np.concatenate((X_source, X_target[train])) #Creates the X_train set by combining the full X_source set with the train part of X_target
                y_train = np.concatenate((y_source, y_target[train])) #Y_train by combining y_source with train part of y_target
                X_test = X_target[test] #Takes the test part of X_target
                y_test = y_target[test] #The test part of y_target
                # make sure the sum weight of the target data do not change with CV's split sampling
                target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train]) #Normalizes the weight such that the sum stays the same in each fold
                model.fit(X_train, y_train, sample_weight = np.concatenate((source_weight, target_weight_train))) #Fit the model using the modified adaboost.r2 fit function
                y_predict = model.predict(X_test) #Predict model with adaboost.R2 predict function.
                error.append(mean_squared_error(y_predict, y_test)) #Calculate the MSE and add it to the error array

            self.errors_.append(np.array(error).mean()) #Add the mean of the errors from the cross validation to the final errors array

            sample_weight = self._twostage_adaboostR2(istep, X, y, sample_weight) #Update the sample_weight 

            if sample_weight is None: #Is this even usefull?
                break
            if np.array(error).mean() == 0: #Is this even usefull?
                break

            sample_weight_sum = np.sum(sample_weight) #Sum up all the sample weights

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0: #Is this even usefull?
                break

            if istep < self.steps - 1:
                # Normalize
                sample_weight /= sample_weight_sum #Normalize sample_weight such that the sum of all weights is 1
        return self


    def _twostage_adaboostR2(self, istep, X, y, sample_weight): #Used to update the sample_weight
        # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)
        estimator = copy.deepcopy(self.base_estimator) #Creates a copy of the base estimater with all the variables

        """
        Comment on how the next part works..
        We want to create random bags of indexes
        
        We want the distributions to weighted wrt. the size difference between weights
        So if one sample is clearly more important than the others, it is more likely to be included in our bag

        We cumsum to find the cumulative sum of weight values, such that we can see the value difference from
        each weight compared to the previous one, so [1, 0.5, 2, 1] becomes [1, 1.5, 3.5, 4.5]

        Now, we create a random distribution of numbers between 0 and the max value, 4.5, we will have 
        more of our random values be placed between index 2 and index 3, since index 3 in our original list
        contained the most important sample.

        So now, we use searchsorted, to find the indexes that our random distributions are placed within.
        There you go, now we have a weighted distribution of random values, that make up our bag of sample
        """
        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight) #Does the cumulative sum of the sample_weight array.
        cdf /= cdf[-1] #Didvides each element of cdf with the last element (the sum of sample_weight) this means all variables are between 0-1
        uniform_samples = self.random_state.random_sample(X.shape[0]) #Creates an array of random variables, with length of our X
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right') #searchsorted gives the index where a given value should be inserted to keep the list sorted. So we get an array of indecies where our ranomd numbers should be inserted.
        # searchsorted returns a scalar or array
        bootstrap_idx = np.array(bootstrap_idx, copy=False) #converts bootstrap_idx to an array incase it was a scalar

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx]) #Uses the sklearn fit function with the weighted sampling of weights from before as input
        y_predict = estimator.predict(X) #sklearns predict function to get a prediction out.


        error_vect = np.abs(y_predict - y) #An array of the error of all predections (distance between prediction and actual value)
        error_max = error_vect.max() #The highest error value

        if error_max != 0.:
            error_vect /= error_max #Normalize all error values to be between 0-1

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Update the weight vector
        beta = self._beta_binary_search(istep, sample_weight, error_vect, stp = 1e-30) #Find the beta value to update the weights according to the calculations in the paper

        if not istep == self.steps - 1: #Check if we are at the final step TODO start with doing this?
            sample_weight[:-self.sample_size[-1]] *= np.power(
                    beta,
                    (error_vect[:-self.sample_size[-1]]) * self.learning_rate) #Updates the sample weights using the formula in the paper
        return sample_weight


    def _beta_binary_search(self, istep, sample_weight, error_vect, stp):
        # calculate the specified sum of weight for the target data
        n_target = self.sample_size[-1]
        n_source = np.array(self.sample_size).sum() - n_target
        theoretical_sum = n_target/(n_source+n_target) + istep/(self.steps-1)*(1-n_target/(n_source+n_target))
        # for the last iteration step, beta is 0.
        if istep == self.steps - 1:
            beta = 0.
            return beta
        # binary search for beta
        L = 0.
        R = 1.
        beta = (L+R)/2
        sample_weight_ = copy.deepcopy(sample_weight)
        sample_weight_[:-n_target] *= np.power(
                    beta,
                    (error_vect[:-n_target]) * self.learning_rate)
        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)

        while np.abs(updated_weight_sum - theoretical_sum) > 0.01:
            if updated_weight_sum < theoretical_sum:
                R = beta - stp
                if R > L:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
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
                    sample_weight_ = copy.deepcopy(sample_weight)
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


    def predict(self, X):
        # select the model with the least CV error
        fmodel = self.models_[np.array(self.errors_).argmin()]
        predictions = fmodel.predict(X)
        return predictions