import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK }
    # Replace this with DataCollect_component + CNN_Baseline + Evaluation_component
    # Remember execution time as well

# Initialize an empty trials database
trials = Trials()

#space = { "KERNEL_SIZE" : hp.uniform("KERNEL_SIZE", 2, 100),
#          "FILTERS" : hp.uniform("FILTERS", 2, 100),
#          "OPTIMIZER" : hp.choice("OPTIMIZER", ["adadelta", "adam", "rmsprop"]) }

space = hp.uniform("x", 1, 100)
# print hyperopt.pyll.stochastic.sample(space)

# Define search spaces
# MAKE SOME MORE

# Try if my search space works: with pyll
# If you like, you can evaluate a sample space by sampling from it.
# import hyperopt.pyll.stochastic
# print hyperopt.pyll.stochastic.sample(space)


# Perform 100 evaluations on the search space
best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=100)

# The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method
pickle.dump(trials, open("trials.p", "wb"))
trials = pickle.load(open("trials.p", "rb"))

# Perform an additional 100 evaluations
# Note that max_evals is set to 200 because 100 entries already exist in the database
best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=200)

print(best)