import pickle
from hyperopt import fmin, tpe, Trials

def perform_hyperopt(objective, space, max_evals):
    # Initialize an empty trials database
    trials = Trials()

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