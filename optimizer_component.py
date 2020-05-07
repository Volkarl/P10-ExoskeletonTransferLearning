import pickle
from hyperopt import fmin, tpe, Trials

def perform_hyperopt(objective, space, max_evals):
    # Initialize an empty trials database
    trials = Trials()

    total_evals = 500
    steps_before_saving = 5
    for i in range(0, total_evals, steps_before_saving):
        best = fmin(objective,
               space=space,
               algo=tpe.suggest,
               trials=trials,
               max_evals=i + steps_before_saving)
               # Runs STEPS each time it runs, saves all results, then it runs STEPS more attempts

        pickle.dump(trials, open("trials.p", "wb"))
        trials = pickle.load(open("trials.p", "rb"))

    # TODO: Add code to open a trials file and continue on, if one such file exists

    print(best)