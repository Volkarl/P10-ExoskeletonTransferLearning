import pickle
from hyperopt import fmin, tpe, Trials

def perform_hyperopt(objective, space, total_evals):
    try: 
        trials = pickle.load(open("trials.p", "rb"))
    except FileNotFoundError:
        trials = Trials() # Initialize an empty trials database

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

    print(best)