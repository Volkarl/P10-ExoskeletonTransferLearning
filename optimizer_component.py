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



    # TODO: I already have a max_evals value. Maybe split it up and perform pickling inbetween?

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

    print(best)