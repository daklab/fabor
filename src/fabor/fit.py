import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive

# Stochastic variational inference for probabilistic methods
# using ELBO
def svi(model, guide, params, iterations = 500, print_every = 1000):
    # Initialize SVI objects
    adam = pyro.optim.Adam({'lr': 0.1})
    svi = SVI(model, guide, adam, loss = Trace_ELBO())

    # Run SVI optimization
    attempts = 0
    while attempts < 5:
        try:
            pyro.clear_param_store()
            losses = []
            for i in range(iterations):
                loss = svi.step(**params)
                losses.append(loss)
                if (i + 1) % print_every == 0:
                    print("Iteration: %i ELBO: %.4g" % (i + 1, loss))
            break

        except Exception as e:
            print(f"Ran into exception {e}")
            attempts += 1
        
    if attempts == 5:
        raise ValueError("Unable to fit model")
        
    return np.array(losses)

# Sample from posterior distribution
# after stochastic variational inference
def svi_posterior(model, guide, data, num_samples = 100):
    # Get samples
    predictive = Predictive(model, guide = guide, num_samples = num_samples)
    samples = predictive(**data)

    # Get mean and standard deviation of samples
    posterior_stats = { k : {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            } for k, v in samples.items() }

    return posterior_stats
