import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive

# Stochastic variational inference
def svi(model, guide, params, max_iterations = 1000, patience = 100, print_every = 100):
    # Initialize SVI objects
    adam = pyro.optim.Adam({'lr': 0.1})
    svi = SVI(model, guide, adam, loss = Trace_ELBO())

    # Run SVI optimization
    attempts = 0
    while attempts < 5:
        try:
            patience_counter = patience
            losses = []
            best_loss = np.inf

            pyro.clear_param_store()
            for i in range(max_iterations):
                loss = svi.step(**params)
                losses.append(loss)

                # Stop training early if patience reached
                if loss < best_loss:
                    patience_counter = patience
                else:
                    patience_counter -= 1
                    if patience_counter == 0:
                        print(f"Early stopping at Iteration {i + 1}")
                        break

                # Print ELBO to track progress
                if (i + 1) % print_every == 0:
                    print(f"Iteration: {i + 1} ELBO: {loss:.4g}")
            break

        except Exception as e:
            print(f"Ran into exception {e}")
            attempts += 1
        
    if attempts == 5:
        raise ValueError("Unable to fit model")
        
    return np.array(losses)

# Sample from posterior distribution
# after stochastic variational inference
def svi_posterior(model, guide, data, num_samples = 50):
    # Get samples
    predictive = Predictive(model, guide = guide, num_samples = num_samples)
    samples = predictive(**data)

    # Get mean and standard deviation of latent samples
    posterior_stats = { 
        k : {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim = 0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim = 0)[0],
        } for k, v in samples.items() if 'obs' not in k
    }

    return posterior_stats
