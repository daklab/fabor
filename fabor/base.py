import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal

from .fit import svi, svi_posterior

# Determine datatypes for each phenotype
# in matrix and group into
# categories
def _get_pheno_categories(Y: torch.Tensor) -> dict[str, torch.Tensor]:
    # Get unique values
    P = Y.shape[1]
    num_unique_values = []
    for p in range(P):
        Y_p = Y[:, p]
        unique = torch.unique(Y_p[~torch.isnan(Y_p)])
        num_unique_values.append(unique.shape[0])
    num_unique_values = torch.tensor(num_unique_values)

    # Group phenotypes which have same 
    # number of unique values
    category_counts = torch.unique(num_unique_values)
    pheno_cat = {}
    for count in category_counts:
        pheno_counts = torch.argwhere(num_unique_values == count).squeeze()
        if count == 2:
            pheno_cat['binary'] = pheno_counts
        else:
            pheno_cat[f'ordinal_{count}'] = pheno_counts
    
    return pheno_cat

# Base implementation
# used for all models
class Base:
    # Model structure
    def model():
        raise NotImplementedError

    # Likelihood function used
    # in all models
    def likelihood(
        self,
        W: torch.Tensor, 
        Y: torch.Tensor,
        pheno_cat: dict[str, torch.Tensor]
    ):
        # Get observed indices
        N = Y.shape[0]
        mask = ~torch.isnan(Y)

        # Process each phenotype category
        for cat in pheno_cat:
            # Create mask for category
            pheno_mask = pheno_cat[cat]
            Q = pheno_mask.shape[0]

            # Binary phenotypes
            if cat == 'binary':
                # Y_{np} ~ Bern(W_{np})
                obs_dist = dist.Bernoulli(W[:, pheno_mask]).mask(mask[:, pheno_mask])
            else: # Ordinal phenotypes
                # s_p ~ Dirichlet(1 / D_p)
                D_p = int(cat.split('_')[-1])
                alpha = torch.ones((D_p,), device = mask.device)
                s_dist = dist.Dirichlet(alpha).expand([Q]).to_event(1)
                s = pyro.sample(f"s_{cat}", s_dist)

                # c_p = Σ^{d}_{a=1} s_{pa}
                c = torch.cumsum(s[..., :-1], dim = -1)

                # q_{np} = 1 / 2(c_p - W_{np}) + 1 / 2
                q = 0.5 * (c - W[:, pheno_mask].unsqueeze(-1)) + 0.5

                # θ_{npd} = q_{npd} - q_{np(d - 1)}
                theta = torch.zeros((N,) + s.shape, device = mask.device)
                theta[..., 0] = q[..., 0]
                theta[..., 1:-1] = q[..., 1:] - q[..., :-1]
                theta[..., -1] = 1 - q[..., -1]

                # Y_{np} ~ Cat(θ_{np})
                obs_dist = dist.Categorical(theta).mask(mask[:, pheno_mask])
            
        # Likelihood of observed variable
        with pyro.plate(f"Q_{cat}", Q):
            with pyro.plate(f"N_{cat}", N):
                obs = pyro.sample(f"obs_{cat}", obs_dist, obs = torch.nan_to_num(Y[:, pheno_mask]))
    
    # Fit model to given data
    def fit(
        self, 
        Y: torch.Tensor, 
        K: int, 
        mnar_by_data: bool = False,
        max_iterations: int = 1000,
        posterior_samples: int = 50,
        detailed: bool = False
    ):
        # Add missingness to data if enabled
        Y = Y.float()
        if mnar_by_data:
            mask = (~torch.isnan(Y)).float()
            Y = torch.cat(torch.cat([Y, mask], axis = 1))

        # Get phenotype categories
        pheno_cat = _get_pheno_categories(Y)

        # Initialize setup for SVI
        guide = AutoDiagonalNormal(self.model)
        params = {
            "Y": Y,
            "K": K,
            "pheno_cat": pheno_cat
        }

        # Run SVI and return average of samples 
        # from predicted posterior
        loss = svi(self.model, guide, params, max_iterations = max_iterations)
        stats = svi_posterior(self.model, guide, params, num_samples = posterior_samples)

        # Simplify posterior stats if needed
        if not detailed:
            stats = {
                key: value['mean'] for key, value in stats.items()
            }

        return stats, pheno_cat, loss

# Base implementation used for MNAR models
class BaseMNAR(Base):
    # Likelihood function used
    # in MNAR models
    def likelihood(
        self,
        W: torch.Tensor,
        sigma_W: torch.Tensor, 
        Y: torch.Tensor,
        M: torch.Tensor,
        pheno_cat: dict[str, torch.Tensor]
    ):
        super().likelihood(W, Y, pheno_cat)
        N, P = M.shape
        with pyro.plate("P", P):
            with pyro.plate("N", N):
                obs_m = pyro.sample('obs_m', dist.Bernoulli(sigma_W), obs = M)

    # Fit model to given data
    def fit(
        self, 
        Y: torch.Tensor,
        M: torch.Tensor, 
        K: int, 
    ):
        # Get phenotype categories
        pheno_cat = _get_pheno_categories(Y)

        # Initialize setup for SVI
        guide = AutoDiagonalNormal(self.model)
        params = {
            "Y": Y,
            "M": M,
            "K": K,
            "pheno_cat": pheno_cat
        }

        # Run SVI and return average of samples 
        # from predicted posterior
        loss = svi(self.model, guide, params, iterations = 1000, print_every = 100)
        samples = svi_posterior(self.model, guide, params, num_samples = 50)

        return samples, loss