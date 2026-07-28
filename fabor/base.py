import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoGuideList, AutoDiagonalNormal, AutoDelta

from .fit import svi, svi_posterior

# Determine datatypes for each phenotype
# in matrix and group into
# categories
def get_pheno_categories(Y: torch.Tensor) -> dict[str, torch.Tensor]:
    # Get unique values
    P = Y.shape[1]
    num_unique_values = []
    for p in range(P):
        Y_p = Y[:, p]
        unique = torch.unique(Y_p[~torch.isnan(Y_p)])

        # Verify that categories are ordered from 0 to D - 1
        expected = torch.arange(unique.shape[0], device = Y.device).float()
        if not torch.allclose(unique, expected):
            raise ValueError(
                f"Incorrect data format in column {p}. "
                f"Categories must be written as {expected.cpu()} instead of {unique.cpu()}"
            )

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

# Mark parameters sampled from
# Dirichlet prior to be handled
# differently in SVI
def dirichlet_selector(msg):
    if msg['type'] == 'sample':
        if msg['fn'].has_rsample:
            return isinstance(msg['fn'].base_dist, dist.Dirichlet)
    return False

# Logit function with clamping
# to prevent overflow
def safe_logit(x: torch.Tensor):
    eps = torch.finfo(x.dtype).eps
    y = torch.clamp(x, eps, 1 - eps)
    return torch.logit(y)

# Base implementation
# used for all models
class Base:
    # Model structure
    def model(self):
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
                # c_p = Σ^{d}_{a=1} σ^-1(s_{pa})
                D_p = int(cat.split('_')[-1])
                alpha = torch.ones((D_p,), device = mask.device)
                s_dist = dist.Dirichlet(alpha).expand([Q]).to_event(1)
                s = pyro.sample(f"s_{cat}", s_dist)
                c = torch.cumsum(s[..., :-1], dim = -1)

                # q_{np} = σ(c_p - σ^-1(W_{np}))
                # θ_{npd} = q_{npd} - q_{np(d - 1)}
                # Y_{np} ~ Cat(θ_{np})
                obs_dist = dist.OrderedLogistic(safe_logit(W[:, pheno_mask]), safe_logit(c)).mask(mask[:, pheno_mask])

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
        # Get phenotype categories
        Y = Y.float()
        pheno_cat = get_pheno_categories(Y)

        # Add missingness to data if enabled
        if mnar_by_data:
            num_pheno = Y.shape[1]
            M = (~torch.isnan(Y)).float()
            Y = torch.cat([Y, M], axis = 1)
            pheno_cat.setdefault('binary', torch.tensor([], dtype = int))
            new_indices = torch.arange(num_pheno, num_pheno * 2)
            pheno_cat['binary'] = torch.cat([pheno_cat['binary'], new_indices])

        # Initialize guide for SVI
        guide = AutoGuideList(self.model)
        # For non-Dirichlet variables
        guide.append(AutoDiagonalNormal(poutine.block(self.model, hide_fn = dirichlet_selector)))
        # For Dirichlet variables
        guide.append(AutoDelta(poutine.block(self.model, expose_fn = dirichlet_selector)))

        # Compile input parameters
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

        # Return fitted values for non-Bayesian parameters
        trace = poutine.trace(self.model).get_trace(**params)
        for name, node in trace.nodes.items():
            if node['type'] == 'param':
                stats[name] = node['value']

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
        K: int,
        max_iterations: int = 1000,
        posterior_samples: int = 50,
        detailed: bool = False
    ):
        # Get phenotype categories
        pheno_cat = get_pheno_categories(Y)

        # Get missingness mask
        M = (~torch.isnan(Y)).float()

        # Initialize guide for SVI
        guide = AutoGuideList(self.model)
        # For non-Dirichlet variables
        guide.append(AutoDiagonalNormal(poutine.block(self.model, hide_fn = dirichlet_selector)))
         # For Dirichlet variables
        guide.append(AutoDelta(poutine.block(self.model, expose_fn = dirichlet_selector)))

        # Compile input parameters
        params = {
            "Y": Y,
            "M": M,
            "K": K,
            "pheno_cat": pheno_cat
        }

        # Run SVI and return average of samples
        # from predicted posterior
        loss = svi(self.model, guide, params, max_iterations = max_iterations, print_every = 100)
        stats = svi_posterior(self.model, guide, params, num_samples = posterior_samples)

        # Simplify posterior stats if needed
        if not detailed:
            stats = {
                key: value['mean'] for key, value in stats.items()
            }

        # Return fitted values for non-Bayesian parameters
        trace = poutine.trace(self.model).get_trace(**params)
        for name, node in trace.nodes.items():
            if node['type'] == 'param':
                stats[name] = node['value']

        return stats, pheno_cat, loss
