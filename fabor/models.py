import torch
import pyro
import pyro.distributions as dist

from .base import Base, BaseMNAR

# Normal FA model
class Normal(Base):
    # Model structure
    def model(
        self,
        Y: torch.Tensor,
        K: int,
        pheno_cat: dict[str, torch.Tensor]
    ):
        N, P = Y.shape

        # U ~ N(0, 1)
        U_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([N, K]).to_event(2)
        U = pyro.sample("U", U_dist)

        # V ~ N(0, 1)
        V_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([P, K]).to_event(2)
        V = pyro.sample("V", V_dist)

        # f(U, V) = σ(UV^T)
        W = pyro.deterministic("W", torch.sigmoid(U @ V.T))

        self.likelihood(W, Y, pheno_cat)

# Dirichlet-Beta FA model
class DirichletBeta(Base):
    # Model structure
    def model(
        self,
        Y: torch.Tensor,
        K: int,
        pheno_cat: dict[str, torch.Tensor]
    ):
        N, P = Y.shape

        # U ~ Dirichlet(1)
        U_dist = dist.Dirichlet(torch.ones(K, device = Y.device)).expand([N]).to_event(1)
        U = pyro.sample("U", U_dist)

        # V ~ Beta(2, 2)
        V_dist = dist.Beta(torch.tensor(2., device = Y.device), 2.).expand([P, K]).to_event(2)
        V = pyro.sample("V", V_dist)

        # f(U, V) = UV^T
        W = pyro.deterministic("W", U @ V.T)

        self.likelihood(W, Y, pheno_cat)

# Dirichlet-Normal FA model
class DirichletNormal(Base):
    # Model structure
    def model(
        self,
        Y: torch.Tensor,
        K: int,
        pheno_cat: dict[str, torch.Tensor]
    ):
        N, P = Y.shape

        # U ~ Dirichlet(1)
        U_dist = dist.Dirichlet(torch.ones(K, device = Y.device)).expand([N]).to_event(1)
        U = pyro.sample("U", U_dist)

        # V ~ N(0, 1)
        V_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([P, K]).to_event(2)
        V = pyro.sample("V", V_dist)

        # f(U, V) = U σ(V^T)
        W = pyro.deterministic("W", U @ torch.sigmoid(V.T))

        self.likelihood(W, Y, pheno_cat)

# Lognormal FA model
class Lognormal(Base):
    # Model structure
    def model(
        self,
        Y: torch.Tensor,
        K: int,
        pheno_cat: dict[str, torch.Tensor]
    ):
        N, P = Y.shape

        # U ~ Lognormal(0, 1)
        U_dist = dist.LogNormal(torch.tensor(0., device = Y.device), 1.).expand([N, K]).to_event(2)
        U = pyro.sample("U", U_dist)

        # V ~ Lognormal(0, 1)
        V_dist = dist.LogNormal(torch.tensor(0., device = Y.device), 1.).expand([P, K]).to_event(2)
        V = pyro.sample("V", V_dist)

        # f(U, V) = 1 - exp(-UV^T)
        W = pyro.deterministic("W", 1 - torch.exp(-(U @ V.T)))

        self.likelihood(W, Y, pheno_cat)

# Normal MNAR FA model
class NormalMNAR(BaseMNAR):
    # Model structure
    def model(
        self,
        Y: torch.Tensor,
        M: torch.Tensor,
        K: int,
        pheno_cat: dict[str, torch.Tensor]
    ):
        N, P = Y.shape

        # μ_U ~ N(0, 1)
        mu_U_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([N, K]).to_event(2)
        mu_U = pyro.sample("mu_U", mu_U_dist)

        # μ_V ~ N(0, 1)
        mu_V_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([P, K]).to_event(2)
        mu_V = pyro.sample("mu_V", mu_V_dist)

        # Σ_U ~ N(0, 1)
        sigma_U_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([N, K]).to_event(2)
        sigma_U = pyro.sample("sigma_U", sigma_U_dist)

        # Σ_V ~ N(0, 1)
        sigma_V_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([P, K]).to_event(2)
        sigma_V = pyro.sample("sigma_V", sigma_V_dist)

        # U_unit ~ N(0, 1)
        # U = μ_U + U_unit * exp(-Σ_U)
        U_unit_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([N, K]).to_event(2)
        U_unit = pyro.sample("U_unit", U_unit_dist)
        U = pyro.deterministic("U", mu_U + U_unit * torch.exp(-sigma_U))

        # V_unit ~ N(0, 1)
        # V = μ_V + V_unit * exp(-Σ_V)
        V_unit_dist = dist.Normal(torch.tensor(0., device = Y.device), 1.).expand([P, K]).to_event(2)
        V_unit = pyro.sample("V_unit", V_unit_dist)
        V = pyro.deterministic("V", mu_V + V_unit * torch.exp(-sigma_V))

        # f(U, V) = σ(UV^T)
        W = pyro.deterministic("W", torch.sigmoid(U @ V.T))
        sigma_W = pyro.deterministic("sigma_W", torch.sigmoid(sigma_U @ sigma_V.T))

        self.likelihood(W, sigma_W, Y, M, pheno_cat)
