<div align="center">
    <img src="assets/logo.svg" alt="FABOr logo" width="300px"/>
</div>

# FABOr: Factor Analysis of Binary and Ordinal data

FABOr is a Bayesian framework for matrix factorization that models the low-rank matrices as latent variables and is designed to analyze the underlying relationships in binary and/or ordinal data. It accepts missing entries in the input dataset and can model structured missingness (i.e., missing not at random data) if required. It was initially designed for analyzing phenotypes from clinical and self-reported medical questionnaires but is applicable to any type of binary/ordinal data. For more information on how the framework is designed, please refer to our recent preprint on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.15.733660v1).

## Installation

FABOr is released as a Python package. To install the latest version of this package, run:
```
pip install git+https://github.com/daklab/fabor
```

To install a development copy of the package that can be modified as needed, start by cloning the repo, then run:
```
pip install -e .
```

The package requires `pyro` and `pytorch`, which are automatically installed as dependencies if your Python setup does not have it.

## Usage

FABOr currently includes five models: `Normal`, `DirichletBeta`, `DirichletNormal`, `Lognormal`, and `NormalMNAR`. To start, import one of these models and call `fit()` like so:
```
import fabor

model = fabor.Normal()
stats, pheno_cat, loss = model.fit(Y, K)
```
- `Y`: the dataset of interest. Observed values have to be encoded starting from 0, such that binary values are represented using 0 and 1, and ordinal values are represented from 0 to `D - 1` (where `D` is the number of categories for the phenotype). Missing values have to be encoded as `nan`.
- `K` is the number of latent factors to fit the model.

The results will then return the following:
- `stats`: a dictionary of latent and deterministic variables from the fitted posterior distribution.
- `pheno_cat`: a dictionary of phenotype categories (e.g., binary, 4-scale ordinal, 2-scale ordinal, etc.) and the corresponding column indices in `Y`. 
- `loss`: a list of loss values across the number of iterations used to fit the model.

Optional arguments to `fit()` include:
- `mnar_by_data`: whether to enable the MNAR by Data extension and concatenate the missingness pattern to the observed phenotypes (not applicable for `NormalMNAR`). Default is `False`.
- `max_iterations`: the max number of iterations to run during optimization. Default is `1000`.
- `posterior_samples`: the number of samples to draw from the fitted posterior distribution. Default is `50`.
- `detailed`: whether to include additional information about the variance and confidence intervals with `stats`. Default is `False`.

In general, the `Normal` or `NormalMNAR` model is a good starting point for analysis, while the other models can be used to test whether other Bayesian priors make the latent factors more interpretable for different use cases. The framework is designed such that any continuous-valued Bayesian prior can be applied to the low-rank matrices, as long as there exists a differentiable function to transform the product into the unit interval.   

For more information on how to use FABOr and customize the framework to your needs, please check the [examples](examples/) directory.

## Citing FABOr
If you find this framework useful in your research, please cite our paper:

N Shashaank and DA Knowles. 2026. Bayesian Factor Analysis for Binary and Ordinal Phenotypes with Missingness. bioRxiv. https://doi.org/10.64898/2026.07.15.733660

```
@article{
    shashaank2026bayesian,
    title={Bayesian Factor Analysis for Binary and Ordinal Phenotypes with Missingness},
    author={Shashaank, N. and Knowles, David A.},
    journal={bioRxiv},
    year={2026},
    doi={10.64898/2026.07.15.733660}
}
```
