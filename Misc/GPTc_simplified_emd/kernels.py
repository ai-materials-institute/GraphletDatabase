"""
GP Kernel and Model Definitions.

Contains:
- SgEMDKernelV2: Histogram EMD + RBF Symmetry dual kernel.
- SAAS prior utilities for lengthscale regularization.
- SgEmdExactGPModelV2: Exact GP for regression.
- EmdSgGpClassificationModel: Variational GP for binary classification.
"""

import math
import torch
import ot
from torch import Tensor
from torch.nn import Parameter

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel
from gpytorch.priors import Prior, HalfCauchyPrior
from gpytorch.constraints import Positive, Interval
from gpytorch.utils.memoize import cached


# ═══════════════════════════════════════════════════════════════════════════════
# SgEMDKernelV2 — Histogram EMD × RBF Symmetry kernel
# ═══════════════════════════════════════════════════════════════════════════════

class SgEMDKernelV2(Kernel):

    def __init__(
        self,
        n_sg,
        n_histogram=0,
        data_shape=None,
        n_batches=None,
        weights_prior: Prior = None,
        **kwargs,
    ):
        super(SgEMDKernelV2, self).__init__(has_lengthscale=False, **kwargs)
        self.n_sg = n_sg
        if n_sg > 0:
            self.sg_kernel = ScaleKernel(RBFKernel(ard_num_dims=self.n_sg))
        self.n_batches = n_batches
        self.data_shape = data_shape
        self.n_histogram = n_histogram
        if self.n_histogram > 0:
            self.raw_weights = torch.nn.Parameter(torch.rand(n_histogram) * 0.1)
            self.raw_lengthscales = torch.nn.Parameter(torch.rand(n_histogram) * 0.1)
            if weights_prior is not None:
                self.register_prior("weights_prior", weights_prior, lambda: self.weights, "weights")
            self.register_constraint("raw_weights", Positive())
            self.register_constraint("raw_lengthscales", Positive())

    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        self._set_weights(value)

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))

    @property
    def lengthscales(self):
        return self.raw_lengthscales_constraint.transform(self.raw_lengthscales)

    @lengthscales.setter
    def lengthscales(self, value):
        self._set_lengthscales(value)

    def _set_lengthscales(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_lengthscales=self.raw_lengthscales_constraint.inverse_transform(value))

    @cached
    def compute_emd_more_memory_efficient(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
    ) -> torch.Tensor:
        """Precompute EMD distances between all pairs in X1 and X2."""
        X1 = X1.reshape(-1, *self.data_shape)
        X2 = X2.reshape(-1, *self.data_shape)
        X1_X2 = torch.equal(X1, X2)

        n1_samples = X1.shape[0]
        n2_samples = X2.shape[0]
        n_histograms = X1.shape[1]

        def get_valid_bin_lengths(h_array):
            bin_lengths = []
            for h in range(n_histograms):
                valid_bins = torch.sum(
                    (h_array[:, h, :, 0] != -1.0) | (h_array[:, h, :, 1] != -1.0), axis=1
                )
                bin_lengths.append(int(valid_bins.max()))
            return bin_lengths

        bin_lengths_1 = get_valid_bin_lengths(X1)
        bin_lengths_2 = get_valid_bin_lengths(X2)
        if bin_lengths_1 != bin_lengths_2:
            raise ValueError("Inconsistent number of histogram bins")
        bin_lengths = bin_lengths_1

        X1_bins = X1[:, :, :, 0]
        X1_counts = X1[:, :, :, 1]
        X2_bins = X2[:, :, :, 0]
        X2_counts = X2[:, :, :, 1]

        for k in range(n_histograms):
            bin_len = bin_lengths[k]
            X1_bins_k = X1_bins[:, k]
            X1_counts_k = X1_counts[:, k]
            X1_counts_k = X1_counts_k / X1_counts_k.sum(dim=-1, keepdim=True)
            X1_bins_k = X1_bins_k[:, :bin_len]
            X1_counts_k = X1_counts_k[:, :bin_len]

            X2_bins_k = X2_bins[:, k]
            X2_counts_k = X2_counts[:, k]
            X2_counts_k = X2_counts_k / X2_counts_k.sum(dim=-1, keepdim=True)
            X2_bins_k = X2_bins_k[:, :bin_len]
            X2_counts_k = X2_counts_k[:, :bin_len]

            X1_bins_k_repeated = torch.cat([X1_bins_k] * n2_samples, 0)
            X2_bins_k_repeated = X2_bins_k.repeat_interleave(n1_samples, 0)
            X1_counts_k_repeated = torch.cat([X1_counts_k] * n2_samples, 0)
            X2_counts_k_repeated = X2_counts_k.repeat_interleave(n1_samples, 0)

            if self.n_batches is None:
                emd = ot.wasserstein_1d(
                    X1_bins_k_repeated.T, X2_bins_k_repeated.T,
                    u_weights=X1_counts_k_repeated.T, v_weights=X2_counts_k_repeated.T,
                )
            else:
                n_inputs = X1_bins_k_repeated.shape[0]
                bsz = max(math.ceil(n_inputs / self.n_batches), 1)
                emds = []
                for batch_n in range(self.n_batches):
                    start_ix = batch_n * bsz
                    stop_ix = min((batch_n + 1) * bsz, n_inputs)
                    if start_ix >= n_inputs:
                        break
                    emds.append(ot.wasserstein_1d(
                        X1_bins_k_repeated[start_ix:stop_ix].T,
                        X2_bins_k_repeated[start_ix:stop_ix].T,
                        u_weights=X1_counts_k_repeated[start_ix:stop_ix].T,
                        v_weights=X2_counts_k_repeated[start_ix:stop_ix].T,
                    ))
                emd = torch.cat(emds)

            emd = emd.reshape(n2_samples, n1_samples).T
            if X1_X2:
                for i in range(emd.shape[0]):
                    emd[i, i] = 0.0
            emd_K_ix = torch.exp(-(emd / self.lengthscales[k])) * self.weights[k]
            emd_K = emd_K_ix if k == 0 else emd_K + emd_K_ix

        return emd_K

    def forward(self, x1, x2, diag=False, **params):
        if len(x1.shape) != 2 or len(x2.shape) != 2:
            x1 = x1.squeeze()
            x2 = x2.squeeze()
            assert len(x1.shape) == 2
            assert len(x2.shape) == 2

        if self.n_sg > 0:
            kernel = self.sg_kernel(x1[:, 0:self.n_sg], x2[:, 0:self.n_sg])
            if self.n_histogram > 0:
                kernel = kernel * self.compute_emd_more_memory_efficient(x1[:, self.n_sg:], x2[:, self.n_sg:])
        else:
            kernel = self.compute_emd_more_memory_efficient(x1[:, self.n_sg:], x2[:, self.n_sg:])

        if diag:
            return kernel.diagonal(dim1=-2, dim2=-1)
        return kernel


# ═══════════════════════════════════════════════════════════════════════════════
# SAAS Prior Utilities (from BoTorch MAP-SAAS)
# ═══════════════════════════════════════════════════════════════════════════════

EPS = 1e-8


class SaasPriorHelper:
    """Helper for specifying SAAS parameter and setting closures."""

    def __init__(self, tau: float | None = None):
        self._tau = torch.as_tensor(tau) if tau is not None else None

    def tau(self, m: Kernel) -> Tensor:
        return (
            self._tau.to(m.lengthscales)
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscales_prior_param_or_closure(self, m: Kernel) -> Tensor:
        tau = self.tau(m)
        return tau.view(*tau.shape, 1, 1) / (m.lengthscales ** 2)

    def inv_lengthscales_prior_setting_closure(self, m: Kernel, value: Tensor) -> None:
        tau = self.tau(m)
        tau = tau.view(*tau.shape, 1, 1)
        lb = m.raw_lengthscales_constraint.lower_bound.to(tau)
        ub = m.raw_lengthscales_constraint.upper_bound.to(tau)
        m._set_lengthscales((tau / value.to(tau)).sqrt().clamp(lb + EPS, ub - EPS))

    def tau_prior_param_or_closure(self, m: Kernel) -> Tensor:
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m: Kernel, value: Tensor) -> None:
        lb = m.raw_tau_constraint.lower_bound.to(m.raw_tau)
        ub = m.raw_tau_constraint.upper_bound.to(m.raw_tau)
        m.raw_tau.data.fill_(
            m.raw_tau_constraint.inverse_transform(
                value.to(m.raw_tau).clamp(lb + EPS, ub - EPS)
            ).item()
        )


def add_saas_prior(base_kernel: Kernel, tau: float | None = None) -> Kernel:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is: tau / lengthscales^2 ~ HC(1.0).
    If tau is None, an additional HC(0.1) prior is placed on tau.
    """
    tkwargs = {"device": base_kernel.device, "dtype": base_kernel.dtype}
    batch_shape = base_kernel.raw_lengthscales.shape[:-2]
    IntervalClass = Interval

    base_kernel.register_constraint(
        param_name="raw_lengthscales",
        constraint=IntervalClass(0.01, 1e4, initial_value=1),
        replace=True,
    )
    prior_helper = SaasPriorHelper(tau=tau)

    if tau is None:
        base_kernel.register_parameter(
            name="raw_tau",
            parameter=Parameter(torch.full(batch_shape, 0.1, **tkwargs)),
        )
        base_kernel.register_constraint(
            param_name="raw_tau",
            constraint=IntervalClass(1e-3, 10, initial_value=0.1),
            replace=True,
        )
        base_kernel.register_prior(
            name="tau_prior",
            prior=HalfCauchyPrior(torch.tensor(0.1, **tkwargs)),
            param_or_closure=prior_helper.tau_prior_param_or_closure,
            setting_closure=prior_helper.tau_prior_setting_closure,
        )

    base_kernel.register_prior(
        name="inv_lengthscales_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscales_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscales_prior_setting_closure,
    )
    return base_kernel


# ═══════════════════════════════════════════════════════════════════════════════
# GP Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class SgEmdExactGPModelV2(gpytorch.models.ExactGP):
    """Exact GP with dual EMD-histogram + RBF-symmetry kernel for regression."""

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        n_sg,
        map_saas_tau,
        n_histogram=31,
        data_shape=(31, 20, 2),
        n_batches_emd_kernel=None,
        add_saas_ls_prior=False,
    ):
        super(SgEmdExactGPModelV2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SgEMDKernelV2(
            n_sg,
            n_histogram=n_histogram,
            data_shape=data_shape,
            n_batches=n_batches_emd_kernel,
        )
        if add_saas_ls_prior:
            self.covar_module = add_saas_prior(
                base_kernel=self.covar_module,
                tau=map_saas_tau,
            )

    def forward(self, x):
        mean_f = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_f, covar)


class EmdSgGpClassificationModel(ApproximateGP):
    """Variational GP with EMD-histogram + RBF-symmetry kernel for binary classification."""

    def __init__(
        self,
        train_x,
        n_histogram=21,
        n_sg=11,
        data_shape=(21, 20, 2),
        n_batches_emd_kernel=None,
    ):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(EmdSgGpClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SgEMDKernelV2(
            n_sg=n_sg,
            n_histogram=n_histogram,
            data_shape=data_shape,
            n_batches=n_batches_emd_kernel,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
