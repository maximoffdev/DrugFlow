from abc import ABC
from abc import abstractmethod
import math
import torch
from torch_scatter import scatter_mean, scatter_add
from dataclasses import dataclass
from typing import Literal, Optional
from IPython.display import display, Markdown

import src.data.so3_utils as so3


class ICFM(ABC):
    r"""
    Abstract base class for all Independent-coupling CFM classes.
    Defines a common interface.
    Notation:
    - zt is the intermediate representation at time step t \in [0, 1]
    - zs is the noised representation at time step s < t

    # TODO: add interpolation schedule (not necessrily linear)
    """
    def __init__(self, sigma):
        self.sigma = sigma

    @abstractmethod
    def sample_zt(self, z0, z1, t, *args, **kwargs) -> torch.Tensor:
        """ TODO. """
        pass

    @abstractmethod
    def sample_zt_given_zs(self, *args, **kwargs) -> torch.Tensor:
        """ Perform update, typically using an explicit Euler step. """
        pass

    @abstractmethod
    def sample_z0(self, *args, **kwargs) -> torch.Tensor:
        """ Prior. """
        pass

    @abstractmethod
    def compute_loss(self, pred, z0, z1, *args, **kwargs) -> torch.Tensor:
        """ Compute loss per sample. """
        pass


class CoordICFM(ICFM):
    def __init__(self, sigma):
        self.dim = 3
        self.scale = 2.7
        super().__init__(sigma)

    def sample_zt(self, z0, z1, t, batch_mask, temperature=None):
        zt = t[batch_mask] * z1 + (1 - t)[batch_mask] * z0
        # zt = self.sigma * z0 + t[batch_mask] * z1 + (1 - t)[batch_mask] * z0  # TODO: do we have to compute Psi?
        return zt

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask, temperature=None):
        """ Perform an explicit Euler step. """
        step_size = t - s

        # Temperature-dependent diffusion scaling (COMMENTED OUT for now).
        # Treat temperature like time: batch-level (B,1), broadcast via batch_mask.
        #
        # if temperature is not None:
        #     # Example scaling law (placeholder): scale ∝ sqrt(T)
        #     # Choose a reference temperature so temperature=1.0 leaves behavior unchanged.
        #     T_ref = 1.0
        #     temp_node = temperature[batch_mask] if temperature.numel() > 1 else temperature
        #     temp_scale = torch.sqrt(torch.clamp(temp_node / T_ref, min=1e-6))
        #     zt = zs + step_size[batch_mask] * (self.scale * temp_scale) * pred
        # else:
        #     zt = zs + step_size[batch_mask] * self.scale * pred

        zt = zs + step_size[batch_mask] * self.scale * pred
        return zt

    def sample_z0(self, com, batch_mask, temperature=None):
        """ Prior. """
        z0 = torch.randn((len(batch_mask), self.dim), device=batch_mask.device)

        # Move center of mass
        z0 = z0 + com[batch_mask]

        return z0

    def reduce_loss(self, loss, batch_mask, reduce):
        assert reduce in {'mean', 'sum', 'none'}

        if reduce == 'mean':
            loss = scatter_mean(loss / self.dim, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss

    def compute_loss(self, pred, z0, z1, t, batch_mask, reduce='mean', temperature=None):
        """ Compute loss per sample. """

        # Temperature-dependent diffusion scaling (COMMENTED OUT for now).
        # If enabled, you would typically need to adjust the target and/or normalization
        # consistently with sample_zt_given_zs.
        #
        # if temperature is not None:
        #     T_ref = 1.0
        #     temp_node = temperature[batch_mask] if temperature.numel() > 1 else temperature
        #     temp_scale = torch.sqrt(torch.clamp(temp_node / T_ref, min=1e-6))
        #     effective_scale = self.scale * temp_scale
        # else:
        #     effective_scale = self.scale
        # loss = torch.sum((pred - (z1 - z0) / effective_scale) ** 2, dim=-1)

        loss = torch.sum((pred - (z1 - z0) / self.scale) ** 2, dim=-1)

        return self.reduce_loss(loss, batch_mask, reduce)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask, temperature=None):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """
        # z1 = z0 + pred
        z1 = zt + (1 - t)[batch_mask] * pred
        return z1


class TorusICFM(ICFM):
    """
    Following:
    Chen, Ricky TQ, and Yaron Lipman.
    "Riemannian flow matching on general geometries."
    arXiv preprint arXiv:2302.03660 (2023).
    """
    def __init__(self, sigma, dim, scheduler_args=None):
        super().__init__(sigma)
        self.dim = dim

        # Scheduler that determines the rate at which the geodesic distance decreases
        scheduler_args = scheduler_args or {}
        scheduler_args["type"] = scheduler_args.get("type", "linear")  # default
        scheduler_args["learn_scaled"] = scheduler_args.get("learn_scaled", False)  # default

        # linear scheduler: kappa(t) = 1-t (default)
        if scheduler_args["type"] == "linear":
            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: t

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: torch.ones_like(t)

        # exponential scheduler: kappa(t) = exp(-c*t)
        elif scheduler_args["type"] == "exponential":

            self.c = scheduler_args["c"]
            assert self.c > 0

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: 1 - torch.exp(-self.c * t)

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: self.c * torch.exp(-self.c * t)

        # polynomial scheduler: kappa(t) = (1-t)^k
        elif scheduler_args["type"] == "polynomial":
            self.k = scheduler_args["k"]
            assert self.k > 0

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: 1 - (1 - t)**self.k

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: self.k * (1 - t)**(self.k - 1)

        else:
            raise NotImplementedError(f"Scheduler {scheduler_args['type']} not implemented.")

        kappa_interval = self.flow_scaling(torch.tensor([0.0, 1.0]))
        if kappa_interval[0] != 0.0 or kappa_interval[1] != 1.0:
            print(f"Scheduler should satisfy kappa(0)=1 and kappa(1)=0. Found "
                  f"interval {kappa_interval.tolist()} instead.")

        # determines whether the scaled vector field is learned or the scheduler
        # is post-multiplied
        self.learn_scaled = scheduler_args["learn_scaled"]

    @staticmethod
    def wrap(angle):
        r""" Maps angles to range [-\pi, \pi). """
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    def exponential_map(self, x, u):
        """
        :param x: point on the manifold
        :param u: point on the tangent space
        """
        return self.wrap(x + u)

    @staticmethod
    def logarithm_map(x, y):
        """
        :param x, y: points on the manifold
        """
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def sample_zt(self, z0, z1, t, batch_mask):
        """ expressed in terms of exponential and logarithm maps """

        # apply logarithm map
        # zt_tangent = t[batch_mask] * self.logarithm_map(z0, z1)
        zt_tangent = self.flow_scaling(t)[batch_mask] * self.logarithm_map(z0, z1)

        # apply exponential map
        return self.exponential_map(z0, zt_tangent)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # estimate z1_tangent based on zt and pred only
        if self.learn_scaled:
            pred = pred / torch.clamp(self.velocity_scaling(t), min=1e-3)[batch_mask]

        z1_tangent = (1 - t)[batch_mask] * pred

        # exponential map
        return self.exponential_map(zt, z1_tangent)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        step_size = t - s
        zt_tangent = step_size[batch_mask] * pred

        if not self.learn_scaled:
            zt_tangent = self.velocity_scaling(t)[batch_mask] * zt_tangent

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def sample_z0(self, batch_mask):
        """ Prior. """

        # Uniform distribution
        z0 = torch.rand((len(batch_mask), self.dim), device=batch_mask.device)

        return 2 * math.pi * z0 - math.pi

    def compute_loss(self, pred, z0, z1, zt, t, batch_mask, reduce='mean'):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}
        mask = ~torch.isnan(z1)
        z1 = torch.nan_to_num(z1, nan=0.0)

        zt_dot = self.logarithm_map(z0, z1)
        if self.learn_scaled:
            # NOTE: potentially requires output magnitude to vary substantially
            zt_dot = self.velocity_scaling(t)[batch_mask] * zt_dot
        loss = mask * (pred - zt_dot) ** 2
        loss = torch.sum(loss, dim=-1)

        if reduce == 'mean':
            denom = mask.sum(dim=-1) + 1e-6
            loss = scatter_mean(loss / denom, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)
        return loss


class SO3ICFM(ICFM):
    """
    All rotations are assumed to be in axis-angle format.
    Mostly following descriptions from the FoldFlow paper:
    https://openreview.net/forum?id=kJFIH23hXb

    See also:
    https://geomstats.github.io/_modules/geomstats/geometry/special_orthogonal.html#SpecialOrthogonal
    https://geomstats.github.io/_modules/geomstats/geometry/lie_group.html#LieGroup
    """
    def __init__(self, sigma):
        super().__init__(sigma)

    def exponential_map(self, base, tangent):
        """
        Args:
            base: base point (rotation vector) on the manifold
            tangent: point in tangent space at identity
        Returns:
            rotation vector on the manifold
        """
        # return so3.exp_not_from_identity(tangent, base_point=base)
        return so3.compose_rotations(base, so3.exp(tangent))

    def logarithm_map(self, base, r):
        """
        Args:
            base: base point (rotation vector) on the manifold
            r: rotation vector on the manifold
        Return:
            point in tangent space at identity
        """
        # return so3.log_not_from_identity(r, base_point=base)
        return so3.log(so3.compose_rotations(-base, r))

    def sample_zt(self, z0, z1, t, batch_mask):
        """
        Expressed in terms of exponential and logarithm maps.
        Corresponds to SLERP interpolation: R(t) = R1 exp( t * log(R1^T R2) )
        (see https://lucaballan.altervista.org/pdfs/IK.pdf, slide 16)
        """

        # apply logarithm map
        zt_tangent = t[batch_mask] * self.logarithm_map(z0, z1)

        # apply exponential map
        return self.exponential_map(z0, zt_tangent)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # estimate z1_tangent based on zt and pred only
        z1_tangent = (1 - t)[batch_mask] * pred

        # exponential map
        return self.exponential_map(zt, z1_tangent)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        # # parallel transport vector field to lie algebra so3 (at identity)
        # # (FoldFlow paper, Algorithm 3, line 8)
        # # TODO: is this correct? is it necessary?
        # pred = so3.compose(so3.inverse(zs), pred)

        step_size = t - s
        zt_tangent = step_size[batch_mask] * pred

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def sample_z0(self, batch_mask):
        """ Prior. """
        return so3.random_uniform(n_samples=len(batch_mask), device=batch_mask.device)

    @staticmethod
    def d_R_squared_SO3(rot_vec_1, rot_vec_2):
        r"""
        Squared Riemannian metric on SO(3).
        Defined as d(R1, R2) = sqrt(0.5) ||log(R1^T R2)||_F
        where R1, R2 are rotation matrices.

        The following is equivalent if the difference between the rotations is
        expressed as a rotation vector \omega_diff:
        d(r1, r2) = ||\omega_diff||_2
        -----
        With the definition of the Frobenius matrix norm ||A||_F^2 = trace(A^H A):
        d^2(R1, R2) = 1/2 ||log(R1^T R2)||_F^2
                    = 1/2 || hat(R_d) ||_F^2
                    = 1/2 tr( hat(R_d)^T hat(R_d) )
                    = 1/2 * 2 * ||\omega||_2^2
        """

        # rot_mat_1 = so3.matrix_from_rotation_vector(rot_vec_1)
        # rot_mat_2 = so3.matrix_from_rotation_vector(rot_vec_2)
        # rot_mat_diff = rot_mat_1.transpose(-2, -1) @ rot_mat_2
        # return torch.norm(so3.log(rot_mat_diff, as_skew=True), p='fro', dim=(-2, -1))

        diff_rot = so3.compose_rotations(-rot_vec_1, rot_vec_2)
        return diff_rot.square().sum(dim=-1)

    def compute_loss(self, pred, z0, z1, zt, t, batch_mask, reduce='mean', eps=5e-2):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}

        zt_dot = self.logarithm_map(zt, z1) / torch.clamp(1 - t, min=eps)[batch_mask]

        # TODO: do I need this?
        # pred_at_id = self.logarithm_map(zt, pred) / torch.clamp(1 - t, min=eps)[batch_mask]

        loss = torch.sum((pred - zt_dot)**2, dim=-1)  # TODO: is this the right loss in SO3?
        # loss = self.d_R_squared_SO3(zt_dot, pred)

        if reduce == 'mean':
            loss = scatter_mean(loss, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss


#################
# Predicting z1 #
#################

class CoordICFMPredictFinal(CoordICFM):
    def __init__(self, sigma):
        self.dim = 3
        super().__init__(sigma)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask, temperature=None):
        """ Perform an explicit Euler step. """

        # step_size = t - s
        # zt = zs + step_size[batch_mask] * z1_minus_zs_pred / (1.0 - s)[batch_mask]

        # for numerical stability
        step_size = (t - s) / (1.0 - s)
        assert torch.all(step_size <= 1.0)
        # step_size = torch.clamp(step_size, max=1.0)

        # Temperature-dependent diffusion scaling (COMMENTED OUT for now).
        # if temperature is not None:
        #     T_ref = 1.0
        #     temp_node = temperature[batch_mask] if temperature.numel() > 1 else temperature
        #     temp_scale = torch.sqrt(torch.clamp(temp_node / T_ref, min=1e-6))
        #     zt = zs + step_size[batch_mask] * temp_scale * z1_minus_zs_pred
        #     return zt

        zt = zs + step_size[batch_mask] * pred
        return zt

    def compute_loss(self, pred, z0, z1, t, batch_mask, reduce='mean', temperature=None):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}
        t = torch.clamp(t, max=0.9)
        zt = self.sample_zt(z0, z1, t, batch_mask, temperature=temperature)

        # Temperature-dependent diffusion scaling (COMMENTED OUT for now).
        # If enabled, update the loss consistently with sample_zt_given_zs.
        #
        # if temperature is not None:
        #     T_ref = 1.0
        #     temp_node = temperature[batch_mask] if temperature.numel() > 1 else temperature
        #     temp_scale = torch.sqrt(torch.clamp(temp_node / T_ref, min=1e-6))
        #     loss = torch.sum((z1_minus_zt_pred * temp_scale + zt - z1) ** 2, dim=-1) / torch.square(1 - t)[batch_mask].squeeze()
        # else:
        #     loss = torch.sum((z1_minus_zt_pred + zt - z1) ** 2, dim=-1) / torch.square(1 - t)[batch_mask].squeeze()

        loss = torch.sum((pred + zt - z1) ** 2, dim=-1) / torch.square(1 - t)[batch_mask].squeeze()

        if reduce == 'mean':
            loss = scatter_mean(loss / self.dim, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask, temperature=None):
        return pred + zt


class TorusICFMPredictFinal(TorusICFM):
    """
    Following:
    Chen, Ricky TQ, and Yaron Lipman.
    "Riemannian flow matching on general geometries."
    arXiv preprint arXiv:2302.03660 (2023).
    """
    def __init__(self, sigma, dim):
        super().__init__(sigma, dim)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # exponential map
        return self.exponential_map(zt, pred)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        # step_size = t - s
        # zt_tangent = step_size[batch_mask] * z1_tangent_pred / (1.0 - s)[batch_mask]

        # for numerical stability
        step_size = (t - s) / (1.0 - s)
        assert torch.all(step_size <= 1.0)
        # step_size = torch.clamp(step_size, max=1.0)
        zt_tangent = step_size[batch_mask] * pred

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def compute_loss(self, pred, z0, z1, zt=None, t=None, batch_mask=None, reduce='mean'):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}
        if t is None or batch_mask is None:
            raise ValueError("t and batch_mask are required")
        zt = self.sample_zt(z0, z1, t, batch_mask)
        t = torch.clamp(t, max=0.9)

        mask = ~torch.isnan(z1)
        z1 = torch.nan_to_num(z1, nan=0.0)
        loss = mask * (pred - self.logarithm_map(zt, z1)) ** 2
        loss = torch.sum(loss, dim=-1) / torch.square(1 - t)[batch_mask].squeeze()

        if reduce == 'mean':
            denom = mask.sum(dim=-1) + 1e-6
            loss = scatter_mean(loss / denom, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss


############################
# Score-based diffusion (x) #
############################

@dataclass(frozen=True)
class SDEParams:
    """SDE / perturbation-kernel parameters.

    time convention:
      t=0 -> max noise
      t=1 -> clean
    """
    kind: Literal["ve", "vp"] = "ve"

    # VE params (geometric sigma schedule)
    sigma_min: float = 0.01
    sigma_max: float = 50.0

    # VP params (linear beta schedule in forward time; we internally use tau=1-t)
    beta_min: float = 0.1
    beta_max: float = 20.0
    vp_sigma_scale: float = 1.0  # optional scaling for non-normalized coordinate domains


class CoordScoreDiffusion:
    """Score-based diffusion utilities for Cartesian coordinates.

    This lives alongside the existing flow-matching (ICFM) classes.

        Time convention (match DrugFlow / flow-matching configs):
            - t in [0, 1]
            - t=0 corresponds to maximum noise (prior)
            - t=1 corresponds to data (clean)

        This implementation uses a VE (variance exploding) style perturbation kernel:
            x_t = x_clean + sigma(t) * eps,  eps ~ N(0, I)
        with sigma(t) decreasing from sigma_max at t=0 to sigma_min at t=1.

    Training target score for DSM is:
      s*(x_t, t) = \nabla_{x_t} log p(x_t | x_0) = (x_0 - x_t) / sigma(t)^2

        Sampling provides a simple denoising step compatible with VE discretizations:
            x_{t} = x_{s} + (sigma(s)^2 - sigma(t)^2) * score(x_s, s) + sqrt(sigma(s)^2 - sigma(t)^2) * z
        where 0 <= s < t <= 1 (i.e. we move from noisier to cleaner).

    Temperature:
      The API accepts an optional batch-level `temperature` tensor (B,1) and
      provides commented-out hooks for making sigma depend on temperature.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        dim: int = 3,
        *,
        sde: SDEParams | None = None,
        enforce_zero_com_noise: bool = False,
        project_predicted_score_zero_com: bool = False,
    ):
        self.dim = int(dim)
        self.enforce_zero_com_noise = bool(enforce_zero_com_noise)
        self.project_predicted_score_zero_com = bool(project_predicted_score_zero_com)

        # Backward-compatible defaults: VE schedule controlled by sigma_min/max
        if sde is None:
            sde = SDEParams(kind="ve", sigma_min=float(sigma_min), sigma_max=float(sigma_max))
        self.sde = sde

    @staticmethod
    def _project_zero_com(v: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """Project per-example node vectors to the zero center-of-mass (mean) subspace.

        This removes the global translation mode per graph, i.e. enforces
        sum_i v_i = 0 for each example.
        """
        if v.numel() == 0:
            return v
        if batch_mask.dtype != torch.long:
            batch_mask = batch_mask.to(dtype=torch.long)
        mean_v = scatter_mean(v, batch_mask, dim=0)
        return v - mean_v[batch_mask]

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        """Convert 'clean-time' t (0 noise -> 1 clean) to forward-time tau (0 clean -> 1 noisy)."""
        t = torch.clamp(t, 0.0, 1.0)
        return 1.0 - t

    def alpha(self, t: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
        """Mean coefficient alpha(t) for x_t = alpha(t) x0 + sigma(t) eps."""
        t = torch.clamp(t, 0.0, 1.0)

        if self.sde.kind == "ve":
            return torch.ones_like(t)

        if self.sde.kind == "vp":
            tau = self._tau(t)

            # Temperature control for VP that preserves alpha(t)^2 + sigma(t)^2 = 1:
            # scale the beta schedule by temperature (higher T -> faster diffusion).
            # T_ref=1.0 keeps default behavior unchanged.
            if temperature is not None and not isinstance(temperature, torch.Tensor):
                temperature = torch.as_tensor(temperature, device=t.device, dtype=t.dtype)
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.to(device=t.device, dtype=t.dtype)
                if temperature.numel() == 1:
                    temperature = temperature.expand_as(t)
                T_ref = 1.0
                beta_scale = torch.clamp(temperature / T_ref, min=1e-6)
            else:
                beta_scale = torch.ones_like(t)

            beta0 = torch.as_tensor(self.sde.beta_min, device=t.device, dtype=t.dtype) * beta_scale
            beta1 = torch.as_tensor(self.sde.beta_max, device=t.device, dtype=t.dtype) * beta_scale
            int_beta = beta0 * tau + 0.5 * (beta1 - beta0) * tau * tau
            return torch.exp(-0.5 * int_beta)

        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")

    def sigma(self, t: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
        """Std coefficient sigma(t) for x_t = alpha(t) x0 + sigma(t) eps."""
        t = torch.clamp(t, 0.0, 1.0)

        # Optional temperature control:
        # scale sigma by sqrt(T / T_ref). With T_ref=1.0 this is a no-op by default.
        if temperature is not None and not isinstance(temperature, torch.Tensor):
            temperature = torch.as_tensor(temperature, device=t.device, dtype=t.dtype)
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.to(device=t.device, dtype=t.dtype)
            # Allow either scalar temperature or batch-level (B,1).
            if temperature.numel() == 1:
                temperature = temperature.expand_as(t)
            T_ref = 1.0
            temp_scale = torch.sqrt(torch.clamp(temperature / T_ref, min=1e-6))
        else:
            temp_scale = None

        if self.sde.kind == "ve":
            # t=0 -> sigma_max, t=1 -> sigma_min
            sigma = float(self.sde.sigma_max) * (float(self.sde.sigma_min) / float(self.sde.sigma_max)) ** t

            if temp_scale is not None:
                sigma = sigma * temp_scale

            return sigma

        if self.sde.kind == "vp":
            a = self.alpha(t, temperature=temperature)
            sig = torch.sqrt(torch.clamp(1.0 - a * a, min=1e-12))
            sig = sig * float(self.sde.vp_sigma_scale)
            return sig

        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")

    def sample_zt(self, x_clean: torch.Tensor, t: torch.Tensor, batch_mask: torch.Tensor, temperature=None):
        """Sample x_t given x_clean.

        Returns:
          xt: (N,3)
          eps: (N,3) the noise used
        """
        eps = torch.randn_like(x_clean)
        if self.enforce_zero_com_noise:
            eps = self._project_zero_com(eps, batch_mask)
        a_t = self.alpha(t, temperature=temperature)[batch_mask]
        sigma_t = self.sigma(t, temperature=temperature)[batch_mask]
        xt = a_t * x_clean + sigma_t * eps
        return xt, eps

    def sample_z1(self, com: torch.Tensor, batch_mask: torch.Tensor, temperature=None) -> torch.Tensor:
        """Prior for sampling: x_{t=0} ~ N(com, sigma_max^2 I).

        Name kept as sample_z1 for historical symmetry with ICFM codepaths; for
        score diffusion with DrugFlow convention, this returns the t=0 state.
        """
        z = torch.randn((len(batch_mask), self.dim), device=batch_mask.device, dtype=com.dtype)
        if self.enforce_zero_com_noise:
            z = self._project_zero_com(z, batch_mask)
        t0 = torch.zeros((com.size(0), 1), device=com.device, dtype=com.dtype)
        sigma_0 = self.sigma(t0, temperature=temperature)
        z = z * sigma_0[batch_mask] + com[batch_mask]
        return z

    def score_target(self, xt: torch.Tensor, x_clean: torch.Tensor, t: torch.Tensor, batch_mask: torch.Tensor, temperature=None):
        """Compute the analytical target score for DSM."""
        a_t = self.alpha(t, temperature=temperature)[batch_mask]
        sigma_t = self.sigma(t, temperature=temperature)[batch_mask]
        return (a_t * x_clean - xt) / torch.clamp(sigma_t * sigma_t, min=1e-12)

    def reduce_loss(self, loss: torch.Tensor, batch_mask: torch.Tensor, reduce: str):
        assert reduce in {"mean", "sum", "none"}
        if reduce == "mean":
            return scatter_mean(loss / self.dim, batch_mask, dim=0)
        if reduce == "sum":
            return scatter_add(loss, batch_mask, dim=0)
        return loss

    def score_loss(
        self,
        force_pred: torch.Tensor,
        x_clean: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        reduce: str = "mean",
        temperature=None,
        weight_by_sigma: bool = False,
    ) -> torch.Tensor:
        """Denoising score matching loss.

        If `weight_by_sigma` is True, uses a common VE weighting: multiply by sigma(t)^2.
        """

        score_pred = force_pred
        if self.project_predicted_score_zero_com:
            score_pred = self._project_zero_com(score_pred, batch_mask)
        
        target = self.score_target(xt, x_clean, t, batch_mask, temperature=temperature)
        per_node = torch.sum((score_pred - target) ** 2, dim=-1)

        if weight_by_sigma:
            # print("per npde before sigma weighting: ", per_node.mean().item(), per_node.std().item())
            sigma_t = self.sigma(t, temperature=temperature)[batch_mask].squeeze(-1)
            per_node = per_node * (sigma_t * sigma_t)
        #     print("sigma_t:", sigma_t.mean().item(), sigma_t.std().item())
        # print("Score Loss Stats: ", per_node.mean().item(), per_node.std().item())

        # print("True Score Stats: ", target.mean().item(), target.std().item())
        # print("Pred Score Stats: ", score_pred.mean().item(), score_pred.std().item())
        # print()

        return self.reduce_loss(per_node, batch_mask, reduce)

    def _ve_sigma_forward(self, tau: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
        """Forward-time VE sigma schedule.

        This is expressed in diffusion (forward) time tau where tau=0 is clean and tau=1 is max noise.
        """
        tau = torch.clamp(tau, 0.0, 1.0)
        sigma = float(self.sde.sigma_min) * (float(self.sde.sigma_max) / float(self.sde.sigma_min)) ** tau
        if isinstance(temperature, torch.Tensor):
            # Mirror sigma() temperature scaling: sigma <- sigma * sqrt(T)
            if temperature.numel() == 1:
                temperature = temperature.expand_as(tau)
            T_ref = 1.0
            sigma = sigma * torch.sqrt(torch.clamp(temperature / T_ref, min=1e-6))
        return sigma

    def _sde_g2_forward(self, tau: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
        """Return g(tau)^2 for the forward SDE coefficients.

        Shapes:
          tau: (B,1)
          returns: (B,)
        """
        tau = torch.clamp(tau, 0.0, 1.0)

        if self.sde.kind == "ve":
            sigma_tau = self._ve_sigma_forward(tau, temperature=temperature).view(-1)
            log_ratio = math.log(float(self.sde.sigma_max) / float(self.sde.sigma_min))
            return (2.0 * log_ratio) * (sigma_tau * sigma_tau)

        if self.sde.kind == "vp":
            beta0 = float(self.sde.beta_min)
            beta1 = float(self.sde.beta_max)
            beta = (beta0 + (beta1 - beta0) * tau)
            if isinstance(temperature, torch.Tensor):
                beta = beta * torch.clamp(temperature, min=1e-6)
            vp_scale = float(self.sde.vp_sigma_scale)
            return beta.view(-1) * (vp_scale * vp_scale)

        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")

    def _sde_f_forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        batch_mask: torch.Tensor,
        temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return drift f(x,tau) for the forward SDE coefficients.

        Shapes:
          x: (N,dim)
          tau: (B,1)
          returns: (N,dim)
        """
        if self.sde.kind == "ve":
            return torch.zeros_like(x)

        if self.sde.kind == "vp":
            tau = torch.clamp(tau, 0.0, 1.0)
            beta0 = float(self.sde.beta_min)
            beta1 = float(self.sde.beta_max)
            beta = (beta0 + (beta1 - beta0) * tau)
            if isinstance(temperature, torch.Tensor):
                beta = beta * torch.clamp(temperature, min=1e-6)
            beta_node = beta[batch_mask]
            return -0.5 * beta_node * x

        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")

    def _sde_div_f_forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        batch_mask: torch.Tensor,
        temperature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return div(f)(x,tau) aggregated per example (B,)."""
        B = int(batch_mask.max().item()) + 1 if batch_mask.numel() > 0 else 0
        if B == 0:
            return torch.empty((0,), device=x.device, dtype=x.dtype)

        if self.sde.kind == "ve":
            return torch.zeros((B,), device=x.device, dtype=x.dtype)

        if self.sde.kind == "vp":
            tau = torch.clamp(tau, 0.0, 1.0)
            beta0 = float(self.sde.beta_min)
            beta1 = float(self.sde.beta_max)
            beta = (beta0 + (beta1 - beta0) * tau)
            if isinstance(temperature, torch.Tensor):
                beta = beta * torch.clamp(temperature, min=1e-6)
            beta_node = beta[batch_mask]
            div_f_node = -0.5 * beta_node.view(-1) * float(x.size(-1))
            return scatter_add(div_f_node, batch_mask, dim=0)

        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")
    
    def hjb_loss(
        self,
        force_pred: torch.Tensor,
        energy_pred: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        *,
        temperature: torch.Tensor | float | None = None,
        reduce: str = "mean",
        divergence: Literal["exact", "hutchinson"] = "hutchinson",
        n_trace_samples: int = 1,
        enable_higher_order: bool = True,
        compute_hjb: bool = True,
        compute_consistency: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute HJB residual loss + force/energy consistency.

        This is meant to be called with `energy_pred` and `force_pred` computed from a
        dynamics model where `x` and `t` were created with `requires_grad_(True)`.

        Returns:
          loss_hjb: shape (B,) mean-squared HJB residual per example
          loss_consistency: shape (B,) mean-squared (F + ∇_x u) per example
        """
        assert reduce in {"mean", "sum", "none"}

        if int(n_trace_samples) <= 0:
            raise ValueError(f"n_trace_samples must be >= 1; got {n_trace_samples}")

        create_graph = bool(enable_higher_order)

        B = int(batch_mask.max().item()) + 1 if batch_mask.numel() > 0 else 0
        if B == 0:
            empty = torch.empty((0,), device=x.device, dtype=x.dtype)
            return empty, empty

        F_pred = force_pred
        if self.project_predicted_score_zero_com:
            F_pred = self._project_zero_com(F_pred, batch_mask)
        u_pred = energy_pred.view(-1)
        if u_pred.numel() != B:
            raise ValueError(f"energy_pred must have shape (B,), got {tuple(energy_pred.shape)}")

        # Temperature handling: allow scalar or (B,1)
        if temperature is None:
            temperature_t = torch.ones((B, 1), device=t.device, dtype=t.dtype)
        elif isinstance(temperature, (int, float)):
            temperature_t = torch.full((B, 1), float(temperature), device=t.device, dtype=t.dtype)
        else:
            temperature_t = temperature.to(device=t.device, dtype=t.dtype)
            if temperature_t.ndim == 1:
                temperature_t = temperature_t.unsqueeze(-1)
            if temperature_t.shape != (B, 1):
                raise ValueError(f"temperature must have shape (B,1) or scalar; got {tuple(temperature_t.shape)}")

        tau = self._tau(t)

        # We evaluate SDE coefficients using diffusion forward-time tau = 1 - t.
        # The network was evaluated at `t`, so to obtain du/dtau we use the chain rule:
        #   tau = 1 - t  =>  du/dtau = -du/dt.
        du_dx = None
        if compute_hjb:
            # If enable_higher_order=True we keep retain_graph=True to ensure the outer backward
            # can backpropagate through these autograd.grad calls.
            du_dt = torch.autograd.grad(
                outputs=u_pred.sum(),
                inputs=t,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]
            du_dt = -du_dt.view(-1)    # du/dtau

            # If we also need the consistency term, compute ∇_x u *before* the divergence
            # computation. In eval/logging mode (create_graph=False) the divergence autograd.grad
            # would otherwise free the shared forward graph and break this subsequent grad call.
            if compute_consistency:
                du_dx = torch.autograd.grad(
                    outputs=u_pred.sum(),
                    inputs=x,
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=False,
                )[0]

            # div(F) = trace(dF/dx)
            # Exact computation is expensive (and memory-heavy). Hutchinson estimator is the default.
            if divergence == "exact":
                div_F_node = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
                for i in range(x.size(-1)):
                    # If training with higher-order grads, retain for outer backward.
                    # Otherwise, only retain until the final component.
                    retain = True if create_graph else (i < x.size(-1) - 1)
                    grad_Fi = torch.autograd.grad(
                        outputs=F_pred[:, i].sum(),
                        inputs=x,
                        create_graph=create_graph,
                        retain_graph=retain,
                        allow_unused=False,
                    )[0]
                    div_F_node = div_F_node + grad_Fi[:, i]
                div_F = scatter_add(div_F_node, batch_mask, dim=0)

            elif divergence == "hutchinson":
                # Hutchinson trace estimator:
                #   tr(J) = E_v [ v^T J v ] for Rademacher v.
                # We compute v^T J v using a single VJP: grad_x (sum_i F_i(x) v_i) = J^T v.
                div_F_accum = torch.zeros((B,), device=x.device, dtype=x.dtype)
                n_samples = int(n_trace_samples)
                for s_idx in range(n_samples):
                    v = torch.empty_like(x).bernoulli_(0.5)
                    v = v.mul_(2.0).add_(-1.0)

                    scalar = torch.sum(F_pred * v)
                    # If training with higher-order grads, retain for outer backward.
                    # Otherwise, only retain if we will take another trace sample.
                    retain = True if create_graph else (s_idx < n_samples - 1)
                    Jt_v = torch.autograd.grad(
                        outputs=scalar,
                        inputs=x,
                        create_graph=create_graph,
                        retain_graph=retain,
                        allow_unused=False,
                    )[0]
                    div_node = torch.sum(Jt_v * v, dim=-1)
                    div_F_accum = div_F_accum + scatter_add(div_node, batch_mask, dim=0)
                div_F = div_F_accum / float(n_samples)

            else:
                raise ValueError(f"Unknown divergence mode: {divergence}")
        else:
            du_dt = torch.zeros((B,), device=x.device, dtype=x.dtype)
            div_F = torch.zeros((B,), device=x.device, dtype=x.dtype)

        temp_denom = torch.clamp(temperature_t.view(-1), min=1e-6)

        g2 = self._sde_g2_forward(tau, temperature=temperature_t)
        f_node = self._sde_f_forward(x, tau, batch_mask, temperature=temperature_t)
        F_dot_f = scatter_add(torch.sum(F_pred * f_node, dim=-1), batch_mask, dim=0)
        div_f = self._sde_div_f_forward(x, tau, batch_mask, temperature=temperature_t)

        F_norm2 = scatter_add(torch.sum(F_pred * F_pred, dim=-1), batch_mask, dim=0)

        if compute_hjb:
            hjb_residual = (
                du_dt
                - F_dot_f
                + 0.5 * g2 * F_norm2 / temp_denom
                - div_f
                + 0.5 * g2 * div_F / temp_denom
            )
            loss_hjb = hjb_residual * hjb_residual
        else:
            loss_hjb = torch.zeros((B,), device=x.device, dtype=x.dtype)

        if compute_consistency:
            # ∇_x u is only needed for the consistency term; skip it when lambda_consistency=0.
            if du_dx is None:
                du_dx = torch.autograd.grad(
                    outputs=u_pred.sum(),
                    inputs=x,
                    create_graph=create_graph,
                    retain_graph=True if create_graph else False,
                    allow_unused=False,
                )[0]
            per_node_cons = torch.mean((F_pred + du_dx) ** 2, dim=-1)
            loss_consistency = scatter_mean(per_node_cons, batch_mask, dim=0)
        else:
            loss_consistency = torch.zeros((B,), device=x.device, dtype=x.dtype)
            
        # Create a single string with labels
        status_msg = (
            f"du_dt: {du_dt.mean():.4f} | "
            f"F_dot_f: {F_dot_f.mean():.4f} | "
            f"Term3: {torch.mean(0.5 * g2 * F_norm2 / temp_denom):.4f} | "
            f"div_f: {div_f.mean():.4f} | "
            f"Term5: {torch.mean(0.5 * g2 * div_F / temp_denom):.4f}"
        )

        # Print with end='\r' to return to the start of the line instead of a new line
        print(status_msg, end='\r')

        if reduce == "none":
            return loss_hjb, loss_consistency
        
        # print("force_pred Stats: ", F_pred.mean().item(), F_pred.std().item())
        # print("force true Stats: ", -du_dx.mean().item(), (-du_dx).std().item())

        # Keep per-example shape (B,) for compatibility; reduce only affects node aggregation above.
        return loss_hjb, loss_consistency

    def reverse_step(
        self,
        xs: torch.Tensor,
        score_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        temperature=None,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """Single denoising step from time s to time t (expects 0 <= s < t <= 1).

        VE: uses the original discretization.
        VP: uses a reverse-SDE Euler-Maruyama step when stochastic=True; otherwise uses
            the deterministic probability-flow ODE step.
        """
        if self.project_predicted_score_zero_com:
            score_s = self._project_zero_com(score_s, batch_mask)
        if self.sde.kind == "vp":
            if not stochastic:
                return self.ode_step(xs, score_s, s=s, t=t, batch_mask=batch_mask, temperature=temperature)

            # VP reverse SDE in our time convention (t=0 noisy -> t=1 clean).
            # Internally, VP schedules are defined in forward time tau=1-t, and sampling
            # from noisy->clean corresponds to tau decreasing.
            # Euler-Maruyama update for a step tau_s -> tau_t (dtau = t - s):
            #   x <- x - (f - g^2 * score) * dtau + g * sqrt(dtau) * z
            # with forward drift f = -0.5 * beta(tau) * x and g^2 = beta(tau) * vp_sigma_scale^2.

            # Ensure temperature is a torch tensor if provided.
            if temperature is not None and not isinstance(temperature, torch.Tensor):
                temperature = torch.as_tensor(temperature, device=s.device, dtype=s.dtype)
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.to(device=s.device, dtype=s.dtype)

            tau_s = self._tau(s)
            beta0 = float(self.sde.beta_min)
            beta1 = float(self.sde.beta_max)
            beta = beta0 + (beta1 - beta0) * tau_s  # (B,1)

            # Match alpha()/sigma() temperature scaling: scale beta by temperature.
            if isinstance(temperature, torch.Tensor):
                if temperature.numel() == 1:
                    temperature = temperature.expand_as(beta)
                beta = beta * torch.clamp(temperature, min=1e-6)

            beta_node = beta[batch_mask]  # (N,1)
            dt = (t - s)[batch_mask]  # (N,1)
            dt = torch.clamp(dt, min=0.0)

            vp_scale = float(self.sde.vp_sigma_scale)
            g2_node = beta_node * (vp_scale * vp_scale)

            # -f*dt = 0.5 * beta * x * dt; and +g^2*score*dt
            drift = (0.5 * beta_node) * xs + g2_node * score_s

            noise = torch.randn_like(xs)
            if self.enforce_zero_com_noise:
                noise = self._project_zero_com(noise, batch_mask)
            g_node = torch.sqrt(torch.clamp(beta_node, min=0.0)) * vp_scale
            diffusion = g_node * torch.sqrt(torch.clamp(dt, min=0.0)) * noise

            return xs + drift * dt + diffusion

        sigma_s = self.sigma(s, temperature=temperature)[batch_mask]
        sigma_t = self.sigma(t, temperature=temperature)[batch_mask]
        delta = torch.clamp(sigma_s * sigma_s - sigma_t * sigma_t, min=0.0)

        drift = delta * score_s
        if stochastic:
            noise = torch.randn_like(xs)
            diffusion = torch.sqrt(torch.clamp(delta, min=0.0)) * noise
        else:
            diffusion = torch.zeros_like(xs)

        return xs + drift + diffusion

    def ode_step(
        self,
        xs: torch.Tensor,
        score_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        temperature=None,
    ) -> torch.Tensor:
        """Deterministic (ODE) denoising step from s to t.

        For VP: DDIM-like update using alpha/sigma.
        For VE: keep the original deterministic discretization.
        """
        if self.project_predicted_score_zero_com:
            score_s = self._project_zero_com(score_s, batch_mask)
        if self.sde.kind == "ve":
            return self.reverse_step(
                xs,
                score_s,
                s=s,
                t=t,
                batch_mask=batch_mask,
                temperature=temperature,
                stochastic=False,
            )

        a_s = self.alpha(s, temperature=temperature)[batch_mask]
        a_t = self.alpha(t, temperature=temperature)[batch_mask]
        sig_s = self.sigma(s, temperature=temperature)[batch_mask]
        sig_t = self.sigma(t, temperature=temperature)[batch_mask]

        eps_hat = -sig_s * score_s
        x0_hat = (xs - sig_s * eps_hat) / torch.clamp(a_s, min=1e-12)
        xt = a_t * x0_hat + sig_t * eps_hat
        return xt

    @torch.no_grad()
    def sample_ode(
        self,
        x0: torch.Tensor,
        batch_mask: torch.Tensor,
        score_fn,
        *,
        n_steps: int = 100,
        t_start: float = 0.0,
        t_end: float = 1.0,
        temperature=None,
    ) -> torch.Tensor:
        """ODE sampler from noise (t_start) to clean (t_end).

        Args:
          x0: initial coordinates at t_start (typically a noisy prior sample)
          batch_mask: (N,) mapping nodes -> example
          score_fn: callable(score_fn(x, t_array, batch_mask, temperature) -> score)
            - must return score with shape (N,3)
            - must accept t_array shaped (B,1)
          n_steps: number of Euler steps
        """
        assert n_steps > 0
        device = x0.device
        n_samples = int(batch_mask.max().item()) + 1 if batch_mask.numel() > 0 else 0

        x = x0
        dt = (t_end - t_start) / float(n_steps)
        for i in range(n_steps):
            s_val = t_start + i * dt
            t_val = t_start + (i + 1) * dt
            s = torch.full((n_samples, 1), s_val, device=device, dtype=x.dtype)
            t = torch.full((n_samples, 1), t_val, device=device, dtype=x.dtype)
            score_s = score_fn(x, s, batch_mask, temperature)
            x = self.ode_step(x, score_s, s=s, t=t, batch_mask=batch_mask, temperature=temperature)
        return x


################################
# Score-based diffusion (h/logits)
################################


class CategoricalLogitScoreDiffusion:
    """Score-based diffusion utilities for categorical variables via logit space.

    This implements a simple VE-style diffusion on *logits* y (R^K):

      y_t = y_clean + sigma(t) * eps,  eps ~ N(0, I)

    where the categorical distribution is obtained with softmax(y).

    Time convention matches DrugFlow:
      - t=0 : max noise (prior)
      - t=1 : clean (data)

    Notes:
      - This is a continuous relaxation of categorical diffusion.
      - It is meant to coexist with (not replace) the existing Markov bridge.
      - If you use this sampler, the network should be trained to predict a score in
        logit space. For now, EnergyForceDiffusion can *use* it for sampling, but
        training is still Markov-bridge CE/VLB.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        n_classes: int = 8,
        *,
        sde: SDEParams | None = None,
    ):
        self.n_classes = int(n_classes)

        # Backward-compatible defaults: VE schedule controlled by sigma_min/max
        if sde is None:
            sde = SDEParams(kind="ve", sigma_min=float(sigma_min), sigma_max=float(sigma_max))
        self.sde = sde

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return 1.0 - t

    def alpha(self, t: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        if self.sde.kind == "ve":
            return torch.ones_like(t)
        if self.sde.kind == "vp":
            tau = self._tau(t)

            if temperature is not None and not isinstance(temperature, torch.Tensor):
                temperature = torch.as_tensor(temperature, device=t.device, dtype=t.dtype)
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.to(device=t.device, dtype=t.dtype)
                if temperature.numel() == 1:
                    temperature = temperature.expand_as(t)
                T_ref = 1.0
                beta_scale = torch.clamp(temperature / T_ref, min=1e-6)
            else:
                beta_scale = torch.ones_like(t)

            beta0 = torch.as_tensor(self.sde.beta_min, device=t.device, dtype=t.dtype) * beta_scale
            beta1 = torch.as_tensor(self.sde.beta_max, device=t.device, dtype=t.dtype) * beta_scale
            int_beta = beta0 * tau + 0.5 * (beta1 - beta0) * tau * tau
            return torch.exp(-0.5 * int_beta)
        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")

    def sigma(self, t: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
        t = torch.clamp(t, 0.0, 1.0)

        if temperature is not None and not isinstance(temperature, torch.Tensor):
            temperature = torch.as_tensor(temperature, device=t.device, dtype=t.dtype)
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.to(device=t.device, dtype=t.dtype)
            if temperature.numel() == 1:
                temperature = temperature.expand_as(t)
            T_ref = 1.0
            temp_scale = torch.sqrt(torch.clamp(temperature / T_ref, min=1e-6))
        else:
            temp_scale = None

        if self.sde.kind == "ve":
            sigma = float(self.sde.sigma_max) * (float(self.sde.sigma_min) / float(self.sde.sigma_max)) ** t
            if temp_scale is not None:
                sigma = sigma * temp_scale
            return sigma
        if self.sde.kind == "vp":
            a = self.alpha(t, temperature=temperature)
            sig = torch.sqrt(torch.clamp(1.0 - a * a, min=1e-12))
            sig = sig * float(self.sde.vp_sigma_scale)
            return sig
        raise ValueError(f"Unknown SDE kind: {self.sde.kind}")

    @staticmethod
    def one_hot_to_logits(one_hot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.clamp(one_hot, min=eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.log(probs)

    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def sample_zt(self, y_clean: torch.Tensor, t: torch.Tensor, batch_mask: torch.Tensor, temperature=None):
        eps = torch.randn_like(y_clean)
        a_t = self.alpha(t, temperature=temperature)[batch_mask]
        sigma_t = self.sigma(t, temperature=temperature)[batch_mask]
        yt = a_t * y_clean + sigma_t * eps
        return yt, eps

    def sample_prior(self, batch_mask: torch.Tensor, temperature=None, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Prior for sampling at t=0: y_0 ~ N(0, sigma_max^2 I)."""
        if batch_mask.numel() == 0:
            return torch.empty((0, self.n_classes), device=batch_mask.device, dtype=dtype or torch.float32)

        device = batch_mask.device
        if dtype is None:
            dtype = torch.float32
        n_samples = int(batch_mask.max().item()) + 1
        sigma_0 = self.sigma(torch.zeros((n_samples, 1), device=device, dtype=dtype), temperature=temperature)
        eps = torch.randn((batch_mask.size(0), self.n_classes), device=device, dtype=dtype)
        return eps * sigma_0[batch_mask]

    def score_target(self, yt: torch.Tensor, y_clean: torch.Tensor, t: torch.Tensor, batch_mask: torch.Tensor, temperature=None):
        a_t = self.alpha(t, temperature=temperature)[batch_mask]
        sigma_t = self.sigma(t, temperature=temperature)[batch_mask]
        return (a_t * y_clean - yt) / torch.clamp(sigma_t * sigma_t, min=1e-12)

    def reduce_loss(self, per_node: torch.Tensor, batch_mask: torch.Tensor, reduce: str):
        assert reduce in {"mean", "sum", "none"}
        if reduce == "mean":
            return scatter_mean(per_node, batch_mask, dim=0)
        if reduce == "sum":
            return scatter_add(per_node, batch_mask, dim=0)
        return per_node

    def compute_loss(
        self,
        score_pred: torch.Tensor,
        y_clean: torch.Tensor,
        yt: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        *,
        reduce: str = "mean",
        temperature=None,
        weight_by_sigma: bool = False,
    ) -> torch.Tensor:
        target = self.score_target(yt, y_clean, t, batch_mask, temperature=temperature)
        per_node = torch.sum((score_pred - target) ** 2, dim=-1)
        if weight_by_sigma:
            sigma_t = self.sigma(t, temperature=temperature)[batch_mask].squeeze(-1)
            per_node = per_node * (sigma_t * sigma_t)
        return self.reduce_loss(per_node, batch_mask, reduce)

    def reverse_step(
        self,
        ys: torch.Tensor,
        score_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        *,
        temperature=None,
        stochastic: bool = True,
    ) -> torch.Tensor:
        if self.sde.kind == "vp":
            return self.ode_step(ys, score_s, s=s, t=t, batch_mask=batch_mask, temperature=temperature)

        sigma_s = self.sigma(s, temperature=temperature)[batch_mask]
        sigma_t = self.sigma(t, temperature=temperature)[batch_mask]
        delta = torch.clamp(sigma_s * sigma_s - sigma_t * sigma_t, min=0.0)
        drift = delta * score_s
        if stochastic:
            noise = torch.randn_like(ys)
            diffusion = torch.sqrt(torch.clamp(delta, min=0.0)) * noise
        else:
            diffusion = torch.zeros_like(ys)
        return ys + drift + diffusion

    def ode_step(
        self,
        ys: torch.Tensor,
        score_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        batch_mask: torch.Tensor,
        *,
        temperature=None,
    ) -> torch.Tensor:
        if self.sde.kind == "ve":
            return self.reverse_step(
                ys,
                score_s,
                s=s,
                t=t,
                batch_mask=batch_mask,
                temperature=temperature,
                stochastic=False,
            )

        a_s = self.alpha(s, temperature=temperature)[batch_mask]
        a_t = self.alpha(t, temperature=temperature)[batch_mask]
        sig_s = self.sigma(s, temperature=temperature)[batch_mask]
        sig_t = self.sigma(t, temperature=temperature)[batch_mask]
        eps_hat = -sig_s * score_s
        y0_hat = (ys - sig_s * eps_hat) / torch.clamp(a_s, min=1e-12)
        yt = a_t * y0_hat + sig_t * eps_hat
        return yt

    @torch.no_grad()
    def sample_ode(
        self,
        y0: torch.Tensor,
        batch_mask: torch.Tensor,
        score_fn,
        *,
        n_steps: int = 100,
        t_start: float = 0.0,
        t_end: float = 1.0,
        temperature=None,
    ) -> torch.Tensor:
        assert n_steps > 0
        device = y0.device
        n_samples = int(batch_mask.max().item()) + 1 if batch_mask.numel() > 0 else 0

        y = y0
        dt = (t_end - t_start) / float(n_steps)
        for i in range(n_steps):
            s_val = t_start + i * dt
            t_val = t_start + (i + 1) * dt
            s = torch.full((n_samples, 1), s_val, device=device, dtype=y.dtype)
            t = torch.full((n_samples, 1), t_val, device=device, dtype=y.dtype)
            score_s = score_fn(y, s, batch_mask, temperature)
            y = self.ode_step(y, score_s, s=s, t=t, batch_mask=batch_mask, temperature=temperature)
        return y
