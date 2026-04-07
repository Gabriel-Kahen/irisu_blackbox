from __future__ import annotations

from types import MethodType

import torch as th
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MultiInputLstmPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)


class _NoopBiasMixin:
    def __init__(self, *args, noop_action_bias: float = 0.0, **kwargs) -> None:
        self.noop_action_bias = float(noop_action_bias)
        super().__init__(*args, **kwargs)

    def _apply_noop_bias(self, action_logits: th.Tensor) -> th.Tensor:
        if self.noop_action_bias == 0.0 or action_logits.shape[-1] <= 0:
            return action_logits
        biased_logits = action_logits.clone()
        biased_logits[..., 0] = biased_logits[..., 0] + self.noop_action_bias
        return biased_logits

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        if isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(
                action_logits=self._apply_noop_bias(mean_actions)
            )
        if isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        raise ValueError("Invalid action distribution")


class NoopBiasedCnnLstmPolicy(_NoopBiasMixin, CnnLstmPolicy):
    pass


class NoopBiasedMultiInputLstmPolicy(_NoopBiasMixin, MultiInputLstmPolicy):
    pass


def install_noop_bias(policy, noop_action_bias: float) -> None:
    policy.noop_action_bias = float(noop_action_bias)
    policy._apply_noop_bias = MethodType(_NoopBiasMixin._apply_noop_bias, policy)
    policy._get_action_dist_from_latent = MethodType(
        _NoopBiasMixin._get_action_dist_from_latent,
        policy,
    )
