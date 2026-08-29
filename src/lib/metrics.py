"""Reference implementation of the metrics defined in Section 3.3 of the paper:
Regret Dominance Score (RDS), neuron categorisation, Group Impact Coefficient
(GIC) and group mutual information.

How this relates to the pipeline:

* `05_train_probe_rds_gic.py` carries its own inline copies of `compute_rds`
  (there called `compute_RDS_corrected`), `classify_neurons` and
  `group_mutual_information` (there called `compute_mutual`). Those inline
  copies are what the pipeline actually executes. They were checked against the
  functions here and produce element-wise identical output.
* `group_impact_coefficient` has no counterpart in stage 5. Stage 5 records the
  probe accuracy after each single-group and combined-group ablation; GIC is
  obtained by feeding those recorded accuracies to the function here.

So this module is the readable statement of the equations, plus the only
implementation of Eq. 3. It is not imported by the pipeline.
"""

import numpy as np
from sklearn.metrics import mutual_info_score

# Value that a deactivated neuron's activation is set to (paper, Eq. 3).
DEACTIVATION_VALUE = -1.0

GROUP_NAMES = ("RegretD", "NonRegretD", "DualD")


def compute_rds(activations_regret, activations_no_regret, epsilon=1e-8):
    """Regret Dominance Score per neuron (paper Eq. 1).

    RDS_k = mean(Z_regret[:, k]) / (mean(Z_regret[:, k]) + mean(Z_no_regret[:, k]) + eps)

    The two activation matrices are unpaired, so the ratio is taken between the
    per-neuron means rather than averaged over per-sample ratios.

    Args:
        activations_regret: array (num_regret_samples, num_neurons).
        activations_no_regret: array (num_no_regret_samples, num_neurons).
        epsilon: numerical stabiliser.

    Returns:
        array (num_neurons,) of RDS values.
    """
    regret_mean = np.mean(activations_regret, axis=0)
    no_regret_mean = np.mean(activations_no_regret, axis=0)
    return regret_mean / (regret_mean + no_regret_mean + epsilon)


def classify_neurons(rds, threshold_std=1.0):
    """Partition neurons into the three functional groups (paper Eq. 2).

    RegretD     : RDS >  mu + tau * sigma
    NonRegretD  : RDS <  mu - tau * sigma
    DualD       : otherwise

    The three sets are disjoint and cover every neuron.

    Args:
        rds: array (num_neurons,) produced by `compute_rds`.
        threshold_std: tau, the number of standard deviations used as margin.

    Returns:
        dict with keys 'regret_dominant', 'no_regret_dominant', 'mixed'
        (the last one is DualD), each an array of neuron indices.
    """
    rds = rds.astype(np.float32)
    mu = np.mean(rds)
    sigma = np.std(rds)

    regret_dominant = np.where(rds > mu + threshold_std * sigma)[0]
    no_regret_dominant = np.where(rds < mu - threshold_std * sigma)[0]
    mixed = np.where(
        (rds >= mu - threshold_std * sigma) & (rds <= mu + threshold_std * sigma)
    )[0]

    return {
        "regret_dominant": regret_dominant,
        "no_regret_dominant": no_regret_dominant,
        "mixed": mixed,
    }


def group_impact_coefficient(acc_combined, acc_individual, acc_baseline):
    """Group Impact Coefficient (paper Eq. 3).

    n == 1:  GIC = Acc(Z - S1) / Acc(Z)
    n >= 2:  GIC = Acc(Z - union(S_i)) / mean_i(Acc(Z - S_i))

    GIC < 1 means the joint deactivation hurts the probe more than the
    individual deactivations do on average, i.e. the groups act compositionally.

    Args:
        acc_combined: accuracy after deactivating the union of the groups.
        acc_individual: sequence of accuracies, one per group deactivated alone.
        acc_baseline: accuracy with every neuron active, Acc(Z).

    Returns:
        float GIC value, or None when the denominator is zero.
    """
    acc_individual = list(acc_individual)
    if len(acc_individual) == 1:
        denominator = acc_baseline
    else:
        denominator = float(np.mean(acc_individual))
    if denominator == 0:
        return None
    return float(acc_combined) / denominator


def _entropy(p):
    """Shannon entropy in bits of a discrete distribution."""
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def group_mutual_information(regret_indices, no_regret_indices, mixed_indices,
                             all_activations, num_bins=20):
    """Normalised mutual information between the three neuron groups.

    Each group is summarised by the mean activation of its neurons per sample,
    discretised into `num_bins` bins over [0, 1]; the mutual information between
    two groups is then normalised by the geometric mean of their entropies.

    Args:
        regret_indices / no_regret_indices / mixed_indices: neuron index arrays.
        all_activations: array (num_samples, num_neurons).
        num_bins: number of discretisation bins.

    Returns:
        (mi_matrix, group_names) where mi_matrix is a 3x3 array ordered as
        GROUP_NAMES.
    """

    def calc_group_mi(group_a, group_b):
        if len(group_a) == 0 or len(group_b) == 0:
            return 0.0
        mean_a = np.mean(all_activations[:, group_a], axis=1)
        mean_b = np.mean(all_activations[:, group_b], axis=1)

        bins = np.linspace(0, 1, num_bins)
        disc_a = np.digitize(mean_a, bins)
        disc_b = np.digitize(mean_b, bins)

        mi = mutual_info_score(disc_a, disc_b)
        entropy_a = _entropy(np.bincount(disc_a) / len(disc_a))
        entropy_b = _entropy(np.bincount(disc_b) / len(disc_b))
        return mi / np.sqrt(entropy_a * entropy_b)

    groups = [regret_indices, no_regret_indices, mixed_indices]
    mi_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mi_matrix[i, j] = calc_group_mi(groups[i], groups[j])

    return mi_matrix, list(GROUP_NAMES)
