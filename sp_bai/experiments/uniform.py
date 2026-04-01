from dataclasses import dataclass, field
import time
from typing import Optional

import numpy as np

from sp_bai.core.utils import fmt_sec

from .synthetic_linear import (
    SyntheticLinearConfig,
    default_nu_t,
    evaluate_suite,
    make_result_store,
    print_dimension_table,
    print_total_timing,
    summarize_result_store,
    update_result_store,
)


@dataclass
class UniformExperimentConfig(SyntheticLinearConfig):
    dims: tuple[int, ...] = (10,)
    k_arms: int = 100
    alpha: float = 0.2
    delta: float = 1.0 / 10.0
    R1: float = 1 / 3
    R2: float = 1 / 3
    n_simul: int = 100
    seed: int = 1
    lowrank_tol: float = 1e-12
    general_sample_cap: Optional[int] = 5_000_000
    sbe_sample_cap: Optional[int] = 5_000_000
    title: str = "Simulation Results Summary (Uniform Sphere Features)"
    subtitle: str = "== Running experiments (intrinsic low-rank design enabled) =="


def make_feature_set(d, k_arms, rng):
    Z = rng.normal(size=(k_arms, d))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return Z / norms


def theta_star_vec(d):
    theta = np.zeros(d)
    theta[0] = 2.0
    return theta


def run_uniform_experiment(config: UniformExperimentConfig):
    algo_order = config.algo_order()
    overall_store = make_result_store(algo_order)

    print(f"\n{config.subtitle}")
    print("\n" + "=" * 95)
    print(" " * 20 + config.title)
    print("=" * 95)

    grand_t0 = time.perf_counter()
    for d in config.dims:
        dim_t0 = time.perf_counter()
        dim_store = make_result_store(algo_order)

        feature_seed_base = config.seed * 1_000_003 + 13 * d + 7_919 * config.k_arms
        noise_seed_base = config.seed * 2_000_033 + 37 * d + 2_909 * config.k_arms + 424_242

        for sim_idx in range(config.n_simul):
            feature_rng = np.random.default_rng(feature_seed_base + sim_idx)
            X = make_feature_set(d, config.k_arms, feature_rng)
            theta_star = theta_star_vec(d)
            suite_results = evaluate_suite(
                X,
                theta_star,
                config,
                base_seed=noise_seed_base + sim_idx,
                nuisance_fn=default_nu_t,
            )
            update_result_store(dim_store, suite_results)
            update_result_store(overall_store, suite_results)

            if config.show_progress and ((sim_idx + 1) % max(1, config.print_every_sim) == 0):
                for name in algo_order:
                    print(f"[sim {sim_idx + 1}/{config.n_simul}] {name}", flush=True)

        summary = summarize_result_store(dim_store)
        print_dimension_table(f"--- Dimension d = {d}, K = {config.k_arms} ---", summary, algo_order)
        print(f"Elapsed for d={d}, K={config.k_arms}: {fmt_sec(time.perf_counter() - dim_t0)}\n")

    overall_summary = summarize_result_store(overall_store)
    print_total_timing(overall_summary, algo_order, time.perf_counter() - grand_t0)


UniformFeatureExperimentConfig = UniformExperimentConfig


def run_uniform_feature_experiment(config: UniformExperimentConfig):
    return run_uniform_experiment(config)
