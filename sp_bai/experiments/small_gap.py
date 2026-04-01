from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np

from sp_bai.core.designs import deo_policy_global as pkg_deo_policy_global
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
class SmallGapExperimentConfig(SyntheticLinearConfig):
    dims: tuple[int, ...] = (10, 20)
    alpha: float = 0.2
    delta: float = 1.0 / 20.0
    R1: float = 1 / 3
    R2: float = 1 / 3
    n_simul: int = 100
    seed: int = 100
    lowrank_tol: float = 1e-10
    sbe_sample_cap: Optional[int] = 9_000_000
    general_sample_cap: Optional[int] = None
    title: str = "Simulation Results Summary"
    subtitle: str = "== Running experiments (low-rank design enabled) =="


def make_feature_set(d, alpha):
    E = np.eye(d)
    z = np.zeros(d)
    z[0] = math.cos(alpha)
    if d >= 2:
        z[1] = math.sin(alpha)
    return np.vstack([E, z[None, :]])


def theta_star_vec(d):
    theta = np.zeros(d)
    theta[0] = 2.0
    return theta


def run_small_gap_experiment(config: SmallGapExperimentConfig):
    algo_order = config.algo_order()
    overall_store = make_result_store(algo_order)

    print(f"\n{config.subtitle}")
    print("\n" + "=" * 85)
    print(f" {config.title}")
    print("=" * 85)

    grand_t0 = time.perf_counter()
    for d in config.dims:
        dim_t0 = time.perf_counter()
        dim_store = make_result_store(algo_order)

        X = make_feature_set(d, config.alpha)
        theta_star = theta_star_vec(d)
        p_G = pkg_deo_policy_global(
            X,
            prefer_slsqp=True,
            use_lowrank=config.use_lowrank,
            lowrank_tol=config.lowrank_tol,
        )
        noise_seed_base = config.seed * 2_000_033 + 97 * d + 314_159

        for sim_idx in range(config.n_simul):
            base_seed = noise_seed_base + sim_idx * 10_000_019
            suite_results = evaluate_suite(
                X,
                theta_star,
                config,
                base_seed=base_seed,
                nuisance_fn=default_nu_t,
                p_G=p_G,
            )
            update_result_store(dim_store, suite_results)
            update_result_store(overall_store, suite_results)

            if config.show_progress and ((sim_idx + 1) % max(1, config.print_every_sim) == 0):
                for name in algo_order:
                    print(f"[sim {sim_idx + 1}/{config.n_simul}] {name}", flush=True)

        summary = summarize_result_store(dim_store)
        print_dimension_table(f"--- Dimension d = {d} ---", summary, algo_order)
        print(f"Elapsed for d={d}: {fmt_sec(time.perf_counter() - dim_t0)}\n")

    overall_summary = summarize_result_store(overall_store)
    print_total_timing(overall_summary, algo_order, time.perf_counter() - grand_t0)


SmallGapInstanceConfig = SmallGapExperimentConfig


def run_small_gap_instance(config: SmallGapExperimentConfig):
    return run_small_gap_experiment(config)
