from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np

from sp_bai.algorithms.gopt import run_g_opt as pkg_run_g_opt
from sp_bai.algorithms.rage import run_rage as pkg_run_rage
from sp_bai.algorithms.sbe import run_sbe as pkg_run_sbe
from sp_bai.algorithms.spbai import run_sp_bai as pkg_run_sp_bai
from sp_bai.core.designs import deo_policy as pkg_deo_policy
from sp_bai.core.designs import deo_policy_global as pkg_deo_policy_global
from sp_bai.core.designs import xor_policy as pkg_xor_policy
from sp_bai.core.designs import xy_design_optimal_lowrank as pkg_xy_design_optimal_lowrank
from sp_bai.core.metrics import vari_cov_XY as pkg_vari_cov_XY
from sp_bai.core.phases import ols_phase_deterministic_linear as pkg_ols_phase_deterministic_linear
from sp_bai.core.phases import ortho_phase_linear as pkg_ortho_phase_linear
from sp_bai.core.utils import fmt_sec


@dataclass
class SyntheticLinearConfig:
    delta: float
    R1: float = 1 / 3
    R2: float = 1 / 3
    n_simul: int = 100
    seed: int = 1
    max_phases: int = 10
    rage_noise_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    lowrank_tol: float = 1e-12
    use_lowrank: bool = True
    print_every_sim: int = 1
    verbose: bool = False
    show_progress: bool = True
    epsilon_round: float = 0.2
    c_rage: float = 1.0
    general_sample_cap: Optional[int] = None
    sbe_sample_cap: Optional[int] = None
    title: str = "Simulation Results Summary"
    subtitle: str = "== Running experiments (intrinsic low-rank design enabled) =="

    def algo_order(self):
        rage_keys = [f"RAGE(sigma={sigma})" for sigma in self.rage_noise_sigmas]
        return ["SP-BAI", "G-Opt", "SBE"] + rage_keys


def default_nu_t(t):
    return 1.0 + math.sin(2.0 * float(t))


def make_synthetic_reward(theta_star, nuisance_fn=default_nu_t, noise_sigma=1.0):
    def reward_fn(_arm_index, arm_feature, t, rng_local):
        return float(arm_feature @ theta_star + nuisance_fn(t) + rng_local.normal(0, noise_sigma))

    return reward_fn


def _run_sbe(X, theta_star, rng, config, nuisance_fn):
    reward_fn = make_synthetic_reward(theta_star, nuisance_fn=nuisance_fn, noise_sigma=1.0)
    phase_fn = lambda X_local, delta_local, p_local, n_local, rng_local, t0=1: pkg_ortho_phase_linear(
        X_local, delta_local, p_local, n_local, rng_local, reward_fn, t0=t0
    )
    policy_fn = lambda X_local, A_local, rng_local: pkg_deo_policy(
        X_local,
        A_local,
        rng=rng_local,
        prefer_slsqp=True,
        use_lowrank=config.use_lowrank,
        lowrank_tol=config.lowrank_tol,
        random_anchor=False,
    )
    return pkg_run_sbe(
        X,
        config.delta,
        config.R1,
        config.R2,
        rng,
        phase_fn,
        policy_fn,
        max_phases=config.max_phases,
        verbose=config.verbose,
        sample_cap=config.sbe_sample_cap if config.sbe_sample_cap is not None else config.general_sample_cap,
    )


def _run_g_opt(X, theta_star, rng, config, nuisance_fn):
    reward_fn = make_synthetic_reward(theta_star, nuisance_fn=nuisance_fn, noise_sigma=1.0)
    phase_fn = lambda X_local, delta_local, p_local, n_local, rng_local, t0=1: pkg_ortho_phase_linear(
        X_local, delta_local, p_local, n_local, rng_local, reward_fn, t0=t0
    )
    policy_fn = lambda X_local, A_local, rng_local: pkg_deo_policy(
        X_local,
        A_local,
        rng=rng_local,
        prefer_slsqp=True,
        use_lowrank=config.use_lowrank,
        lowrank_tol=config.lowrank_tol,
        random_anchor=False,
    )
    return pkg_run_g_opt(
        X,
        config.delta,
        config.R1,
        config.R2,
        rng,
        theta_star,
        phase_fn,
        policy_fn,
        verbose=config.verbose,
        sample_cap=config.general_sample_cap,
    )


def _run_sp_bai(X, theta_star, p_G, rng, config, nuisance_fn):
    reward_fn = make_synthetic_reward(theta_star, nuisance_fn=nuisance_fn, noise_sigma=1.0)
    phase_fn = lambda X_local, delta_local, p_local, n_local, rng_local, t0=1: pkg_ortho_phase_linear(
        X_local, delta_local, p_local, n_local, rng_local, reward_fn, t0=t0
    )
    xor_policy_fn = lambda X_local, A_local, rng_local, anchor=None: pkg_xor_policy(
        X_local,
        A_local,
        rng_local,
        anchor=anchor,
        use_lowrank=config.use_lowrank,
        lowrank_tol=config.lowrank_tol,
    )
    return pkg_run_sp_bai(
        X,
        config.delta,
        config.R1,
        config.R2,
        rng,
        p_G,
        phase_fn,
        xor_policy_fn,
        vari_fn=pkg_vari_cov_XY,
        max_phases=config.max_phases,
        verbose=config.verbose,
        sample_cap=config.general_sample_cap,
    )


def _run_rage(X, theta_star, rng, config, nuisance_fn, noise_sigma):
    reward_fn = make_synthetic_reward(theta_star, nuisance_fn=nuisance_fn, noise_sigma=noise_sigma)
    ols_phase_fn = lambda X_local, schedule_local, rng_local, t0=1: pkg_ols_phase_deterministic_linear(
        X_local, schedule_local, rng_local, reward_fn, t0=t0
    )
    xy_design_fn = lambda X_local, A_local: pkg_xy_design_optimal_lowrank(
        X_local,
        A_local,
        iters=500,
        use_lowrank=config.use_lowrank,
        lowrank_tol=config.lowrank_tol,
    )
    return pkg_run_rage(
        X,
        config.delta,
        rng,
        ols_phase_fn,
        xy_design_fn,
        epsilon_round=config.epsilon_round,
        max_phases=config.max_phases,
        c_rage=config.c_rage,
        sample_scale=noise_sigma**2,
        verbose=config.verbose,
        sample_cap=config.general_sample_cap,
    )


def evaluate_suite(X, theta_star, config, base_seed, nuisance_fn=default_nu_t, p_G=None):
    if p_G is None:
        p_G = pkg_deo_policy_global(
            X,
            prefer_slsqp=True,
            use_lowrank=config.use_lowrank,
            lowrank_tol=config.lowrank_tol,
        )

    a_opt = int(np.argmax(X @ theta_star))
    suite_results = {}

    rng_our = np.random.default_rng(base_seed + 17)
    t0 = time.perf_counter()
    pulls, hat_arm = _run_sp_bai(X, theta_star, p_G, rng_our, config, nuisance_fn)
    suite_results["SP-BAI"] = {"T": pulls, "err": int(hat_arm != a_opt), "time": time.perf_counter() - t0}

    rng_gopt = np.random.default_rng(base_seed + 23)
    t0 = time.perf_counter()
    pulls, hat_arm = _run_g_opt(X, theta_star, rng_gopt, config, nuisance_fn)
    suite_results["G-Opt"] = {"T": pulls, "err": int(hat_arm != a_opt), "time": time.perf_counter() - t0}

    rng_sbe = np.random.default_rng(base_seed + 31)
    t0 = time.perf_counter()
    pulls, hat_arm = _run_sbe(X, theta_star, rng_sbe, config, nuisance_fn)
    suite_results["SBE"] = {"T": pulls, "err": int(hat_arm != a_opt), "time": time.perf_counter() - t0}

    for idx, sigma in enumerate(config.rage_noise_sigmas):
        key = f"RAGE(sigma={sigma})"
        rng_rage = np.random.default_rng(base_seed + 43 + 11 * idx)
        t0 = time.perf_counter()
        pulls, hat_arm = _run_rage(X, theta_star, rng_rage, config, nuisance_fn, noise_sigma=sigma)
        suite_results[key] = {"T": pulls, "err": int(hat_arm != a_opt), "time": time.perf_counter() - t0}

    return suite_results


def make_result_store(algo_order):
    return {
        name: {"T": [], "err": [], "time": 0.0}
        for name in algo_order
    }


def update_result_store(store, suite_results):
    for name, result in suite_results.items():
        store[name]["T"].append(result["T"])
        store[name]["err"].append(result["err"])
        store[name]["time"] += result["time"]


def summarize_result_store(store):
    summary = {}
    for name, result in store.items():
        n = len(result["T"])
        summary[name] = {
            "avg_T": float(np.mean(result["T"])) if n else 0.0,
            "std_T": float(np.std(result["T"], ddof=1)) if n > 1 else 0.0,
            "avg_err": float(np.mean(result["err"])) if n else 0.0,
            "std_err": float(np.std(result["err"], ddof=1)) if n > 1 else 0.0,
            "total_time": result["time"],
        }
    return summary


def print_dimension_table(label, summary, algo_order):
    print(label)
    print(
        f"{'Algorithm':<16} | {'Avg. T':<18} | {'Std. T':<18} | "
        f"{'Avg. Error Prob.':<18} | {'Std. Error Prob.':<18}"
    )
    print("-" * 95)
    for name in algo_order:
        stats = summary[name]
        print(
            f"{name:<16} | {stats['avg_T']:<18.2f} | {stats['std_T']:<18.2f} | "
            f"{stats['avg_err']:<18.4f} | {stats['std_err']:<18.4f}"
        )
    print("-" * 95)


def print_total_timing(summary, algo_order, grand_total):
    print("\n== TOTAL TIMING SUMMARY ==")
    for name in algo_order:
        print(f"  {name:<16}: total {fmt_sec(summary[name]['total_time'])}")
    print(f"\nGrand total elapsed: {fmt_sec(grand_total)}")
