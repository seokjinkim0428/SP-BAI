from collections import Counter
from dataclasses import dataclass
import math
import os
import time

import numpy as np
import pandas as pd

from sp_bai.algorithms.ae import run_ae as pkg_run_ae
from sp_bai.algorithms.lucb import run_lucb as pkg_run_lucb
from sp_bai.algorithms.sbe import run_sbe as pkg_run_sbe
from sp_bai.algorithms.spbai import run_sp_bai as pkg_run_sp_bai
from sp_bai.algorithms.spbai import run_sp_bai_budgeted as pkg_run_sp_bai_budgeted
from sp_bai.core.designs import deo_policy as pkg_deo_policy
from sp_bai.core.designs import deo_policy_global as pkg_deo_policy_global
from sp_bai.core.designs import xor_policy as pkg_xor_policy
from sp_bai.core.metrics import vari_cov_XY as pkg_vari_cov_XY
from sp_bai.core.phases import ortho_phase_linear as pkg_ortho_phase_linear


DEFAULT_ARM_IDS = (7, 8, 13, 15, 16, 17, 18, 19)
DEFAULT_JESTER_SUBSET_FILE = "jester_subset_50699_8.csv"
DEFAULT_JESTER_RATINGS_FILE = "jester_ratings.csv"


@dataclass
class ToyRankingConfig:
    data_file: str = DEFAULT_JESTER_SUBSET_FILE
    arm_ids: tuple[int, ...] = DEFAULT_ARM_IDS
    t_budget: int = 3000
    k_simulations: int = 100
    delta: float = 0.05
    lowrank_tol: float = 1e-12


@dataclass
class FixedConfidenceConfig:
    data_file: str = DEFAULT_JESTER_SUBSET_FILE
    arm_ids: tuple[int, ...] = DEFAULT_ARM_IDS
    delta: float = 0.1
    R1: float = 1 / 3
    R2: float = 1 / 3
    n_runs: int = 100
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    max_phases: int = 10
    lowrank_tol: float = 1e-12
    max_total_pulls_ae: int = 3_000_000
    verbose: bool = False


@dataclass
class OursOnlyConfig:
    data_file: str = DEFAULT_JESTER_SUBSET_FILE
    arm_ids: tuple[int, ...] = DEFAULT_ARM_IDS
    delta: float = 0.05
    R1: float = 1 / 3
    R2: float = 1 / 3
    n_runs: int = 100
    max_phases: int = 10
    lowrank_tol: float = 1e-12
    verbose: bool = False


@dataclass
class SyntheticSuiteConfig:
    K: int = 5
    eps: float = 0.05
    n_runs: int = 100
    delta: float = 0.1
    R1: float = 1 / 3
    R2: float = 1 / 3
    sigma_env: float = 1.0
    sigma_assumed_list: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    seed0: int = 12_345
    max_total_pulls: int = 2_000_000
    max_phases: int = 10
    lowrank_tol: float = 1e-12
    verbose: bool = False


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _data_candidates(filename):
    if os.path.isabs(filename):
        return [filename]

    repo_root = _repo_root()
    candidates = [
        filename,
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(repo_root, filename),
        os.path.join(repo_root, "data", filename),
        os.path.join("/content/drive/MyDrive/Colab Notebooks/ExpDes_OR_Codes", filename),
        os.path.join("/content/drive/MyDrive/Colab Notebooks/ExpDes_Journal", filename),
    ]
    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _resolve_existing_file(filename):
    for candidate in _data_candidates(filename):
        if os.path.exists(candidate):
            return candidate
    return None


def prepare_jester_subset(
    ratings_file=DEFAULT_JESTER_RATINGS_FILE,
    output_file=DEFAULT_JESTER_SUBSET_FILE,
    arm_ids=DEFAULT_ARM_IDS,
    verbose=True,
):
    """Create the complete-rating Jester subset used by the paper's real-data experiments."""
    ratings_path = _resolve_existing_file(ratings_file)
    if ratings_path is None:
        raise FileNotFoundError(
            f"Could not find the raw Jester ratings file '{ratings_file}'. "
            "Place it in the repository root, a local data/ folder, or a configured Colab Drive path."
        )

    arm_ids = tuple(int(arm_id) for arm_id in arm_ids)
    output_path = output_file if os.path.isabs(output_file) else os.path.abspath(output_file)

    if verbose:
        print(f"Preparing Jester subset from: {ratings_path}")
        print(f"Target arm IDs: {list(arm_ids)}")

    ratings = pd.read_csv(ratings_path)
    subset = ratings.loc[ratings["jokeId"].isin(arm_ids), ["userId", "jokeId", "rating"]].copy()
    counts = subset.groupby("userId")["jokeId"].nunique()
    complete_users = counts[counts == len(arm_ids)].index
    subset = subset.loc[subset["userId"].isin(complete_users)].copy()
    # Preserve the original row order from the raw ratings file so the generated
    # subset matches the reference subset file used throughout the repository.
    subset = subset.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    subset.to_csv(output_path, index=False)

    if verbose:
        print(
            f"Wrote {output_path} with {subset['userId'].nunique()} users, "
            f"{subset['jokeId'].nunique()} jokes, and {len(subset)} rows."
        )

    return output_path


def resolve_data_file(filename):
    existing_path = _resolve_existing_file(filename)
    if existing_path is not None:
        return existing_path

    if os.path.basename(filename) == DEFAULT_JESTER_SUBSET_FILE:
        generated_path = prepare_jester_subset(
            ratings_file=DEFAULT_JESTER_RATINGS_FILE,
            output_file=os.path.join(os.getcwd(), DEFAULT_JESTER_SUBSET_FILE),
            arm_ids=DEFAULT_ARM_IDS,
            verbose=True,
        )
        return generated_path

    candidates = _data_candidates(filename)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in the current workspace or Colab Drive.")


def _load_data_frame(data_file):
    path = resolve_data_file(data_file)
    print(f"Loading data from: {path}")
    return path, pd.read_csv(path)


def _build_rewards_lookup(df):
    rewards_lookup = {}
    for user_id, group in df.groupby("userId"):
        rewards_lookup[user_id] = pd.Series(group["rating"].values, index=group["jokeId"]).to_dict()
    return rewards_lookup, df["userId"].unique()


def _ground_truth(df, arm_ids):
    true_means = df.groupby("jokeId")["rating"].mean().reindex(arm_ids)
    true_best_joke_id = int(true_means.idxmax())
    true_best_arm_index = arm_ids.index(true_best_joke_id)
    return true_means, true_best_joke_id, true_best_arm_index


def _make_sigma_sq_dict(K, sigma_value):
    sigma_sq = float(sigma_value) ** 2
    return {i: sigma_sq for i in range(K)}


def _jester_reward_fn(all_user_ids, rewards_lookup, arm_index_to_joke_id):
    def reward_fn(arm_index, _arm_feature, _t, rng_local):
        user_id = rng_local.choice(all_user_ids)
        return rewards_lookup[user_id][arm_index_to_joke_id[arm_index]]

    return reward_fn


def _jester_bandit_reward_fn(all_user_ids, rewards_lookup, arm_index_to_joke_id):
    def reward_fn(arm_index, _t, rng_local):
        user_id = rng_local.choice(all_user_ids)
        return rewards_lookup[user_id][arm_index_to_joke_id[arm_index]]

    return reward_fn


def _run_sp_bai_jester(X, config, all_user_ids, rewards_lookup, rng, p_G):
    arm_index_to_joke_id = {i: joke_id for i, joke_id in enumerate(config.arm_ids)}
    reward_fn = _jester_reward_fn(all_user_ids, rewards_lookup, arm_index_to_joke_id)
    phase_fn = lambda X_local, delta_local, p_local, n_local, rng_local, t0=1: pkg_ortho_phase_linear(
        X_local, delta_local, p_local, int(n_local), rng_local, reward_fn, t0=t0
    )
    xor_policy_fn = lambda X_local, A_local, rng_local, anchor=None: pkg_xor_policy(
        X_local,
        A_local,
        rng_local,
        anchor=anchor,
        lowrank_tol=config.lowrank_tol,
    )
    return pkg_run_sp_bai(
        X,
        config.delta,
        config.R1,
        config.R2,
        rng,
        p_G=p_G,
        phase_fn=phase_fn,
        xor_policy_fn=xor_policy_fn,
        vari_fn=pkg_vari_cov_XY,
        max_phases=config.max_phases,
        verbose=config.verbose,
    )


def _run_sbe_jester(X, config, all_user_ids, rewards_lookup, rng):
    arm_index_to_joke_id = {i: joke_id for i, joke_id in enumerate(config.arm_ids)}
    reward_fn = _jester_reward_fn(all_user_ids, rewards_lookup, arm_index_to_joke_id)
    phase_fn = lambda X_local, delta_local, p_local, n_local, rng_local, t0=1: pkg_ortho_phase_linear(
        X_local, delta_local, p_local, int(n_local), rng_local, reward_fn, t0=t0
    )
    policy_fn = lambda X_local, A_local, rng_local: pkg_deo_policy(
        X_local,
        A_local,
        rng=rng_local,
        prefer_slsqp=True,
        lowrank_tol=config.lowrank_tol,
        random_anchor=False,
    )
    return pkg_run_sbe(
        X,
        config.delta,
        config.R1,
        config.R2,
        rng,
        phase_fn=phase_fn,
        policy_fn=policy_fn,
        max_phases=config.max_phases,
        verbose=config.verbose,
    )


def _run_lucb_jester(config, all_user_ids, rewards_lookup, rng, sigma_sq_dict):
    K = len(config.arm_ids)
    arm_index_to_joke_id = {i: joke_id for i, joke_id in enumerate(config.arm_ids)}
    reward_fn = _jester_bandit_reward_fn(all_user_ids, rewards_lookup, arm_index_to_joke_id)
    return pkg_run_lucb(K, config.delta, sigma_sq_dict, rng, reward_fn)


def _run_ae_jester(config, all_user_ids, rewards_lookup, rng, sigma_sq_dict, max_total_pulls):
    K = len(config.arm_ids)
    arm_index_to_joke_id = {i: joke_id for i, joke_id in enumerate(config.arm_ids)}
    reward_fn = _jester_bandit_reward_fn(all_user_ids, rewards_lookup, arm_index_to_joke_id)
    return pkg_run_ae(
        K,
        config.delta,
        sigma_sq_dict,
        rng,
        reward_fn,
        max_total_pulls=max_total_pulls,
        return_status=True,
    )


def run_toy_deo_vs_uniform(config: ToyRankingConfig):
    data_path, df = _load_data_frame(config.data_file)
    rewards_lookup, all_user_ids = _build_rewards_lookup(df)
    arm_ids = list(config.arm_ids)
    X = np.eye(len(arm_ids))
    p_deo = pkg_deo_policy_global(X, prefer_slsqp=True, lowrank_tol=config.lowrank_tol)
    true_means, true_best_joke_id, _ = _ground_truth(df, arm_ids)
    arm_index_to_joke_id = {i: joke_id for i, joke_id in enumerate(arm_ids)}

    deo_summary = {rank: Counter() for rank in range(1, len(arm_ids) + 1)}
    naive_summary = {rank: Counter() for rank in range(1, len(arm_ids) + 1)}

    def single_run(seed):
        # Keep this RNG stream fixed so the ranking comparison stays close
        # to the original notebook outputs.
        rng = np.random.RandomState(seed)
        users_t = rng.permutation(all_user_ids)[: config.t_budget]

        arm_rewards_naive = {i: [] for i in range(len(arm_ids))}
        for t in range(config.t_budget):
            arm_index = t % len(arm_ids)
            reward = rewards_lookup[users_t[t]][arm_index_to_joke_id[arm_index]]
            arm_rewards_naive[arm_index].append(reward)
        naive_results = []
        for i in range(len(arm_ids)):
            mean_i = np.mean(arm_rewards_naive[i]) if arm_rewards_naive[i] else 0.0
            naive_results.append((arm_index_to_joke_id[i], mean_i))
        naive_results.sort(key=lambda item: item[1], reverse=True)

        def reward_fn(arm_index, _arm_feature, t, _rng_local):
            user_id = users_t[t - 1]
            return rewards_lookup[user_id][arm_index_to_joke_id[arm_index]]

        theta_hat, _ = pkg_ortho_phase_linear(
            X,
            config.delta,
            p_deo,
            config.t_budget,
            rng,
            reward_fn,
            t0=1,
        )
        smart_results = [(arm_index_to_joke_id[i], value) for i, value in enumerate(X @ theta_hat)]
        smart_results.sort(key=lambda item: item[1], reverse=True)
        return smart_results, naive_results

    print(f"Starting {config.k_simulations} simulations with T_BUDGET={config.t_budget}...")
    print(f"Using data file: {data_path}")
    t0 = time.time()
    for sim_idx in range(config.k_simulations):
        smart_results, naive_results = single_run(sim_idx)
        for rank, (joke_id, _score) in enumerate(smart_results, 1):
            deo_summary[rank][joke_id] += 1
        for rank, (joke_id, _score) in enumerate(naive_results, 1):
            naive_summary[rank][joke_id] += 1
        print(f"Progress: {(sim_idx + 1) / config.k_simulations * 100:.1f}% ({sim_idx + 1}/{config.k_simulations})")

    print(f"\nTotal runtime for {config.k_simulations} simulations: {time.time() - t0:.2f} seconds")
    print("\n" + "=" * 60)
    print(f"--- FINAL SUMMARY (Across {config.k_simulations} Runs) ---")
    print("=" * 60)
    print(f"(Ground Truth Best Arm is: #{true_best_joke_id})\n")
    print("--- Ground Truth (full data) Ranking ---")
    gt_results = [(arm_ids[i], true_means.values[i]) for i in range(len(arm_ids))]
    gt_results.sort(key=lambda item: item[1], reverse=True)
    for rank, (joke_id, score) in enumerate(gt_results, 1):
        print(f"  Rank #{rank}: Joke #{joke_id} ({score:.4f})")

    print("\n--- G-Opt Strategy - Rank Frequency ---")
    for rank in range(1, len(arm_ids) + 1):
        line = f"  Rank #{rank}: "
        for joke_id, count in deo_summary[rank].most_common():
            line += f"Joke #{joke_id} ({count}x)  "
        print(line)

    print("\n--- Uniform-8 Strategy - Rank Frequency ---")
    for rank in range(1, len(arm_ids) + 1):
        line = f"  Rank #{rank}: "
        for joke_id, count in naive_summary[rank].most_common():
            line += f"Joke #{joke_id} ({count}x)  "
        print(line)

    print("\n" + "=" * 60)
    print("--- Overall Win Rate (Rank #1 = Ground Truth Best) ---")
    deo_wins = deo_summary[1][true_best_joke_id]
    naive_wins = naive_summary[1][true_best_joke_id]
    print(f"G-Opt:     {deo_wins} / {config.k_simulations} Wins ({deo_wins / config.k_simulations:.0%})")
    print(f"Uniform-8: {naive_wins} / {config.k_simulations} Wins ({naive_wins / config.k_simulations:.0%})")


def run_fixed_confidence_benchmark(config: FixedConfidenceConfig):
    data_path, df = _load_data_frame(config.data_file)
    rewards_lookup, all_user_ids = _build_rewards_lookup(df)
    arm_ids = list(config.arm_ids)
    X = np.eye(len(arm_ids))
    _, true_best_joke_id, true_best_arm_index = _ground_truth(df, arm_ids)
    p_G = pkg_deo_policy_global(X, prefer_slsqp=True, lowrank_tol=config.lowrank_tol)
    stats = {}

    def update_stats(name, pulls, success, elapsed):
        if name not in stats:
            stats[name] = {"successes": 0, "total_pulls": 0, "total_time": 0.0, "runs": 0}
        stats[name]["successes"] += int(success)
        stats[name]["total_pulls"] += pulls
        stats[name]["total_time"] += elapsed
        stats[name]["runs"] += 1

    for rep in range(config.n_runs):
        rng_sbe = np.random.default_rng(12_345 + rep * 1_000 + 1)
        rng_spbai = np.random.default_rng(12_345 + rep * 1_000 + 2)

        t0 = time.time()
        pulls, hat_arm = _run_sbe_jester(X, config, all_user_ids, rewards_lookup, rng_sbe)
        update_stats("SBE", pulls, hat_arm == true_best_arm_index, time.time() - t0)

        t0 = time.time()
        pulls, hat_arm = _run_sp_bai_jester(X, config, all_user_ids, rewards_lookup, rng_spbai, p_G)
        update_stats("SP-BAI", pulls, hat_arm == true_best_arm_index, time.time() - t0)

        for sigma_idx, sigma_val in enumerate(config.sigma_grid):
            sigma_label = f"{sigma_val:g}"
            sigma_sq_dict = _make_sigma_sq_dict(len(arm_ids), sigma_val)
            rng_lucb = np.random.default_rng(12_345 + rep * 1_000 + 3 + 2 * sigma_idx)
            rng_ae = np.random.default_rng(12_345 + rep * 1_000 + 4 + 2 * sigma_idx)

            t0 = time.time()
            hat_arm, total_pulls = _run_lucb_jester(config, all_user_ids, rewards_lookup, rng_lucb, sigma_sq_dict)
            update_stats(f"LUCB_sigma={sigma_label}", total_pulls, hat_arm == true_best_arm_index, time.time() - t0)

            t0 = time.time()
            hat_arm, total_pulls, converged = _run_ae_jester(
                config,
                all_user_ids,
                rewards_lookup,
                rng_ae,
                sigma_sq_dict,
                max_total_pulls=config.max_total_pulls_ae,
            )
            success = converged and (hat_arm == true_best_arm_index)
            update_stats(f"AE_sigma={sigma_label}", total_pulls, success, time.time() - t0)

        if (rep + 1) % max(1, config.n_runs // 10) == 0:
            print(f"Progress: {(rep + 1) / config.n_runs * 100:.1f}% ({rep + 1}/{config.n_runs})")

    print("\n" + "=" * 80)
    print(f"=== SUMMARY OVER {config.n_runs} RUNS (Delta={config.delta}) ===")
    print("=" * 80)
    print(f"Ground Truth Best Arm: Joke #{true_best_joke_id}\n")
    for name, summary in stats.items():
        succ_prob = summary["successes"] / summary["runs"]
        avg_pulls = summary["total_pulls"] / summary["runs"]
        avg_time = summary["total_time"] / summary["runs"]
        print(f"Algorithm: {name}")
        print(f"  - Success Probability: {succ_prob:.3f}  ({summary['successes']}/{summary['runs']})")
        print(f"  - Average Total Pulls: {avg_pulls:.1f}")
        print(f"  - Average Total Time : {avg_time:.3f} s")
        print("-" * 40)


def run_ours_only_benchmark(config: OursOnlyConfig):
    data_path, df = _load_data_frame(config.data_file)
    rewards_lookup, all_user_ids = _build_rewards_lookup(df)
    arm_ids = list(config.arm_ids)
    X = np.eye(len(arm_ids))
    _, true_best_joke_id, true_best_arm_index = _ground_truth(df, arm_ids)
    p_G = pkg_deo_policy_global(X, prefer_slsqp=True, lowrank_tol=config.lowrank_tol)
    stats = {"successes": 0, "total_pulls": 0, "total_time": 0.0}
    for rep in range(config.n_runs):
        rng_spbai = np.random.default_rng(12_345 + rep * 1_000 + 2)
        t0 = time.time()
        pulls, hat_arm = _run_sp_bai_jester(X, config, all_user_ids, rewards_lookup, rng_spbai, p_G)
        stats["successes"] += int(hat_arm == true_best_arm_index)
        stats["total_pulls"] += pulls
        stats["total_time"] += time.time() - t0
        if (rep + 1) % max(1, config.n_runs // 10) == 0:
            print(f"Progress: {(rep + 1) / config.n_runs * 100:.1f}% ({rep + 1}/{config.n_runs})")

    succ_prob = stats["successes"] / config.n_runs
    avg_pulls = stats["total_pulls"] / config.n_runs
    avg_time = stats["total_time"] / config.n_runs
    print("\n" + "=" * 80)
    print(f"=== SP-BAI ONLY (global p_G mix) OVER {config.n_runs} RUNS (Delta={config.delta}) ===")
    print("=" * 80)
    print(f"Ground Truth Best Arm: Joke #{true_best_joke_id}")
    print(f"Success Probability: {succ_prob:.3f} ({stats['successes']}/{config.n_runs})")
    print(f"Average Total Pulls: {avg_pulls:.1f}")
    print(f"Average Total Time : {avg_time:.3f} s")
    print("=" * 80)


def beat_drift(t):
    return 2.0 * math.sin(t / 2_000.0) * math.cos(t / 5_000.0)


def alternating_drift(t):
    return -3.0 if t % 3 == 1 else 3.0


def _make_theta_and_mu(K=10, eps=0.01):
    theta = np.ones(K) / 2
    theta[K - 2] = 1.0 + eps
    theta[K - 1] = 1.0
    mu = theta.copy()
    true_best = int(np.argmax(mu))
    return theta, mu, true_best


def run_synthetic_suite(config: SyntheticSuiteConfig, drift_fn, drift_label):
    X = np.eye(config.K)
    theta_star, mu, true_best = _make_theta_and_mu(K=config.K, eps=config.eps)
    p_G = pkg_deo_policy_global(X, prefer_slsqp=True, lowrank_tol=config.lowrank_tol)
    stats = {}

    def update_stats(name, pulls, success, elapsed):
        if name not in stats:
            stats[name] = {"runs": 0, "succ": 0, "pulls": 0, "time": 0.0}
        stats[name]["runs"] += 1
        stats[name]["succ"] += int(success)
        stats[name]["pulls"] += pulls
        stats[name]["time"] += elapsed

    def phase_reward_fn(arm_index, _arm_feature, t, rng_local):
        return float(mu[arm_index] + rng_local.normal(0.0, config.sigma_env) + drift_fn(t))

    def bandit_reward_fn(arm_index, t, rng_local):
        return float(mu[arm_index] + rng_local.normal(0.0, config.sigma_env) + drift_fn(t))

    phase_fn = lambda X_local, delta_local, p_local, n_local, rng_local, t0=1: pkg_ortho_phase_linear(
        X_local, delta_local, p_local, int(n_local), rng_local, phase_reward_fn, t0=t0
    )
    xor_policy_fn = lambda X_local, A_local, rng_local, anchor=None: pkg_xor_policy(
        X_local, A_local, rng_local, anchor=anchor, lowrank_tol=config.lowrank_tol
    )

    for rep in range(config.n_runs):
        rng_ours = np.random.default_rng(config.seed0 + rep * 1_000 + 1)
        t0 = time.time()
        pulls, hat_arm = pkg_run_sp_bai_budgeted(
            X,
            config.delta,
            config.R1,
            config.R2,
            rng_ours,
            p_G=p_G,
            phase_fn=phase_fn,
            xor_policy_fn=xor_policy_fn,
            vari_fn=pkg_vari_cov_XY,
            max_phases=config.max_phases,
            verbose=config.verbose,
            max_total_pulls=config.max_total_pulls,
        )
        update_stats("SP-BAI", pulls, hat_arm == true_best, time.time() - t0)

        for sig in config.sigma_assumed_list:
            sigma_sq = np.full(config.K, float(sig) ** 2)
            rng_lucb = np.random.default_rng(config.seed0 + rep * 1_000 + int(sig * 100) + 20)
            rng_ae = np.random.default_rng(config.seed0 + rep * 1_000 + int(sig * 100) + 40)

            t0 = time.time()
            hat_arm, total_pulls = pkg_run_lucb(
                config.K,
                config.delta,
                sigma_sq,
                rng_lucb,
                bandit_reward_fn,
                max_total_pulls=config.max_total_pulls,
            )
            update_stats(f"LUCB_sigma={sig}", total_pulls, hat_arm == true_best, time.time() - t0)

            t0 = time.time()
            hat_arm, total_pulls = pkg_run_ae(
                config.K,
                config.delta,
                sigma_sq,
                rng_ae,
                bandit_reward_fn,
                max_total_pulls=config.max_total_pulls,
            )
            update_stats(f"AE_sigma={sig}", total_pulls, hat_arm == true_best, time.time() - t0)

        if (rep + 1) % max(1, config.n_runs // 10) == 0:
            print(f"Progress: {(rep + 1) / config.n_runs * 100:.1f}% ({rep + 1}/{config.n_runs})")

    print("\n" + "=" * 90)
    print(
        f"SYNTHETIC MAB: K={config.K}, eps={config.eps}, sigma_env={config.sigma_env}, "
        f"delta={config.delta}, N_RUNS={config.n_runs}"
    )
    print(f"Reward drift: {drift_label}")
    print("theta* = (1/2,...,1/2, 1+eps, 1), x_i=e_i")
    print(f"True best arm index = {true_best} (0-based)")
    print(f"Safety max_total_pulls = {config.max_total_pulls}")
    print("=" * 90)
    for name, summary in stats.items():
        succ_prob = summary["succ"] / summary["runs"]
        err_prob = 1.0 - succ_prob
        avg_pulls = summary["pulls"] / summary["runs"]
        avg_time = summary["time"] / summary["runs"]
        print(f"{name:18s} | succ={succ_prob:.3f}  err={err_prob:.3f} | pulls={avg_pulls:.1f} | time={avg_time:.3f}s")


JesterToyRankingConfig = ToyRankingConfig
JesterFixedConfidenceConfig = FixedConfidenceConfig
JesterOursOnlyConfig = OursOnlyConfig
SemiparametricSyntheticConfig = SyntheticSuiteConfig


def run_jester_toy_ranking(config: ToyRankingConfig):
    return run_toy_deo_vs_uniform(config)


def run_jester_fixed_confidence(config: FixedConfidenceConfig):
    return run_fixed_confidence_benchmark(config)


def run_jester_ours_only(config: OursOnlyConfig):
    return run_ours_only_benchmark(config)


def run_semiparametric_synthetic(config: SyntheticSuiteConfig, drift_fn, drift_label):
    return run_synthetic_suite(config, drift_fn=drift_fn, drift_label=drift_label)
