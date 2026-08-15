"""
DDM M5 / M4 toolkit — notebook-friendly functions.

M5 = 3D-walker contact + per-bite false-positive (epsilon)  [5 params]
M4 = M5 with epsilon = 0                                    [4 params]

Typical notebook flow:
    from ddm_m5 import *

    data = load_data()
    traces = load_or_build_traces(data["sizes"])

    # Fit models (M5 or M4) using the unified fit_model function
    fit_m5 = fit_model("M5", data, traces)
    fit_m4 = fit_model("M4", data, traces)

    # Predictions across all sizes
    pred_m5 = eval_model("M5", fit_m5["params"], traces, data["sizes"])
    pred_m4 = eval_model("M4", fit_m4["params"], traces, data["sizes"])

    # Metrics
    metrics = summarize_models({
        "M4 3D-walker":      (pred_m4, 4),
        "M5 walker + eps":   (pred_m5, 5),
    }, data)

    # Figures (shows inline in notebook by default; pass save_path to save to disk)
    plot_schematic(fit_m5["params"], include_fp=True)
    plot_walker_calibration(traces, data)
    plot_pcache_comparison({"M4":pred_m4, "M5":pred_m5}, metrics, data)
    plot_clamp_distributions(pred_m5["ns"], data, "M5")
    plot_loss_comparison(metrics)
    plot_trajectories(fit_m5["params"], traces, data["sizes"], include_fp=True)

    write_csv_summaries(
        predictions={"M4":pred_m4, "M5":pred_m5},
        fits={"M4":fit_m4, "M5":fit_m5},
        metrics=metrics, data=data, out_dir=".")
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random as rd

sys.path.insert(0, r"C:\Users\nacholab-adm")
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# ---------------------------------------------------------------------------
# Defaults (override at call time if needed)
# ---------------------------------------------------------------------------
DEFAULT_CSV = Path(__file__).parent.resolve() / "data_paper" / "PreliminaryFruitAnalysis - Parametrics.csv"
DEFAULT_TRACE_DIR = Path(__file__).parent.resolve()
NOISE_RATIO = 1.2
Z0 = 0.0
MAX_BITES = 100
N_TRACES = 500

COL = {
    "empirical": "#222",  "M0_null":  "#888",  "M1_logit": "#9b59b6",
    "M2_drift":  "#e67e22", "M3_bern": "#e76f51", "M4_v6":  "#1f8a3f",
    "M5_v7":     "#0c5fa8",
    "cache":     "#c44",  "eat":     "#3a7",
    "contact":   "#ffa500", "fp":     "#a96bd9",
}


# ===========================================================================
# 1) Data loading
# ===========================================================================
def load_data(csv_path: Path = DEFAULT_CSV,
              sizes: Optional[List[float]] = None) -> dict:
    """Load and clean the empirical CSV. Returns a dict with everything the
    rest of the toolkit needs."""
    if sizes is None:
        sizes = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    df = pd.read_csv(csv_path)

    def parse_type(t):
        if pd.isna(t):
            return np.nan
        t = str(t).strip()
        if t == "control":
            return 0.0
        try:
            return float(t.replace(",", "."))
        except ValueError:
            return np.nan

    df["focus_type"] = df["type"]  # backing up type
    df["hole_size"] = df["type"].apply(parse_type)
    for col in ["scatter-hoard", "eaten", "jaws", "teeth-hole contact"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["focus_type"].notna() & df["hole_size"].isin(sizes)].copy()
    sh = df["scatter-hoard"].fillna(-1).astype(int)
    ea = df["eaten"].fillna(-1).astype(int)
    df["outcome_ok"] = ((sh == 1) & (ea == 0)) | ((sh == 0) & (ea == 1))
    data_ok = df[df["outcome_ok"]].copy()
    data_ok["cache"] = (data_ok["scatter-hoard"] == 1).astype(int)

    emp_pcache, emp_n, emp_pcontact, emp_jaws_by_size = {}, {}, {}, {}
    for s in sizes:
        sub = data_ok[data_ok["hole_size"] == s]
        emp_n[s] = len(sub)
        emp_pcache[s] = sub["cache"].mean() if len(sub) else np.nan
        contact = sub["teeth-hole contact"].dropna()
        emp_pcontact[s] = float((contact >= 1).mean()) if len(contact) else 0.0
        jaws = sub["jaws"].dropna().astype(int).values
        emp_jaws_by_size[s] = jaws[jaws > 0]
    emp_pcontact[0.0] = 0.0

    fit_sizes = [s for s in sizes if emp_n[s] >= 5]
    return dict(
        df=data_ok, sizes=sizes, fit_sizes=fit_sizes,
        emp_pcache=emp_pcache, emp_n=emp_n,
        emp_pcontact=emp_pcontact, emp_jaws_by_size=emp_jaws_by_size,
    )


# ===========================================================================
# 2) 3D walker traces (load cached, or build if missing)
# ===========================================================================
def _trace_path(out_dir: Path, n_traces: int, max_bites: int) -> Path:
    return Path(out_dir) / f"peanut_traces_N{n_traces}_B{max_bites}.npz"


def build_traces(sizes: List[float],
                 out_dir: Path = DEFAULT_TRACE_DIR,
                 n_traces: int = N_TRACES,
                 max_bites: int = MAX_BITES,
                 seed: int = rd.randint(0, 1000)) -> Dict[float, np.ndarray]:
    """Pre-generate N_TRACES paired-bite walker traces per hole size and
    cache to disk. Returns dict {size -> bool array (N_TRACES, max_bites)}."""
    from peanut3d import (_sample_first_z, _sample_step_type,
                          _sample_step_direction, _sample_step_distance,
                          _walk_geodesic, _clamp_hole_overlap)
    rng = np.random.default_rng(rd.randint(0, 1000))
    traces: Dict[float, np.ndarray] = {}
    for s in sizes:
        arr = np.zeros((n_traces, max_bites), dtype=bool)
        hole = None if s <= 0 else {"diameter_cm": float(s),
                                    "chamber": "top", "theta": 0.0}
        for k in range(n_traces):
            np.random.seed(rd.randint(0, 1000))
            prev_type = "L" if np.random.rand() < 0.68 else "T"
            z = _sample_first_z()
            theta = float(np.random.uniform(-np.pi, np.pi))
            for t in range(max_bites):
                ang = float(np.random.normal(0, np.deg2rad(10.0)))
                cA = {"z": z, "theta": theta, "angle": ang}
                theta_b = (theta + 2*np.pi) % (2*np.pi) - np.pi
                cB = {"z": z, "theta": theta_b, "angle": ang}
                contact = False
                if hole is not None:
                    contact = (_clamp_hole_overlap(cA, hole)
                               or _clamp_hole_overlap(cB, hole))
                arr[k, t] = contact
                prev_type = _sample_step_type(prev_type)
                dz, ds = _sample_step_direction(prev_type)
                d = _sample_step_distance(prev_type)
                z, theta = _walk_geodesic(z, theta, dz, ds, d)
        traces[s] = arr
        print(f"  size {s:g}: {n_traces} traces, "
              f"P(>=1 contact) = {arr.any(axis=1).mean():.3f}")
    path = _trace_path(out_dir, n_traces, max_bites)
    np.savez_compressed(path, **{f"s_{s}": a for s, a in traces.items()})
    print(f"Cached to {path}")
    return traces


def load_or_build_traces(sizes: List[float],
                         out_dir: Path = DEFAULT_TRACE_DIR,
                         n_traces: int = N_TRACES,
                         max_bites: int = MAX_BITES) -> Dict[float, np.ndarray]:
    path = _trace_path(out_dir, n_traces, max_bites)
    if path.exists():
        print(f"Loading walker traces from {path.name}")
        d = np.load(path)
        return {float(k[2:]): d[k] for k in d.files}
    print("No cached traces — building (this takes a few minutes) ...")
    return build_traces(sizes, out_dir, n_traces, max_bites)


def build_random_traces(sizes: List[float],
                        out_dir: Path = DEFAULT_TRACE_DIR,
                        n_traces: int = N_TRACES,
                        max_bites: int = MAX_BITES,
                        seed: int = 7,
                        sampling_mode: str = "first_z") -> Dict[float, np.ndarray]:
    """Pre-generate N_TRACES paired-bite random traces per hole size.
    
    Each bite t is placed independently in a random location on the peanut surface.
    
    sampling_mode:
      - 'first_z': use the empirical initial-bite longitudinal density for z (via _sample_first_z) and uniform theta. (default)
      - 'uniform_surface': sample uniformly from the 3D surface area of the peanut.
      - 'uniform_z': sample z uniformly in [Z_MIN, Z_MAX] and uniform theta.
    """
    from peanut3d import _sample_first_z, _clamp_hole_overlap, Z_MIN, Z_MAX, radius_at_z
    
    rng = np.random.default_rng(rd.randint(0, 1000))
    traces: Dict[float, np.ndarray] = {}
    
    # Precompute weights for uniform_surface if needed
    if sampling_mode == "uniform_surface":
        z_grid = np.linspace(Z_MIN, Z_MAX, 2000)
        r_grid = radius_at_z(z_grid)
        dz = z_grid[1] - z_grid[0]
        dr = np.diff(r_grid)
        ds = np.sqrt(dz**2 + dr**2)
        r_mid = (r_grid[:-1] + r_grid[1:]) / 2.0
        weights = r_mid * ds
        weights /= weights.sum()
        z_centers = (z_grid[:-1] + z_grid[1:]) / 2.0
        
    for s in sizes:
        arr = np.zeros((n_traces, max_bites), dtype=bool)
        hole = None if s <= 0 else {"diameter_cm": float(s),
                                    "chamber": "top", "theta": 0.0}
        for k in range(n_traces):
            # Seed local generator for reproducibility per trace
            local_seed = rd.randint(0, 1000)
            local_rng = np.random.default_rng(local_seed)
            
            for t in range(max_bites):
                if sampling_mode == "first_z":
                    # Use _sample_first_z() beta sampling but with local_rng
                    rel = 0.07 + local_rng.beta(3.5, 2.2) * (0.73 - 0.07)
                    sign = 1.0 if local_rng.random() < 0.5 else -1.0
                    z = float(sign * rel * Z_MAX)
                elif sampling_mode == "uniform_surface":
                    z = float(local_rng.choice(z_centers, p=weights))
                elif sampling_mode == "uniform_z":
                    z = float(local_rng.uniform(Z_MIN, Z_MAX))
                else:
                    raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
                    
                theta = float(local_rng.uniform(-np.pi, np.pi))
                ang = float(local_rng.normal(0, np.deg2rad(10.0)))
                
                cA = {"z": z, "theta": theta, "angle": ang}
                theta_b = (theta + 2*np.pi) % (2*np.pi) - np.pi
                cB = {"z": z, "theta": theta_b, "angle": ang}
                
                contact = False
                if hole is not None:
                    contact = (_clamp_hole_overlap(cA, hole)
                               or _clamp_hole_overlap(cB, hole))
                arr[k, t] = contact
        traces[s] = arr
        print(f"  random size {s:g} ({sampling_mode}): {n_traces} traces, "
              f"P(>=1 contact) = {arr.any(axis=1).mean():.3f}")
              
    path = Path(out_dir) / f"peanut_random_traces_{sampling_mode}_N{n_traces}_B{max_bites}.npz"
    np.savez_compressed(path, **{f"s_{s}": a for s, a in traces.items()})
    print(f"Cached random traces to {path}")
    return traces


def load_or_build_random_traces(sizes: List[float],
                                out_dir: Path = DEFAULT_TRACE_DIR,
                                n_traces: int = N_TRACES,
                                max_bites: int = MAX_BITES,
                                sampling_mode: str = "first_z") -> Dict[float, np.ndarray]:
    path = Path(out_dir) / f"peanut_random_traces_{sampling_mode}_N{n_traces}_B{max_bites}.npz"
    if path.exists():
        print(f"Loading random traces ({sampling_mode}) from {path.name}")
        d = np.load(path)
        return {float(k[2:]): d[k] for k in d.files}
    print(f"No cached random traces ({sampling_mode}) — building ...")
    return build_random_traces(sizes, out_dir, n_traces, max_bites, sampling_mode=sampling_mode)


# ===========================================================================
# 3) DDM simulators
# ===========================================================================
def _step_ddm(drift, B_cache, B_eat, J, sigma, contact_seq,
              n_trials, max_steps, rng):
    """Vectorized DDM stepper. Returns (outcome_cache bool, n_steps int)."""
    x = np.full(n_trials, Z0, dtype=float)
    n_steps = np.full(n_trials, max_steps, dtype=int)
    decided = np.zeros(n_trials, dtype=bool)
    outcome_cache = np.zeros(n_trials, dtype=bool)
    alive = np.arange(n_trials)
    T = min(max_steps, contact_seq.shape[1])
    for t in range(T):
        if alive.size == 0:
            break
        a = alive
        noise = rng.standard_normal(a.size)
        jump = contact_seq[a, t].astype(float) * J
        x[a] = x[a] + drift + sigma * noise - jump
        hit_u = x[a] >= B_cache
        hit_l = x[a] <= -B_eat
        just = hit_u | hit_l
        n_steps[a[just]] = t + 1
        outcome_cache[a[hit_u]] = True
        decided[a[just]] = True
        alive = a[~just]
    mid = (B_cache - B_eat) / 2.0
    undec = ~decided
    outcome_cache[undec] = x[undec] > mid
    return outcome_cache, n_steps


def simulate_walker_ddm(drift, B_cache, B_eat, J, eps,
                        traces, size, n_trials,
                        max_steps=MAX_BITES, rng=None,
                        return_components=False):
    """The canonical simulator. eps=0 -> M4. eps>0 -> M5."""
    rng = rng or np.random.default_rng()
    sigma = NOISE_RATIO * drift
    eps = max(0.0, min(eps, 1.0))
    trace_idx = rng.integers(0, traces[size].shape[0], size=n_trials)
    real = traces[size][trace_idx].copy()
    T = real.shape[1]
    if eps > 0:
        fp = rng.random((n_trials, T)) < eps
        contact_seq = real | fp
    else:
        fp = np.zeros_like(real)
        contact_seq = real
    out = _step_ddm(drift, B_cache, B_eat, J, sigma,
                    contact_seq, n_trials, max_steps, rng)
    if return_components:
        return out + (real, fp)
    return out


# ===========================================================================
# 4) Fitting (joint cache + clamp-distribution loss)
# ===========================================================================
def _cache_loss(p_emp, p_sim, n):
    return sum(n[s] * (p_emp[s] - p_sim[s]) ** 2 for s in p_emp)


def _steps_loss(emp_jaws, sim_jaws, n):
    tot = 0.0
    for s, emp in emp_jaws.items():
        if len(emp) == 0 or len(sim_jaws[s]) == 0:
            continue
        tot += n[s] * wasserstein_distance(emp, sim_jaws[s])
    return tot


def _fit_generic(sim_fn, x0, data, n_trials=2000, eps_idx=None,
                 maxiter=1000, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000000)
        
    fit_sizes = data["fit_sizes"]
    p_emp_fit = {s: data["emp_pcache"][s] for s in fit_sizes}
    emp_jaws_fit = {s: data["emp_jaws_by_size"][s] for s in fit_sizes}

    def joint_eval(params):
        sp, sj = {}, {}
        for s in fit_sizes:
            oc, ns = sim_fn(params, s, n_trials,
                            rng=np.random.default_rng(seed + int(round(s * 1000))))
            sp[s] = oc.mean()
            sj[s] = ns
        return sp, sj

    sp0, sj0 = joint_eval(x0)
    L0c = _cache_loss(p_emp_fit, sp0, data["emp_n"])
    L0s = _steps_loss(emp_jaws_fit, sj0, data["emp_n"])
    LAMBDA = L0c / max(L0s, 1e-9)

    def loss(params):
        p = np.asarray(params)
        if eps_idx is not None:
            mask = np.ones_like(p, dtype=bool)
            mask[eps_idx] = False
            if np.any(p[mask] <= 0):
                return 1e6
            if p[eps_idx] < 0 or p[eps_idx] > 0.6:
                return 1e6
        elif np.any(p <= 0):
            return 1e6
        sp, sj = joint_eval(params)
        return (_cache_loss(p_emp_fit, sp, data["emp_n"])
                + LAMBDA * _steps_loss(emp_jaws_fit, sj, data["emp_n"]))

    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": maxiter})
    return dict(params=res.x, loss=res.fun, lam=LAMBDA, success=res.success)


def fit_walker_ddm(data, traces, include_fp=True, n_trials=2000, seed=None):
    """Fit M4 (include_fp=False, k=4) or M5 (include_fp=True, k=5)."""
    if include_fp:
        x0 = np.array([0.10, 0.30, 0.50, 0.60, 0.03])
        names = ["drift", "B_cache", "B_eat", "J", "eps"]

        def sim_fn(params, size, n, rng):
            drift, Bc, Be, J, eps = params
            return simulate_walker_ddm(drift, Bc, Be, J, eps,
                                       traces, size, n, rng=rng)
        res = _fit_generic(sim_fn, x0, data, n_trials=n_trials, eps_idx=4, seed=seed)
        k = 5
    else:
        x0 = np.array([0.10, 0.30, 0.50, 0.60])
        names = ["drift", "B_cache", "B_eat", "J"]

        def sim_fn(params, size, n, rng):
            drift, Bc, Be, J = params
            return simulate_walker_ddm(drift, Bc, Be, J, 0.0,
                                       traces, size, n, rng=rng)
        res = _fit_generic(sim_fn, x0, data, n_trials=n_trials, seed=seed)
        k = 4
    res["k"] = k
    res["param_names"] = names
    res["include_fp"] = include_fp
    res["params_dict"] = dict(zip(names, res["params"]))
    label = "M5 (walker + eps)" if include_fp else "M4 (3D-walker)"
    print(f"{label}: " + "  ".join(
        f"{n}={v:+.4f}" for n, v in res["params_dict"].items())
        + f"  loss={res['loss']:.3f}")
    return res


# ===========================================================================
# 5) Unified fit_model and eval_model interfaces
# ===========================================================================
def fit_model(model_name: str, data: dict, traces: dict, n_trials: int = 5000, seed: int = None) -> dict:
    """Fit either 'M4' or 'M5' to the provided data and traces.
    
    Inputs:
      model_name: 'M4' (3D-walker without false-positive contact) or 
                  'M5' (3D-walker with false-positive contact epsilon)
      data: dictionary loaded from load_data()
      traces: dictionary of walker traces loaded/built from load_or_build_traces()
      n_trials: number of simulations per size (default: 2000)
      seed: random seed for the fitting run (default: None, for a random unseeded run)
      
    Returns:
      A dictionary containing the parameters, loss, AIC/SSE metrics, and other fit details.
    """
    model_name = model_name.upper()
    if model_name == "M5":
        return fit_walker_ddm(data, traces, include_fp=True, n_trials=n_trials, seed=seed)
    elif model_name == "M4":
        return fit_walker_ddm(data, traces, include_fp=False, n_trials=n_trials, seed=seed)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Only 'M4' and 'M5' are supported in this version.")


def eval_model(model_name: str, params: np.ndarray, traces: dict, sizes: list, n_trials: int = 20000) -> dict:
    """Evaluate either 'M4' or 'M5' given the parameters, traces, and sizes.
    
    Inputs:
      model_name: 'M4' or 'M5'
      params: fitting parameters (array)
      traces: dictionary of walker traces
      sizes: list of sizes to evaluate
      n_trials: number of evaluation simulations (default: 20000)
      
    Returns:
      A dictionary containing predictions: {"pc": {size: P(cache)}, "ns": {size: array of clamp counts}}
    """
    model_name = model_name.upper()
    if model_name == "M5":
        return eval_walker_ddm(params, traces, sizes, include_fp=True, n_trials=n_trials)
    elif model_name == "M4":
        return eval_walker_ddm(params, traces, sizes, include_fp=False, n_trials=n_trials)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Only 'M4' and 'M5' are supported in this version.")


# ===========================================================================
# 6) Evaluation across all sizes
# ===========================================================================
def _eval_grid(sim_call, sizes, n_trials):
    pc, ns = {}, {}
    for s in sizes:
        rng = np.random.default_rng(2024 + int(round(s * 100)))
        oc, n_s = sim_call(s, n_trials, rng)
        pc[s] = float(oc.mean())
        ns[s] = n_s
    return dict(pc=pc, ns=ns)


def eval_walker_ddm(params, traces, sizes, include_fp=True, n_trials=20000):
    if include_fp:
        drift, Bc, Be, J, eps = params
    else:
        drift, Bc, Be, J = params
        eps = 0.0
    return _eval_grid(lambda s, n, rng:
                      simulate_walker_ddm(drift, Bc, Be, J, eps,
                                          traces, s, n, rng=rng),
                      sizes, n_trials)


# ===========================================================================
# 7) Metrics
# ===========================================================================
def cache_sse(pred_dict, data):
    return sum(data["emp_n"][s] * (data["emp_pcache"][s] - pred_dict["pc"][s]) ** 2
               for s in data["fit_sizes"])


def bin_loglik(pred_dict, data):
    ll = 0.0
    for s in data["fit_sizes"]:
        k = data["emp_n"][s] * data["emp_pcache"][s]
        n = data["emp_n"][s]
        p = min(max(pred_dict["pc"][s], 1e-6), 1 - 1e-6)
        ll += k * np.log(p) + (n - k) * np.log(1 - p)
    return ll


def aic(ll, k):
    return -2 * ll + 2 * k


def summarize_models(model_specs: Dict[str, Tuple[dict, int]], data) -> dict:
    """model_specs: {label -> (predictions dict, k)}."""
    out = {}
    print(f"\n{'model':<22} {'k':>3} {'cache-SSE':>10} {'logL':>10} {'AIC':>10}")
    for name, (pred, k) in model_specs.items():
        sse = cache_sse(pred, data)
        ll = bin_loglik(pred, data)
        out[name] = dict(k=k, sse=sse, ll=ll, aic=aic(ll, k))
        print(f"  {name:<20} {k:>3} {sse:>10.3f} {ll:>10.2f} {aic(ll, k):>10.2f}")
    return out


# ===========================================================================
# 8) Figures
# ===========================================================================
def plot_schematic(params, save_path=None, include_fp=True, color_key="M5_v7"):
    """Fig 1: cartoon of DDM dynamics with the fitted M5 (or M4) params."""
    if include_fp:
        drift, Bc, Be, J, eps = params
    else:
        drift, Bc, Be, J = params
        eps = 0.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.7, 0.5)
    ax.axhline(Bc,  color=COL["cache"], lw=2, label="cache bound $+B_c$")
    ax.axhline(-Be, color=COL["eat"],   lw=2, label="eat bound $-B_e$")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.text(7.4, Bc + 0.02, "CACHE", color=COL["cache"], fontsize=11,
            weight="bold", ha="right")
    ax.text(7.4, -Be - 0.05, "EAT", color=COL["eat"], fontsize=11,
            weight="bold", ha="right")
    ax.text(0.1, 0.02, "start z=0", color="gray", fontsize=9)

    np.random.seed(rd.randint(0, 1000))
    sigma = NOISE_RATIO * drift
    x_no = np.cumsum(np.concatenate([[0], drift + sigma * np.random.randn(5)]))
    ax.plot(np.arange(len(x_no)), x_no, "o-", color=COL["cache"], lw=2, ms=6,
            label="trial without perceived contact")
    ax.annotate("drift $\\mu$ per bite", xy=(2.0, 0.10), xytext=(3.0, 0.18),
                fontsize=10, color="black", ha="center",
                arrowprops=dict(arrowstyle="->", color="black"))

    np.random.seed(rd.randint(0, 1000))
    x_c = [0.0]; contact_bite = 3
    for i in range(7):
        n = sigma * np.random.randn()
        x_new = x_c[-1] + drift + n - (J if i == contact_bite else 0)
        x_c.append(x_new)
        if x_c[-1] <= -Be:
            break
    ax.plot(range(len(x_c)), x_c, "o-", color=COL["eat"], lw=2, ms=6,
            label="trial with perceived contact at bite 4")
    ax.plot(contact_bite + 1, x_c[contact_bite + 1], "o",
            ms=14, color=COL["contact"], mec="black", mew=1.5, zorder=5)
    ax.annotate("perceived contact -> jump $-J$",
                xy=(contact_bite + 1, x_c[contact_bite + 1]),
                xytext=(contact_bite + 0.5, x_c[contact_bite + 1] + 0.15),
                fontsize=10, color="black", ha="center",
                arrowprops=dict(arrowstyle="->", color=COL["contact"], lw=1.5))

    param_str = (f"$\\mu$={drift:.3f}, $\\sigma$={NOISE_RATIO}$\\mu$, "
                 f"$B_c$={Bc:.3f}, $B_e$={Be:.3f}, J={J:.3f}")
    if include_fp:
        param_str += f", $\\epsilon$={eps:.3f}"
    ax.text(0.05, -0.66, param_str, fontsize=9, color="#333")
    ax.set_xlabel("bite # (= step)"); ax.set_ylabel("decision variable $x$")
    title = ("M5 — drift-to-cache + perceived contact (real or false-positive) "
             "jumps to eat") if include_fp else (
        "M4 — drift-to-cache + 3D-walker contact jumps to eat")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


def plot_walker_calibration(traces, data, random_traces=None, save_path=None):
    """Fig 2: 3D walker cumulative contact rate vs empirical p_contact."""
    if random_traces is None:
        path_r = DEFAULT_TRACE_DIR / f"peanut_random_traces_first_z_N{N_TRACES}_B{MAX_BITES}.npz"
        if path_r.exists():
            d = np.load(path_r)
            random_traces = {float(k[2:]): d[k] for k in d.files}

    sizes_no_ctrl = [s for s in data["sizes"] if s > 0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, s in enumerate(sizes_no_ctrl):
        ax = axes[i]
        arr = traces[s]
        ks = np.arange(1, 31)
        cum = np.array([arr[:, :k].any(axis=1).mean() for k in ks])
        ax.plot(ks, cum, lw=2.4, color=COL["M4_v6"],
                label="3D walker (good)")
                
        if random_traces is not None and s in random_traces:
            arr_r = random_traces[s]
            cum_r = np.array([arr_r[:, :k].any(axis=1).mean() for k in ks])
            ax.plot(ks, cum_r, lw=2.0, color="#888888", ls="--",
                    label="Random placement")
                    
        ax.axhline(data["emp_pcontact"][s], color=COL["empirical"],
                   ls="--", lw=1.5,
                   label=f"empirical = {data['emp_pcontact'][s]:.2f}")
        ax.axvline(3, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlim(0, 30); ax.set_ylim(0, 1.02)
        ax.set_title(f"size {s:g} cm")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, loc="lower right")
    for k in [3, 4, 5]:
        axes[k].set_xlabel("bite #")
    axes[0].set_ylabel("P(at least one contact)")
    axes[3].set_ylabel("P(at least one contact)")
    fig.suptitle("Figure 2 — 3D walker vs Random placement cumulative contact rate",
                 fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


def plot_pcache_comparison(predictions, metrics, data, save_path=None):
    """Fig 3: empirical P(cache) and all model bars per size."""
    # Check if there is a random prediction in the caller's globals
    import sys
    globs = sys._getframe(1).f_globals
    
    # If M5 is in predictions, try to add M5_random to predictions and metrics
    if "M5" in predictions and "M5_random" not in predictions:
        pred_r = None
        if "pred_random" in globs:
            pred_random = globs["pred_random"]
            if isinstance(pred_random, dict) and "first_z" in pred_random:
                pred_r = pred_random["first_z"]
            elif isinstance(pred_random, dict) and "pc" in pred_random:
                pred_r = pred_random
        elif "fit_m5" in globs:
            fit_m5 = globs["fit_m5"]
            try:
                r_traces = load_or_build_random_traces(data["sizes"], sampling_mode="first_z")
                pred_r = eval_model("M5", fit_m5["params"], r_traces, data["sizes"], n_trials=20000)
            except Exception as e:
                print(f"Could not automatically compute random prediction: {e}")
                
        if pred_r is not None:
            predictions = predictions.copy()
            predictions["M5_random"] = pred_r
            sse_r = cache_sse(pred_r, data)
            ll_r = bin_loglik(pred_r, data)
            metrics = metrics.copy()
            metrics["M5_random"] = {"k": 5, "sse": sse_r, "ll": ll_r, "aic": aic(ll_r, 5)}

    sizes = data["sizes"]
    labels = ["ctrl"] + [f"{s:g}" for s in sizes[1:]]
    xs = np.arange(len(sizes))
    color_map = {"M0": COL["M0_null"], "M1": COL["M1_logit"],
                 "M2": COL["M2_drift"], "M3": COL["M3_bern"],
                 "M4": COL["M4_v6"], "M5": COL["M5_v7"],
                 "M5_random": "#888888"}
    pretty = {"M0": "M0 null", "M1": "M1 logistic", "M2": "M2 drift-only",
              "M3": "M3 Bernoulli", "M4": "M4 3D-walker",
              "M5": "M5 walker + $\\epsilon$",
              "M5_random": "M5 Random Clamps"}
    keys = [k for k in ["M0","M1","M2","M3","M4","M5","M5_random"] if k in predictions]
    n_bars = 1 + len(keys)
    w = 0.85 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * w
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(xs + offsets[0], [data["emp_pcache"][s] for s in sizes], w,
           label="Empirical", color=COL["empirical"])
    for j, m in enumerate(keys, start=1):
        full_name = pretty[m]
        metric_key = [k for k in metrics if k.startswith(m)]
        sse = metrics[metric_key[0]]["sse"] if metric_key else np.nan
        ax.bar(xs + offsets[j],
               [predictions[m]["pc"][s] for s in sizes], w,
               color=color_map[m], label=f"{full_name} (SSE={sse:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("Hole size (cm)"); ax.set_ylabel("P(cache)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 3 — P(cache) by hole size, models",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    for i, s in enumerate(sizes):
        ax.text(i, 0.02, f"n={data['emp_n'][s]}", ha="center",
                color="white", fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


def plot_clamp_distributions(ns_dict, data, model_label, save_path=None,
                              color=None, include_random=False):
    """Fig 4: per-size clamp histograms, empirical vs model."""
    # Ensure SVG text is saved as text paths (editable in Illustrator)
    plt.rcParams['svg.fonttype'] = 'none'
    
    if color is None:
        color = COL["M5_v7"]
    sizes = data["sizes"]
    
    pred_r = None
    if include_random:
        # Auto-discover random predictions for comparison
        import sys
        globs = sys._getframe(1).f_globals
        if "pred_random" in globs:
            pred_random = globs["pred_random"]
            if isinstance(pred_random, dict) and "first_z" in pred_random:
                pred_r = pred_random["first_z"]
            elif isinstance(pred_random, dict) and "pc" in pred_random:
                pred_r = pred_random
        elif "fit_m5" in globs:
            fit_m5 = globs["fit_m5"]
            try:
                r_traces = load_or_build_random_traces(data["sizes"], sampling_mode="first_z")
                pred_r = eval_model("M5", fit_m5["params"], r_traces, data["sizes"], n_trials=20000)
            except Exception as e:
                pass
            
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    xmax = 20
    bins = np.arange(0, xmax + 2) - 0.5
    
    FONT_SIZE = 14
    AXIS_LW = 2.0
    
    for i, s in enumerate(sizes):
        ax = axes[i]
        
        # Remove frame (top/right spines), keep only bottom/left axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(AXIS_LW)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(AXIS_LW)
        
        # Configure tick parameters
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE, width=AXIS_LW, length=6)
        ax.grid(False)
        
        emp = data["emp_jaws_by_size"][s]
        sim = ns_dict[s]
        if len(emp):
            ax.hist(emp, bins=bins, density=True, alpha=0.4,
                    color=COL["empirical"], label="Empirical")
                    
        # Walker model
        ax.hist(sim, bins=bins, density=True, histtype='step', lw=2.0,
                color=color, label=model_label)
                
        # Random model
        if include_random and pred_r is not None and "ns" in pred_r and s in pred_r["ns"]:
            sim_r = pred_r["ns"][s]
            ax.hist(sim_r, bins=bins, density=True, histtype='step', lw=1.8,
                    ls="--", color="#e67e22", label="M5 Random")
                    
        med_emp = float(np.median(emp)) if len(emp) else np.nan
        med_sim = float(np.median(sim))
        ax.axvline(med_emp, color=COL["empirical"], ls=":", lw=1)
        ax.axvline(med_sim, color=color, ls=":", lw=1)
        
        if include_random and pred_r is not None and "ns" in pred_r and s in pred_r["ns"]:
            med_sim_r = float(np.median(pred_r["ns"][s]))
            ax.axvline(med_sim_r, color="#e67e22", ls=":", lw=1)
            ax.set_title(f"size {s:g}  n_emp={len(emp)}\n"
                         f"med emp={med_emp:.0f}, {model_label}={med_sim:.0f}, Random={med_sim_r:.0f}",
                         fontsize=FONT_SIZE)
        else:
            ax.set_title(f"size {s:g}  n_emp={len(emp)}\n"
                         f"med emp={med_emp:.0f}, {model_label}={med_sim:.0f}",
                         fontsize=FONT_SIZE)
                         
        ax.set_xlabel("clamps per trial", fontsize=FONT_SIZE)
        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)
        if i == 0:
            ax.legend(fontsize=FONT_SIZE)
            
    axes[7].axis("off")
    # Ensure off axes are fully clean
    for spine in axes[7].spines.values():
        spine.set_visible(False)
        
    fig.suptitle(f"Figure 4 — Clamp distributions, empirical vs {model_label}",
                 fontsize=FONT_SIZE + 2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved {save_path}")
    


def plot_loss_comparison(metrics, save_path=None):
    """Fig 5: SSE + AIC bars across all fitted models."""
    import sys
    globs = sys._getframe(1).f_globals
    pred_r = None
    if "pred_random" in globs:
        pred_random = globs["pred_random"]
        if isinstance(pred_random, dict) and "first_z" in pred_random:
            pred_r = pred_random["first_z"]
        elif isinstance(pred_random, dict) and "pc" in pred_random:
            pred_r = pred_random
    elif "fit_m5" in globs:
        fit_m5 = globs["fit_m5"]
        try:
            r_traces = load_or_build_random_traces(globs["data"]["sizes"], sampling_mode="first_z")
            pred_r = eval_model("M5", fit_m5["params"], r_traces, globs["data"]["sizes"], n_trials=20000)
        except Exception as e:
            pass
            
    if pred_r is not None and "M5_random" not in metrics:
        data = globs.get("data")
        if data is not None:
            sse_r = cache_sse(pred_r, data)
            ll_r = bin_loglik(pred_r, data)
            metrics = metrics.copy()
            metrics["M5 Random"] = {"k": 5, "sse": sse_r, "ll": ll_r, "aic": aic(ll_r, 5)}

    color_map = {"M0": COL["M0_null"], "M1": COL["M1_logit"],
                 "M2": COL["M2_drift"], "M3": COL["M3_bern"],
                 "M4": COL["M4_v6"], "M5": COL["M5_v7"]}
    names = list(metrics.keys())
    
    cols = []
    for name in names:
        if "random" in name.lower():
            cols.append("#888888")
        else:
            short = name.split()[0]
            cols.append(color_map.get(short, "#666"))
            
    sse_v = [metrics[n]["sse"] for n in names]
    aic_v = [metrics[n]["aic"] for n in names]
    labels = [n.replace(" ", "\n", 1) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(labels, sse_v, color=cols)
    axes[0].set_ylabel("Cache SSE (sample-weighted)")
    axes[0].set_title("Cache fit")
    for b, v in zip(bars, sse_v):
        axes[0].text(b.get_x() + b.get_width() / 2, v + max(sse_v) * 0.01,
                     f"{v:.2f}", ha="center", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    bars = axes[1].bar(labels, aic_v, color=cols)
    axes[1].set_ylabel("AIC (lower is better)")
    axes[1].set_title("Binomial AIC on P(cache)")
    for b, v in zip(bars, aic_v):
        axes[1].text(b.get_x() + b.get_width() / 2, v + max(aic_v) * 0.005,
                     f"{v:.1f}", ha="center", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Figure 5 — Model comparison (cache-SSE and AIC)",
                 fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


def plot_trajectories(params, traces, sizes, save_path=None,
                      include_fp=True, n_per_size=20, max_steps=MAX_BITES):
    """
    Generate separate example DDM trajectories for each size.
    If random traces are available, also generates and saves comparison random trajectories
    with a '_random' filename prefix suffix.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Ensure SVG text is saved as text paths (editable in Illustrator)
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Check for random traces
    import sys
    globs = sys._getframe(1).f_globals
    r_traces = None
    if "random_traces" in globs:
        rt = globs["random_traces"]
        if isinstance(rt, dict):
            if "first_z" in rt:
                r_traces = rt["first_z"]
            elif any(isinstance(k, float) for k in rt.keys()):
                r_traces = rt

    if r_traces is None:
        path_r = DEFAULT_TRACE_DIR / f"peanut_random_traces_first_z_N{N_TRACES}_B{MAX_BITES}.npz"
        if path_r.exists():
            d = np.load(path_r)
            r_traces = {float(k[2:]): d[k] for k in d.files}

    if include_fp:
        drift, Bc, Be, J, eps = params
    else:
        drift, Bc, Be, J = params; eps = 0.0
    sigma = NOISE_RATIO * drift

    def one_trial(size, tr_dict):
        rng = np.random.default_rng()
        idx = rng.integers(0, tr_dict[size].shape[0])
        real = tr_dict[size][idx]
        T = real.shape[0]
        fp = rng.random(T) < eps if eps > 0 else np.zeros(T, dtype=bool)
        x = Z0; traj = [x]; rc, fc = [], []
        for t in range(min(max_steps, T)):
            r, f = bool(real[t]), bool(fp[t])
            if r: rc.append(t)
            if f and not r: fc.append(t)
            x += drift + sigma * rng.standard_normal() - (J if (r or f) else 0)
            traj.append(x)
            if x >= Bc or x <= -Be:
                break
        outcome = "cache" if traj[-1] >= Bc else "eat"
        return traj, rc, fc, outcome

    FONT_SIZE = 14
    AXIS_LW = 2.0
    
    def apply_aesthetic(ax):
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Keep left and bottom spines, make them thicker
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(AXIS_LW)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(AXIS_LW)
        # Configure tick parameters
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE, width=AXIS_LW, length=6)
        # Font sizes for labels
        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)
        # Remove grid
        ax.grid(False)

    def run_and_save_dataset(traces_data, filename_prefix="trajectory"):
        for s in sizes:
            print(s)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            for k in range(n_per_size):
                traj, rc, fc, outcome = one_trial(s, traces_data)
                c = COL["cache"] if outcome == "cache" else COL["eat"]
                ax.plot(traj, color=c, alpha=0.4, lw=0.9)
                for cs in rc:
                    if cs + 1 < len(traj):
                        ax.plot(cs + 1, traj[cs + 1], "o", ms=3.5,
                                color=COL["contact"], alpha=0.85, zorder=4)
                for cs in fc:
                    if cs + 1 < len(traj):
                        ax.plot(cs + 1, traj[cs + 1], "o", ms=3.5,
                                markerfacecolor="none",
                                markeredgecolor=COL["fp"], mew=1.0,
                                alpha=0.85, zorder=4)
                                
            ax.axhline(Bc,  color="k", lw=1.5)
            ax.axhline(-Be, color="k", lw=1.5)
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            
            ax.set_ylabel("decision variable x")
            ax.set_xlabel("bite #")
            apply_aesthetic(ax)
            fig.tight_layout()
            
            if save_path is not None:
                save_dir = Path(save_path)
                if save_dir.suffix:
                    base = save_dir.parent / save_dir.stem
                    ext = save_dir.suffix
                    path = Path(f"{base}_{filename_prefix}_size_{s:g}{ext}")
                else:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    path = save_dir / f"{filename_prefix}_size_{s:g}.svg"
                    
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.show()
                plt.close(fig)
                print(f"Saved {path}")

    # Run for standard walker
    run_and_save_dataset(traces, filename_prefix="trajectory")
    
    # Run for random traces if available
    if r_traces is not None:
        print("Generating and saving trajectories for Random Clamps...")
        run_and_save_dataset(r_traces, filename_prefix="trajectory_random")


def plot_psychometric_chronometric(
    data: dict,
    model_pred: dict,
    model_label: str = "DDM v7",
    control_pred: Optional[dict] = None,
    control_label: str = "Random Placement Control",
    save_path: Optional[str] = None,
    model_color: Optional[str] = None
):
    """
    Generate side-by-side Psychometric and Chronometric curves comparing
    Empirical data against Model predictions (e.g. M5) and an optional Control.
    
    Args:
        data: Dict returned by load_data() containing empirical statistics.
        model_pred: Dict returned by eval_model() for the primary model (e.g. M5).
        model_label: Label for the primary model (e.g. "DDM v7").
        control_pred: Optional dict returned by eval_model() for a control model (e.g. M4).
        control_label: Label for the control model (e.g. "Random Placement Control").
        save_path: Path to save the generated figure. If None, plots inline.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Ensure SVG text is saved as text paths (editable in Illustrator)
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Robustness check: if control_pred is nested (e.g. {mode: pred_dict}), unpack preferred mode
    if control_pred is not None and isinstance(control_pred, dict):
        if "pc" not in control_pred and any(isinstance(v, dict) and "pc" in v for v in control_pred.values()):
            for preferred_key in ["first_z", "uniform_surface", "uniform_z"]:
                if preferred_key in control_pred:
                    control_pred = control_pred[preferred_key]
                    break
            else:
                control_pred = next(iter(control_pred.values()))

    sizes = data["sizes"]
    emp_p = [data["emp_pcache"][s] for s in sizes]
    emp_n = [data["emp_n"][s] for s in sizes]
    
    # Calculate SEM for empirical P(cache)
    emp_sem = []
    for p, n in zip(emp_p, emp_n):
        if n > 0 and not np.isnan(p):
            emp_sem.append(np.sqrt(p * (1.0 - p) / n))
        else:
            emp_sem.append(0.0)
            
    # Empirical Medians & Quartiles (jaws/clamps count)
    emp_med_jaws = []
    emp_q25 = []
    emp_q75 = []
    for s in sizes:
        jaws = data["emp_jaws_by_size"].get(s, np.array([]))
        if len(jaws) > 0:
            emp_med_jaws.append(np.median(jaws))
            emp_q25.append(np.percentile(jaws, 25))
            emp_q75.append(np.percentile(jaws, 75))
        else:
            emp_med_jaws.append(np.nan)
            emp_q25.append(np.nan)
            emp_q75.append(np.nan)
            
    # Model Predictions (Median & Quartiles)
    model_p = [model_pred["pc"][s] for s in sizes]
    model_med_jaws = []
    model_q25 = []
    model_q75 = []
    for s in sizes:
        ns = model_pred["ns"].get(s, None)
        if ns is not None and len(ns) > 0:
            model_med_jaws.append(np.median(ns))
            model_q25.append(np.percentile(ns, 25))
            model_q75.append(np.percentile(ns, 75))
        else:
            model_med_jaws.append(np.nan)
            model_q25.append(np.nan)
            model_q75.append(np.nan)
            
    # Control Predictions (optional)
    if control_pred is not None:
        control_p = [control_pred["pc"][s] for s in sizes]
        control_med_jaws = []
        control_q25 = []
        control_q75 = []
        for s in sizes:
            ns = control_pred["ns"].get(s, None)
            if ns is not None and len(ns) > 0:
                control_med_jaws.append(np.median(ns))
                control_q25.append(np.percentile(ns, 25))
                control_q75.append(np.percentile(ns, 75))
            else:
                control_med_jaws.append(np.nan)
                control_q25.append(np.nan)
                control_q75.append(np.nan)
                
    # Colors matching the requested styles
    c_emp = "#1f3a52"
    c_ctrl = "#888888"
    
    # Determine model color
    if model_color is not None:
        c_model = model_color
    elif control_pred is None:
        c_model = COL["M4_v6"]  # Default to green for a single model plot (paper style)
    else:
        # If comparing, use blue for M5/v7 and green for M4/v6
        if "m4" in model_label.lower() or "v6" in model_label.lower():
            c_model = COL["M4_v6"]
        else:
            c_model = COL["M5_v7"]
            
    FONT_SIZE = 14
    AXIS_LW = 2.0
    
    def apply_aesthetic(ax):
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Keep left and bottom spines, make them thicker
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(AXIS_LW)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(AXIS_LW)
        # Configure tick parameters
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE, width=AXIS_LW, length=6)
        # Font sizes for labels
        ax.xaxis.label.set_size(FONT_SIZE)
        ax.yaxis.label.set_size(FONT_SIZE)
        # Remove grid
        ax.grid(False)
        
    # Figure 1: Psychometric Curve (Probability of Caching)
    fig1, ax1 = plt.subplots(figsize=(6.5, 5.5))
    if control_pred is not None:
        ax1.plot(sizes, control_p, "--", color=c_ctrl, lw=2)
    ax1.plot(sizes, model_p, "s-", color=c_model, lw=2.5, ms=8)
    ax1.errorbar(sizes, emp_p, yerr=emp_sem, fmt="o", color=c_emp, ecolor=c_emp, elinewidth=1.5, capsize=4, ms=9)
    
    ax1.set_xlabel("Hole Size (cm)")
    ax1.set_ylabel("Probability of Caching (P(cache))")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f"{s:g}" for s in sizes])
    ax1.set_ylim(0.0, 1.05)
    apply_aesthetic(ax1)
    fig1.tight_layout()
    
    # Figure 2: Chronometric Curve (Decision Time in Clamps)
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    if control_pred is not None:
        ax2.fill_between(sizes, control_q25, control_q75, color=c_ctrl, alpha=0.08)
        ax2.plot(sizes, control_med_jaws, "--", color=c_ctrl, lw=2)
    ax2.fill_between(sizes, model_q25, model_q75, color=c_model, alpha=0.15)
    ax2.plot(sizes, model_med_jaws, "s-", color=c_model, lw=2.5, ms=8)
    
    emp_yerr_low = []
    emp_yerr_high = []
    for med, q25, q75 in zip(emp_med_jaws, emp_q25, emp_q75):
        if np.isnan(med) or np.isnan(q25) or np.isnan(q75):
            emp_yerr_low.append(0.0)
            emp_yerr_high.append(0.0)
        else:
            emp_yerr_low.append(med - q25)
            emp_yerr_high.append(q75 - med)
    emp_yerr = [emp_yerr_low, emp_yerr_high]
    
    ax2.errorbar(sizes, emp_med_jaws, yerr=emp_yerr, fmt="o-", color=c_emp, ecolor=c_emp, elinewidth=1.5, capsize=4, lw=2, ms=9)
    
    ax2.set_xlabel("Hole Size (cm)")
    ax2.set_ylabel("Median Clamps (Bite Count)")
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f"{s:g}" for s in sizes])
    ax2.set_ylim(0, 11)
    apply_aesthetic(ax2)
    fig2.tight_layout()
    
    # Handle saving/showing
    if save_path is not None:
        save_dir = Path(save_path)
        # If it has a suffix (like .svg), we can deduce the extension, but save separate files
        if save_dir.suffix:
            base = save_dir.parent / save_dir.stem
            ext = save_dir.suffix
            path1 = Path(f"{base}_psychometric{ext}")
            path2 = Path(f"{base}_chronometric{ext}")
        else:
            # It's a directory path (e.g. figures)
            save_dir.mkdir(parents=True, exist_ok=True)
            path1 = save_dir / "psychometric.svg"
            path2 = save_dir / "chronometric.svg"

        plt.show()
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
        print(f"Saved {path1}")
        print(f"Saved {path2}")
    
   


# ===========================================================================
# 9) CSV summaries
# ===========================================================================
