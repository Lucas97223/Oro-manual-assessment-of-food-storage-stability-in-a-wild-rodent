# pooled_model_pipeline_tau_global.py
# Full corrected script: per-animal fits + global (pooled) parameter estimation + plots
# Change-point model now uses a single global tau shared across animals.
# Requires: numpy, matplotlib, pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os
from scipy.interpolate import griddata
from scipy.special import expit
from scipy.stats import poisson
from scipy.optimize import minimize

rng = np.random.default_rng(2)
# --- SAVE PATH CONFIGURATION ---
# Change this to your desired output path
save_path_first_plot = r"figures/SH_mode_models.svg"
# Ensure directory exists
os.makedirs(os.path.dirname(save_path_first_plot), exist_ok=True)

# ------------------------------------------------------------------
# Put your structured_data in the environment before running this.
# structured_data must be a list of dicts, each with keys:
#   - "y": 1D numpy array of 0/1 observations
#   - "t": 1D numpy array (trial indices) or similar
# ------------------------------------------------------------------
# data = structured_data  # Should be provided in environment
# max_T = max(len(d["y"]) for d in data)

# -----------------------------
# 2) Model-fitting utilities
# -----------------------------
def logistic_fit(X, y, max_iter=1000, tol=1e-6, track_ll=True):
    """IRLS logistic regression (returns beta, log-likelihood, trace)."""
    n, p = X.shape
    beta = np.zeros(p)
    ll_trace = []

    for _ in range(max_iter):
        eta = X @ beta
        p_hat = 1 / (1 + np.exp(-eta))
        W = np.clip(p_hat * (1 - p_hat), 1e-12, None)
        z = eta + (y - p_hat) / W
        XTWX = X.T @ (X * W[:, None])
        XTWz = X.T @ (W * z)
        try:
            step = np.linalg.solve(XTWX, XTWz - XTWX @ beta)
        except np.linalg.LinAlgError:
            XTWX += 1e-6 * np.eye(p)
            step = np.linalg.solve(XTWX, XTWz - XTWX @ beta)
        beta += step

        eta = X @ beta
        p_hat = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        ll = float(np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)))
        if track_ll:
            ll_trace.append(ll)
        if np.max(np.abs(step)) < tol:
            break

    if track_ll:
        return beta, ll, ll_trace
    return beta, ll

def changepoint_fit(y, min_seg=3):
    """Per-animal MLE for single change-point tau with Bernoulli p0 pre, p1 post.
       (Kept for per-animal diagnostics; global fit will search over shared taus.)
    """
    T = len(y)
    best = None
    csum = np.cumsum(y)
    for tau in range(min_seg, T-min_seg+1):
        n0 = tau; s0 = int(csum[tau-1])
        n1 = T - tau; s1 = int(csum[-1] - s0)
        eps = 1e-9
        p0 = np.clip(s0/max(n0,1), eps, 1-eps)
        p1 = np.clip(s1/max(n1,1), eps, 1-eps)
        ll = s0*np.log(p0)+(n0-s0)*np.log(1-p0)+s1*np.log(p1)+(n1-s1)*np.log(1-p1)
        if (best is None) or (ll > best[3]):
            best = (tau, float(p0), float(p1), float(ll))
    # if no valid tau found (short sequence), fall back to no-change
    if best is None:
        eps = 1e-9
        p = np.clip(y.mean(), eps, 1-eps)
        return (len(y), p, p, float(np.sum(y*np.log(p) + (1-y)*np.log(1-p))))
    return best

# HMM forward-backward (returns log-lik, gamma, xi)
def hmm_forward_backward(y, q, r, pi, eps):
    """Given HMM params, run forward-backward and return (ll, gamma, xi)."""
    y = y.astype(int)
    T = len(y)
    py_eat = np.where(y==1, q, 1-q)
    py_sh  = np.where(y==1, r, 1-r)
    T00 = (1 - pi); T01 = pi
    T10 = eps;      T11 = (1 - eps)

    alpha = np.zeros((T,2))
    # Start in Eat with prob 1 (log)
    alpha[0,0] = np.log(py_eat[0] + 1e-12)
    alpha[0,1] = -1e12 + np.log(py_sh[0] + 1e-12)  # essentially zero prob
    for t in range(1, T):
        a0 = alpha[t-1,0] + np.log(T00 + 1e-12)
        a1 = alpha[t-1,1] + np.log(T10 + 1e-12)
        alpha[t,0] = np.log(py_eat[t] + 1e-12) + np.logaddexp(a0, a1)

        b0 = alpha[t-1,0] + np.log(T01 + 1e-12)
        b1 = alpha[t-1,1] + np.log(T11 + 1e-12)
        alpha[t,1] = np.log(py_sh[t] + 1e-12) + np.logaddexp(b0, b1)

    ll = float(np.logaddexp(alpha[-1,0], alpha[-1,1]))

    beta = np.zeros((T,2))
    beta[-1,:] = 0.0
    for t in range(T-2, -1, -1):
        term0 = np.log(T00+1e-12)+np.log(py_eat[t+1]+1e-12)+beta[t+1,0]
        term1 = np.log(T01+1e-12)+np.log(py_sh[t+1]+1e-12) +beta[t+1,1]
        beta[t,0] = np.logaddexp(term0, term1)

        term0 = np.log(T10+1e-12)+np.log(py_eat[t+1]+1e-12)+beta[t+1,0]
        term1 = np.log(T11+1e-12)+np.log(py_sh[t+1]+1e-12) +beta[t+1,1]
        beta[t,1] = np.logaddexp(term0, term1)

    gamma_log = alpha + beta
    m = np.max(gamma_log, axis=1, keepdims=True)
    gamma = np.exp(gamma_log - m)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((T-1,2,2))
    for t in range(T-1):
        M = np.zeros((2,2))
        M[0,0]=alpha[t,0]+np.log(T00+1e-12)+np.log(py_eat[t+1]+1e-12)+beta[t+1,0]
        M[0,1]=alpha[t,0]+np.log(T01+1e-12)+np.log(py_sh[t+1]+1e-12) +beta[t+1,1]
        M[1,0]=alpha[t,1]+np.log(T10+1e-12)+np.log(py_eat[t+1]+1e-12)+beta[t+1,0]
        M[1,1]=alpha[t,1]+np.log(T11+1e-12)+np.log(py_sh[t+1]+1e-12) +beta[t+1,1]
        m2 = np.max(M); Z = np.exp(M-m2).sum()
        xi[t,:,:] = np.exp(M-m2)/Z

    return ll, gamma, xi

def hmm_em_single(y, max_iter=500, tol=1e-9, init=None):
    """EM (Baum-Welch-like) for a single sequence; returns params, ll, gamma, ll_trace."""
    if init is None:
        q, r, pi, eps = 0.1, 0.9, 0.05, 0.01
    else:
        q, r, pi, eps = init["q"], init["r"], init["pi"], init["eps"]

    prev_ll = -np.inf
    ll_trace = []
    for _ in range(max_iter):
        ll, gamma, xi = hmm_forward_backward(y, q, r, pi, eps)
        ll_trace.append(ll)
        ge, gs = gamma[:,0], gamma[:,1]
        q_new = float(np.clip((ge*y).sum()/max(ge.sum(),1e-12), 1e-6, 1-1e-6))
        r_new = float(np.clip((gs*y).sum()/max(gs.sum(),1e-12), 1e-6, 1-1e-6))
        pi_new  = float(np.clip(xi[:,0,1].sum()/max(gamma[:-1,0].sum(),1e-12), 1e-6, 1-1e-6))
        eps_new = float(np.clip(xi[:,1,0].sum()/max(gamma[:-1,1].sum(),1e-12), 1e-6, 1-1e-6))
        if abs(ll - prev_ll) < tol:
            break
        q, r, pi, eps = q_new, r_new, pi_new, eps_new
        prev_ll = ll
    ll, gamma, xi = hmm_forward_backward(y, q, r, pi, eps)
    return {"q": q, "r": r, "pi": pi, "eps": eps}, ll, gamma, ll_trace

def plot_model_comparison(global_aic_table, save_path=None, show=True):
    """
    Generate an AIC comparison bar plot for the global models.
    """
    palette = {
        "IID": "#748067",
        "Trend": "#386C0B",
        "CP": "#7AE582",
        "CP_per_trial": "#2E7D32", # Darker green
        "HMM": "#72A1E5"
    }

    df = global_aic_table.copy()
    # Cleaning for display
    def clean_name(n):
        if "_global" in n: return n.replace("_global","")
        if "_tau" in n: return n.replace("_tau","")
        return n
    df['Model'] = [clean_name(el) for el in df['Model']]
    
    # Assign colors by model name
    colors = [palette[m] if m in palette else "grey" for m in df["Model"]]

    plt.figure(figsize=(6, 4))
    plt.bar(df["Model"], df["AIC"], color=colors)

    plt.ylabel("AIC", fontsize=13)
    
    # Style: remove top/right frame, thicken bottom/left
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2, labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Model comparison plot saved to {save_path}")

    if show:
        plt.show()

def hmm_global_likelihood_aic(data, params):
    """
    Compute global HMM log-likelihood and AIC for a given parameter set.
    """
    q, r, pi, eps = params["q"], params["r"], params["pi"], params["eps"]
    total_ll = 0.0
    for d in data:
        y = np.asarray(d["y"]).astype(int)
        # Use existing forward_backward logic
        ll_i, _, _ = hmm_forward_backward(y, q, r, pi, eps)
        total_ll += ll_i
    
    k = 4  # q, r, pi, eps
    aic = 2*k - 2*total_ll
    return total_ll, aic

def plot_hmm_aic_landscape(data, fitted_params, hmm_variations=None, save_path=None, show=True):
    """
    Generate 2D AIC landscape slices for HMM parameters.
    """
    # Grid settings
    n_res = 30 # Reduced from 50 for speed, feel free to increase
    q_vals = np.linspace(0, 0.9, n_res)
    r_vals = np.linspace(0, 0.9, n_res)
    eps_vals = np.linspace(0, 0.9, n_res)

    q0, r0, pi0, eps0 = fitted_params['q'], fitted_params['r'], fitted_params['pi'], fitted_params['eps']

    # Initialize planes
    AIC_qr   = np.zeros((len(q_vals), len(r_vals)))
    AIC_qeps = np.zeros((len(q_vals), len(eps_vals)))
    AIC_reps = np.zeros((len(r_vals), len(eps_vals)))

    print("Computing AIC landscape slices (this may take a minute)...")
    for i, q in enumerate(q_vals):
        for j, r in enumerate(r_vals):
            AIC_qr[i, j] = hmm_global_likelihood_aic(data, {"q": q, "r": r, "pi": pi0, "eps": eps0})[1]

    for i, q in enumerate(q_vals):
        for k, eps in enumerate(eps_vals):
            AIC_qeps[i, k] = hmm_global_likelihood_aic(data, {"q": q, "r": r0, "pi": pi0, "eps": eps})[1]

    for j, r in enumerate(r_vals):
        for k, eps in enumerate(eps_vals):
            AIC_reps[j, k] = hmm_global_likelihood_aic(data, {"q": q0, "r": r, "pi": pi0, "eps": eps})[1]

    # Scaling
    all_aic = np.concatenate([AIC_qr.flatten(), AIC_qeps.flatten(), AIC_reps.flatten()])
    vmax = np.percentile(all_aic, 85)
    vmin = np.percentile(all_aic, 5)
    cmap = "Greys_r"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    def make_contour(ax, x, y, z, xlabel, ylabel, title, fitted, variations):
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x.flatten(), y.flatten()), z.flatten(), (Xi, Yi), method='cubic')

        cf = ax.contourf(Xi, Yi, Zi, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(Xi, Yi, Zi, levels=10, colors='k', alpha=0.3, linewidths=0.5)

        # Fitted point
        ax.scatter(fitted[0], fitted[1], color='#ff7f0e', s=250, edgecolor='#ff7f0e', label='Fitted HMM', zorder=5)

        # Variation points
        if variations:
            for k, v in variations.items():
                ax.scatter(v[0], v[1], color='white', s=250, edgecolor='#ff7f0e', 
                           label=k if k != 'fitted' else None, zorder=4, linewidths=3)

        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_xlim(-0.03, 0.93)
        ax.set_ylim(-0.03, 0.93)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelsize=18)
        return cf

    # Plot planes
    R_qr, Q_qr = np.meshgrid(r_vals, q_vals)
    make_contour(axes[0], R_qr, Q_qr, AIC_qr, 'r', 'q', "AIC q-r plane", (r0, q0),
                 {k: (v['r'], v['q']) for k, v in hmm_variations.items() if k != 'fitted'} if hmm_variations else None)

    EPS_qe, Q_qe = np.meshgrid(eps_vals, q_vals)
    make_contour(axes[1], EPS_qe, Q_qe, AIC_qeps, 'ε', 'q', "AIC q-ε plane", (eps0, q0),
                 {k: (v['eps'], v['q']) for k, v in hmm_variations.items() if k != 'fitted'} if hmm_variations else None)

    EPS_re, R_re = np.meshgrid(eps_vals, r_vals)
    make_contour(axes[2], EPS_re, R_re, AIC_reps, 'ε', 'r', "AIC r-ε plane", (eps0, r0),
                 {k: (v['eps'], v['r']) for k, v in hmm_variations.items() if k != 'fitted'} if hmm_variations else None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"HMM AIC landscape saved to {save_path}")

    if show:
        plt.show()

def bivariate_hmm_global_likelihood_aic(path_to_param_data, data, params):
    qy, qz = params['qy'], params['qz']
    ry, rz = params['ry'], params['rz']
    pi, eps = params['pi'], params['eps']
    
    # If the user passed structured_data without 'z', load the data with 'z' just like fit_bivariate_hmm does
    if len(data) > 0 and 'z' not in data[0]:
        SH_data = pd.read_csv(path_to_param_data)
        SH_data['trial'] = SH_data['Date'] + SH_data['ID']
        new_data = []
        for t in np.unique(SH_data['trial']):
            sub = SH_data[SH_data['trial'] == t]
            mask = (~sub['scatter-hoard'].isna()) & (~sub['rotation'].isna())
            if mask.sum() > 0:
                y = sub.loc[mask, 'scatter-hoard'].values.astype(int)
                z = sub.loc[mask, 'rotation'].values.astype(int)
                new_data.append({"y": y, "z": z})
        data = new_data
        
    total_ll = 0.0
    for d in data:
        y = np.asarray(d["y"]).astype(int)
        z = np.asarray(d["z"]).astype(int)
        ll_i, _, _ = hmm_bivariate_forward_backward(y, z, qy, qz, ry, rz, pi, eps)
        total_ll += ll_i
        
    k = 6 # qy, qz, ry, rz, pi, eps
    aic = 2*k - 2*total_ll
    return total_ll, aic

def plot_bivariate_hmm_aic_landscape(path_param, data, fitted_params, save_path=None, show=True):
    """
    Generate 2D AIC landscape slices for Bivariate HMM parameters.
    Plots qy vs ry and qz vs rz.
    """
    # Grid settings
    n_res = 30
    qy_vals = np.linspace(0, 0.9, n_res)
    ry_vals = np.linspace(0, 0.9, n_res)
    qz_vals = np.linspace(0, 0.9, n_res)
    rz_vals = np.linspace(0, 0.9, n_res)
    eps_vals = np.linspace(0, 0.9, n_res)

    # Use aliases if present
    qy0 = fitted_params.get('qy', fitted_params.get('q_y'))
    qz0 = fitted_params.get('qz', fitted_params.get('q_z'))
    ry0 = fitted_params.get('ry', fitted_params.get('r_y'))
    rz0 = fitted_params.get('rz', fitted_params.get('r_z'))
    pi0 = fitted_params['pi']
    eps0 = fitted_params['eps']

    # Initialize planes
    AIC_qy_ry = np.zeros((len(qy_vals), len(ry_vals)))
    AIC_qy_eps = np.zeros((len(qy_vals), len(eps_vals)))
    AIC_ry_eps = np.zeros((len(ry_vals), len(eps_vals)))
    AIC_qz_rz = np.zeros((len(qz_vals), len(rz_vals)))

    print("Computing Bivariate HMM AIC landscape slices...")
    for i, qy in enumerate(qy_vals):
        for j, ry in enumerate(ry_vals):
            AIC_qy_ry[i, j] = bivariate_hmm_global_likelihood_aic(path_param, 
                data, {'qy': qy, 'qz': qz0, 'ry': ry, 'rz': rz0, 'pi': pi0, 'eps': eps0}
            )[1]

    for i, qy in enumerate(qy_vals):
        for k, eps in enumerate(eps_vals):
            AIC_qy_eps[i, k] = bivariate_hmm_global_likelihood_aic(path_param,
                data, {'qy': qy, 'qz': qz0, 'ry': ry0, 'rz': rz0, 'pi': pi0, 'eps': eps}
            )[1]

    for j, ry in enumerate(ry_vals):
        for k, eps in enumerate(eps_vals):
            AIC_ry_eps[j, k] = bivariate_hmm_global_likelihood_aic(path_param, 
                data, {'qy': qy0, 'qz': qz0, 'ry': ry, 'rz': rz0, 'pi': pi0, 'eps': eps}
            )[1]

    for i, qz in enumerate(qz_vals):
        for j, rz in enumerate(rz_vals):
            AIC_qz_rz[i, j] = bivariate_hmm_global_likelihood_aic(path_param,
                data, {'qy': qy0, 'qz': qz, 'ry': ry0, 'rz': rz, 'pi': pi0, 'eps': eps0}
            )[1]

    # Scaling
    all_aic = np.concatenate([AIC_qy_ry.flatten(), AIC_qy_eps.flatten(), AIC_ry_eps.flatten(), AIC_qz_rz.flatten()])
    vmax = np.percentile(all_aic, 85)
    vmin = np.percentile(all_aic, 5)
    
    print(f"Minimal AIC in computed landscape: {np.min(all_aic):.4f}")
    
    cmap = "Greys_r"

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    def make_contour(ax, x, y, z, xlabel, ylabel, title, fitted):
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x.flatten(), y.flatten()), z.flatten(), (Xi, Yi), method='cubic')

        cf = ax.contourf(Xi, Yi, Zi, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.contour(Xi, Yi, Zi, levels=10, colors='k', alpha=0.3, linewidths=0.5)

        # Fitted point
        ax.scatter(fitted[0], fitted[1], color='#ff7f0e', s=250, edgecolor='#ff7f0e', label='Fitted Bivariate HMM', zorder=5)

        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_xlim(-0.03, 0.93)
        ax.set_ylim(-0.03, 0.93)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelsize=18)
        return cf

    # Plot planes
    RY_y, QY_y = np.meshgrid(ry_vals, qy_vals)
    make_contour(axes[0], RY_y, QY_y, AIC_qy_ry, 'ry (Outcome SH)', 'qy (Outcome Eat)', "AIC qy-ry plane", (ry0, qy0))

    EPS_qy, QY_eps = np.meshgrid(eps_vals, qy_vals)
    make_contour(axes[1], EPS_qy, QY_eps, AIC_qy_eps, 'ε (eps)', 'qy (Outcome Eat)', "AIC qy-ε plane", (eps0, qy0))

    EPS_ry, RY_eps = np.meshgrid(eps_vals, ry_vals)
    make_contour(axes[2], EPS_ry, RY_eps, AIC_ry_eps, 'ε (eps)', 'ry (Outcome SH)', "AIC ry-ε plane", (eps0, ry0))

    RZ_z, QZ_z = np.meshgrid(rz_vals, qz_vals)
    cf = make_contour(axes[3], RZ_z, QZ_z, AIC_qz_rz, 'rz (Rotation SH)', 'qz (Rotation Eat)', "AIC qz-rz plane", (rz0, qz0))

    cbar = fig.colorbar(cf, ax=axes, pad=0.02, aspect=40)
    cbar.set_label('AIC', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Bivariate HMM AIC landscape saved to {save_path}")

    if show:
        plt.show()

def iid_emit(data, params, rng=None):
    """IID Bernoulli model (constant p)."""
    if rng is None: rng = np.random.default_rng()
    p = params["p"]
    out_y, out_p = [], []
    for d in data:
        n = len(d["y"])
        probs = np.full(n, p)
        y_sim = (rng.random(n) < probs).astype(int)
        out_y.append(y_sim)
        out_p.append(probs)
    return out_y, out_p

def logistic_emit(data, params, rng=None):
    """Logistic model with global slope and intercept."""
    if rng is None: rng = np.random.default_rng()
    out_y, out_p = [], []
    for d in data:
        t = np.asarray(d["t"]).astype(float)
        t_scaled = (t - t.min()) / (t.max() - t.min() + 1e-12)
        p = expit(params["a"] + params["dlt"] * t_scaled)
        y_sim = (rng.random(len(t)) < p).astype(int)
        out_y.append(y_sim)
        out_p.append(p)
    return out_y, out_p

def changepoint_emit(data, params, rng=None):
    """Piecewise-constant model with shared p0, p1, and shared (global) tau."""
    if rng is None: rng = np.random.default_rng()
    tau = params.get("taus", params.get("tau", 0))
    p0 = params["p0"]
    p1 = params["p1"]
    out_y, out_p = [], []
    for i, d in enumerate(data):
        n = len(d["y"])
        # Handle per-trial tau if provided (either as "taus" list or "tau" scalar)
        tau_i = tau[i] if isinstance(tau, (list, np.ndarray)) else tau
        p = np.concatenate([
            np.repeat(p0, min(tau_i, n)),
            np.repeat(p1, max(0, n - tau_i))
        ])
        y_sim = (rng.random(n) < p).astype(int)
        out_y.append(y_sim)
        out_p.append(p)
    return out_y, out_p

def hmm_emit(data, params, force_emit=True, max_tries=1000, rng=None):
    """Simulate 2-state absorbing HMM sequences."""
    if rng is None: rng = np.random.default_rng()
    q, r, pi, eps = params["q"], params["r"], params["pi"], params["eps"]
    out_y, out_p, out_s = [], [], []
    for d in data:
        n = len(d["y"])
        for _ in range(max_tries):
            s = np.zeros(n, dtype=int)
            rand_pi = rng.random(n)
            rand_eps = rng.random(n)
            for t in range(1, n):
                if s[t - 1] == 0:
                    s[t] = 1 if rand_pi[t] < pi else 0
                else:
                    s[t] = 0 if rand_eps[t] < eps else 1
            
            p_emit = np.where(s == 0, q, r)
            y_sim = (rng.random(n) < p_emit).astype(int)
            
            if not force_emit or np.any(y_sim == 1):
                out_y.append(y_sim)
                out_s.append(s)
                out_p.append(p_emit)
                break
        else:
            raise RuntimeError(f"No emission=1 after {max_tries} tries for sequence length {n}")
    return out_y, out_p, out_s

def hsmm_emit(data, params, force_emit=True, max_tries=1000, rng=None):
    if rng is None: rng = np.random.default_rng()
    q, r, lam0, lam1 = params["q"], params["r"], params["lam0"], params["lam1"]
    out_y, out_p, out_s = [], [], []
    for d in data:
        n = len(d["y"])
        for _ in range(max_tries):
            s = np.zeros(n, dtype=int)
            y_sim = np.zeros(n, dtype=int)
            p_emit = np.zeros(n, dtype=float)
            
            t = 0
            curr_state = 0 # Start in Eat
            while t < n:
                lam = lam0 if curr_state == 0 else lam1
                bout_len = 1 + rng.poisson(lam)
                end_t = min(t + bout_len, n)
                
                s[t:end_t] = curr_state
                prob = q if curr_state == 0 else r
                p_emit[t:end_t] = prob
                y_sim[t:end_t] = (rng.random(end_t - t) < prob).astype(int)
                
                t += bout_len
                curr_state = 1 - curr_state
                
            if not force_emit or np.any(y_sim == 1):
                out_y.append(y_sim)
                out_p.append(p_emit)
                out_s.append(s)
                break
        else:
            raise RuntimeError(f"No emission=1 after {max_tries} tries for sequence length {n}")
    return out_y, out_p, out_s

def run_models_with_global_params(data, global_params, seed=0):
    """Wrapper to simulate data from all fitted models."""
    rng = np.random.default_rng(seed)
    outputs = {}
    outputs["IID_y"], outputs["IID_p"] = iid_emit(data, global_params["IID"], rng=rng)
    outputs["Logistic_y"], outputs["Logistic_p"] = logistic_emit(data, global_params["Logistic"], rng=rng)
    outputs["Change_y"], outputs["Change_p"] = changepoint_emit(data, global_params["Change-point"], rng=rng)
    outputs["HMM_y"], outputs["HMM_p"], outputs["HMM_s"] = hmm_emit(data, global_params["HMM"], rng=rng)
    outputs["HSMM_y"], outputs["HSMM_p"], outputs["HSMM_s"] = hsmm_emit(data, global_params["HSMM"], rng=rng)
    return outputs

# -----------------------------
# 4) Global (pooled) fits: one parameter set across animals
#    IMPORTANT: Change-point is now pooled with a single global tau.
# -----------------------------

# IID global: single p (or logit a)
def iid_global_fit(data):
    ys = np.concatenate([np.asarray(d["y"]).astype(int) for d in data])
    p = np.clip(ys.mean(), 1e-12, 1-1e-12)
    ll = float(np.sum(ys*np.log(p) + (1-ys)*np.log(1-p)))
    return {"p": p}, ll

# Trend global: fit one (a, dlt) across all trials (pooling rows)
def logistic_global_fit(data):
    X_list, y_list = [], []
    for d in data:
        y = np.asarray(d["y"]).astype(int)
        t = np.asarray(d["t"]).astype(float)
        t_scaled = (t - t.min()) / (t.max() - t.min() + 1e-12)
        X = np.column_stack([np.ones_like(t), t_scaled])
        X_list.append(X)
        y_list.append(y)
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    beta, ll, _ = logistic_fit(X_all, y_all, track_ll=True)
    return {"a": float(beta[0]), "dlt": float(beta[1])}, ll

# Change-point global: single tau shared across animals; find tau,p0,p1 maximizing joint log-likelihood
def changepoint_global_fit(data, min_seg=3):
    # allowable tau range must fit inside the shortest sequence
    min_len = min(len(d["y"]) for d in data)
    best = None
    best_tau = None
    # search tau across [min_seg, min_len - min_seg]
    lower = min_seg
    upper = max(min_len - min_seg, min_seg)
    if upper < lower:
        # sequences too short: fallback to no-change p
        ys = np.concatenate([d["y"] for d in data])
        p = np.clip(ys.mean(), 1e-12, 1-1e-12)
        total_ll = float(np.sum(ys*np.log(p) + (1-ys)*np.log(1-p)))
        return {"p0": p, "p1": p, "tau": 0}, total_ll

    for tau in range(lower, upper+1):
        # pool pre and post segments across animals using the same tau
        pre_bits = []
        post_bits = []
        for d in data:
            y = np.asarray(d["y"]).astype(int)
            n = len(y)
            # If animal shorter than tau, treat entire sequence as pre (or skip? we include clipped)
            pre_bits.append(y[:min(tau, n)])
            if n > tau:
                post_bits.append(y[tau:])
        if len(pre_bits) == 0:
            p0 = 1e-12
        else:
            p0 = np.clip(np.concatenate(pre_bits).mean(), 1e-12, 1-1e-12)
        if len(post_bits) == 0:
            p1 = 1e-12
        else:
            p1 = np.clip(np.concatenate(post_bits).mean(), 1e-12, 1-1e-12)

        total_ll = 0.0
        for d in data:
            y = np.asarray(d["y"]).astype(int)
            n = len(y)
            # compute ll using the global tau but clip by sequence length
            cut = min(tau, n)
            s0 = int(y[:cut].sum()) if cut>0 else 0
            n0 = max(1, cut)
            total_ll += s0*np.log(p0) + (n0 - s0)*np.log(1 - p0)
            if n > tau:
                s1 = int(y[tau:].sum())
                n1 = n - tau
                total_ll += s1*np.log(p1) + (n1 - s1) * np.log(1 - p1)
        if (best is None) or (total_ll > best):
            best = total_ll
            best_tau = tau
            best_p0 = p0
            best_p1 = p1
    return {"p0": float(best_p0), "p1": float(best_p1), "tau": int(best_tau)}, float(best)


# HMM global: EM across animals (E-step sums expected counts across animals)
def hmm_em_global(data, max_iter=500, tol=1e-9, init=None):
    if init is None:
        q, r, pi, eps = 0.1, 0.9, 0.05, 0.01
    else:
        q, r, pi, eps = init["q"], init["r"], init["pi"], init["eps"]

    prev_ll = -np.inf
    ll_trace = []
    for it in range(max_iter):
        # accumulate expected counts across animals
        num_q = num_r = 0.0
        den_q = den_r = 0.0
        num_pi = den_pi = 0.0
        num_eps = den_eps = 0.0
        total_ll = 0.0

        for d in data:
            y = np.asarray(d["y"]).astype(int)
            ll_i, gamma_i, xi_i = hmm_forward_backward(y, q, r, pi, eps)
            total_ll += ll_i
            ge, gs = gamma_i[:,0], gamma_i[:,1]
            num_q += (ge * y).sum(); den_q += ge.sum()
            num_r += (gs * y).sum(); den_r += gs.sum()
            if len(xi_i) > 0:
                num_pi += xi_i[:,0,1].sum(); den_pi += gamma_i[:-1,0].sum()
                num_eps += xi_i[:,1,0].sum(); den_eps += gamma_i[:-1,1].sum()

        # M-step
        q_new = float(np.clip(num_q / max(den_q, 1e-12), 1e-6, 1-1e-6))
        r_new = float(np.clip(num_r / max(den_r, 1e-12), 1e-6, 1-1e-6))
        pi_new = float(np.clip(num_pi / max(den_pi, 1e-12), 1e-6, 1-1e-6))
        eps_new = float(np.clip(num_eps / max(den_eps, 1e-12), 1e-6, 1-1e-6))

        ll_trace.append(total_ll)
        if abs(total_ll - prev_ll) < tol:
            break
        q, r, pi, eps = q_new, r_new, pi_new, eps_new
        prev_ll = total_ll

    # final posteriors per animal under global params
    gammas = [hmm_forward_backward(np.asarray(d["y"]).astype(int), q, r, pi, eps)[1] for d in data]
    return {"q": q, "r": r, "pi": pi, "eps": eps}, float(prev_ll), gammas, ll_trace

def hmm_bivariate_forward_backward(y, z, qy, qz, ry, rz, pi, eps):
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=int)
    T = len(y)
    py_eat = np.where(y == 1, qy, 1.0 - qy)
    pz_eat = np.where(z == 1, qz, 1.0 - qz)
    p_eat = py_eat * pz_eat

    py_sh = np.where(y == 1, ry, 1.0 - ry)
    pz_sh = np.where(z == 1, rz, 1.0 - rz)
    p_sh = py_sh * pz_sh

    T00 = (1.0 - pi); T01 = pi
    T10 = eps;        T11 = (1.0 - eps)

    alpha = np.zeros((T, 2))
    alpha[0, 0] = np.log(p_eat[0] + 1e-12)
    alpha[0, 1] = -1e12 + np.log(p_sh[0] + 1e-12)

    for t in range(1, T):
        a0 = alpha[t-1, 0] + np.log(T00 + 1e-12)
        a1 = alpha[t-1, 1] + np.log(T10 + 1e-12)
        alpha[t, 0] = np.log(p_eat[t] + 1e-12) + np.logaddexp(a0, a1)

        b0 = alpha[t-1, 0] + np.log(T01 + 1e-12)
        b1 = alpha[t-1, 1] + np.log(T11 + 1e-12)
        alpha[t, 1] = np.log(p_sh[t] + 1e-12) + np.logaddexp(b0, b1)

    ll = float(np.logaddexp(alpha[-1, 0], alpha[-1, 1]))

    beta = np.zeros((T, 2))
    beta[-1, :] = 0.0
    for t in range(T-2, -1, -1):
        term0 = np.log(T00 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
        term1 = np.log(T01 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
        beta[t, 0] = np.logaddexp(term0, term1)

        term0 = np.log(T10 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
        term1 = np.log(T11 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
        beta[t, 1] = np.logaddexp(term0, term1)

    gamma_log = alpha + beta
    m = np.max(gamma_log, axis=1, keepdims=True)
    gamma = np.exp(gamma_log - m)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((T-1, 2, 2))
    for t in range(T-1):
        M = np.zeros((2, 2))
        M[0, 0] = alpha[t, 0] + np.log(T00 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
        M[0, 1] = alpha[t, 0] + np.log(T01 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
        M[1, 0] = alpha[t, 1] + np.log(T10 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
        M[1, 1] = alpha[t, 1] + np.log(T11 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
        m2 = np.max(M); Z = np.exp(M - m2).sum()
        xi[t, :, :] = np.exp(M - m2) / Z

    return ll, gamma, xi


def hsmm_log_likelihood_single(y, q, r, lam0, lam1, max_d=50):
    y = np.asarray(y, dtype=int)
    T = len(y)
    max_d = min(T, max_d)
    
    log_py_s0 = np.where(y == 1, np.log(q + 1e-12), np.log(1.0 - q + 1e-12))
    log_py_s1 = np.where(y == 1, np.log(r + 1e-12), np.log(1.0 - r + 1e-12))
    
    cum_s0 = np.concatenate([[0.0], np.cumsum(log_py_s0)])
    cum_s1 = np.concatenate([[0.0], np.cumsum(log_py_s1)])
    
    ds = np.arange(1, max_d + 1)
    log_p_d_s0 = poisson.logpmf(ds - 1, lam0)
    log_p_d_s1 = poisson.logpmf(ds - 1, lam1)
    
    log_sf_s0 = poisson.logsf(ds - 1, lam0)
    log_sf_s1 = poisson.logsf(ds - 1, lam1)
    
    alpha = np.full((T + 1, 2), -np.inf)
    pi0 = [1.0, 0.0]
    
    for d in range(1, max_d + 1):
        if pi0[0] > 0:
            alpha[d, 0] = np.log(pi0[0]) + log_p_d_s0[d - 1] + cum_s0[d]
        if pi0[1] > 0:
            alpha[d, 1] = np.log(pi0[1]) + log_p_d_s1[d - 1] + cum_s1[d]
            
    for t in range(2, T + 1):
        limit = min(t - 1, max_d)
        if limit < 1:
            continue
            
        prev_log_s1 = alpha[t - limit : t, 1][::-1]
        log_terms_0 = prev_log_s1 + log_p_d_s0[:limit] + (cum_s0[t] - cum_s0[t - limit : t][::-1])
        if np.any(log_terms_0 != -np.inf):
            alpha[t, 0] = np.logaddexp(alpha[t, 0], np.logaddexp.reduce(log_terms_0))
        
        prev_log_s0 = alpha[t - limit : t, 0][::-1]
        log_terms_1 = prev_log_s0 + log_p_d_s1[:limit] + (cum_s1[t] - cum_s1[t - limit : t][::-1])
        if np.any(log_terms_1 != -np.inf):
            alpha[t, 1] = np.logaddexp(alpha[t, 1], np.logaddexp.reduce(log_terms_1))
        
    total_log_lik = np.logaddexp(alpha[T, 0], alpha[T, 1])
    
    limit = min(T - 1, max_d)
    if limit >= 1:
        prev_log_s1 = alpha[T - limit : T, 1][::-1]
        terms_ongoing_0 = prev_log_s1 + log_sf_s0[:limit] + (cum_s0[T] - cum_s0[T - limit : T][::-1])
        if np.any(terms_ongoing_0 != -np.inf):
            total_log_lik = np.logaddexp(total_log_lik, np.logaddexp.reduce(terms_ongoing_0))
        
        prev_log_s0 = alpha[T - limit : T, 0][::-1]
        terms_ongoing_1 = prev_log_s0 + log_sf_s1[:limit] + (cum_s1[T] - cum_s1[T - limit : T][::-1])
        if np.any(terms_ongoing_1 != -np.inf):
            total_log_lik = np.logaddexp(total_log_lik, np.logaddexp.reduce(terms_ongoing_1))
        
    if T <= max_d:
        if pi0[0] > 0:
            total_log_lik = np.logaddexp(total_log_lik, np.log(pi0[0]) + log_sf_s0[T - 1] + cum_s0[T])
        if pi0[1] > 0:
            total_log_lik = np.logaddexp(total_log_lik, np.log(pi0[1]) + log_sf_s1[T - 1] + cum_s1[T])
            
    return total_log_lik

def hsmm_global_fit(data, max_d=50):
    def objective(params):
        q, r, lam0, lam1 = params
        total_ll = 0.0
        for d in data:
            ll = hsmm_log_likelihood_single(d["y"], q, r, lam0, lam1, max_d=max_d)
            total_ll += ll
        return -total_ll

    init_params = [0.1, 0.9, 10.0, 10.0]
    bounds = [(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6), (0.1, 50.0), (0.1, 50.0)]
    
    res = minimize(objective, init_params, method="L-BFGS-B", bounds=bounds)
    q, r, lam0, lam1 = res.x
    return {"q": float(q), "r": float(r), "lam0": float(lam0), "lam1": float(lam1)}, -res.fun

def calculate_test_ll(data_point, model_name, params):
    """Compute log-likelihood of a held-out subject given global parameters."""
    y = np.asarray(data_point["y"]).astype(int)
    n = len(y)
    eps = 1e-12
    
    if model_name == "IID_global":
        p = np.clip(params["p"], eps, 1-eps)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        
    elif model_name == "Trend_global":
        t = np.asarray(data_point["t"]).astype(float)
        t_scaled = (t - t.min()) / (t.max() - t.min() + 1e-12)
        p = expit(params["a"] + params["dlt"] * t_scaled)
        p = np.clip(p, eps, 1-eps)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        
    elif model_name == "CP_global":
        tau = params["tau"]
        p0 = np.clip(params["p0"], eps, 1-eps)
        p1 = np.clip(params["p1"], eps, 1-eps)
        cut = min(tau, n)
        ll = 0.0
        if cut > 0:
            s0 = y[:cut].sum()
            ll += s0 * np.log(p0) + (cut - s0) * np.log(1 - p0)
        if n > tau:
            s1 = y[tau:].sum()
            n1 = n - tau
            ll += s1 * np.log(p1) + (n1 - s1) * np.log(1 - p1)
        return float(ll)
        
    elif model_name == "CP_per_trial":
        # Find best tau for this test animal given global p0, p1
        p0 = np.clip(params["p0"], eps, 1-eps)
        p1 = np.clip(params["p1"], eps, 1-eps)
        best_ll = -np.inf
        min_seg = 3
        for tau in range(min_seg, max(min_seg + 1, n - min_seg + 1)):
            cut = min(tau, n)
            ll = 0.0
            if cut > 0:
                s0 = y[:cut].sum()
                ll += s0 * np.log(p0) + (cut - s0) * np.log(1 - p0)
            if n > tau:
                s1 = y[tau:].sum()
                n1 = n - tau
                ll += s1 * np.log(p1) + (n1 - s1) * np.log(1 - p1)
            if ll > best_ll:
                best_ll = ll
        return float(best_ll if best_ll != -np.inf else 0.0)
        
    elif model_name == "HMM_global":
        q, r, pi, ev = params["q"], params["r"], params["pi"], params["eps"]
        ll, _, _ = hmm_forward_backward(y, q, r, pi, ev)
        return float(ll)
        
    elif model_name == "HSMM_global":
        return float(hsmm_log_likelihood_single(y, params["q"], params["r"], params["lam0"], params["lam1"]))
    
    return 0.0

def run_loocv_comparison(data):
    """Perform Leave-One-Out Cross-Validation on pooled models."""
    n_subjects = len(data)
    models = ["IID_global", "Trend_global", "CP_global", "HMM_global", "HSMM_global"]
    total_test_ll = {m: 0.0 for m in models}
    
    print(f"\n[LOOCV] Running subject-level cross-validation ({n_subjects} animals)...")
    from tqdm import tqdm
    for i in tqdm(range(n_subjects)):
        train_data = data[:i] + data[i+1:]
        test_data_point = data[i]
        
        # 1. IID
        p_iid, _ = iid_global_fit(train_data)
        total_test_ll["IID_global"] += calculate_test_ll(test_data_point, "IID_global", p_iid)
        
        # 2. Trend
        p_trend, _ = logistic_global_fit(train_data)
        total_test_ll["Trend_global"] += calculate_test_ll(test_data_point, "Trend_global", p_trend)
        
        # 3. CP Global
        p_cp_g, _ = changepoint_global_fit(train_data)
        total_test_ll["CP_global"] += calculate_test_ll(test_data_point, "CP_global", p_cp_g)
        
        # 4. HMM
        p_hmm, _, _, _ = hmm_em_global(train_data)
        total_test_ll["HMM_global"] += calculate_test_ll(test_data_point, "HMM_global", p_hmm)
        
        # 5. HSMM
        p_hsmm, _ = hsmm_global_fit(train_data)
        total_test_ll["HSMM_global"] += calculate_test_ll(test_data_point, "HSMM_global", p_hsmm)
        
    cv_results = []
    for m in models:
        cv_results.append({
            "Model": m.replace("_global", ""),
            "Predictive_LL": total_test_ll[m],
            "Avg_LL": total_test_ll[m] / n_subjects
        })
    return pd.DataFrame(cv_results).sort_values("Predictive_LL", ascending=False)

def run_full_pipeline(data):
    max_T = max(len(d["y"]) for d in data)
    fit_results = {"IID": [], "Trend": [], "CP": [], "HMM": []}

    for d in data:
        y = np.asarray(d["y"]).astype(int)
        t = np.asarray(d["t"]).astype(float)
        t_scaled = (t - t.min())/(t.max()-t.min()+1e-12)

        # (A) IID: logit p = a (single-parameter logistic on constant)
        Xa = np.ones((len(y),1))
        coef_a, ll_a, _ = logistic_fit(Xa, y, track_ll=True)
        k_a = 1
        fit_results["IID"].append({"coef": coef_a, "ll": ll_a, "aic": 2*k_a - 2*ll_a})

        # (B) Trend: logit p = a + d*t_scaled
        Xb = np.column_stack([np.ones_like(t), t_scaled])
        coef_b, ll_b, _ = logistic_fit(Xb, y, track_ll=True)
        k_b = 2
        fit_results["Trend"].append({"coef": coef_b, "ll": ll_b, "aic": 2*k_b - 2*ll_b})

        # (C) Change-point (per-animal tau) -- keep for per-animal diagnostics
        tau, p0, p1, ll_c = changepoint_fit(y, min_seg=3)
        k_c = 3  # used only for per-animal diagnostics
        fit_results["CP"].append({"tau": tau, "p0": p0, "p1": p1, "ll": ll_c, "aic": 2*k_c - 2*ll_c})

        # (D) HMM (per-animal)
        params_d, ll_d, gamma_d, ll_trace_d = hmm_em_single(y, max_iter=500, tol=1e-9)
        k_d = 4  # q, r, pi, eps
        fit_results["HMM"].append({"params": params_d, "ll": ll_d, "aic": 2*k_d - 2*ll_d, "gamma": gamma_d})

    # Summarize per-animal AIC sums (diagnostic)
    rows = []
    for name in ["IID","Trend","CP","HMM"]:
        aics = [r["aic"] for r in fit_results[name]]
        rows.append({"Model": name, "AIC_sum": float(np.sum(aics)), "AIC_mean": float(np.mean(aics))})
    aic_table_per_animal = pd.DataFrame(rows).sort_values("AIC_sum")
    print("Per-animal model comparison (sum of AIC across animals):")
    print(aic_table_per_animal)
    print()

    # Run global fits using top-level functions
    iid_global, iid_global_ll = iid_global_fit(data)
    trend_global, trend_global_ll = logistic_global_fit(data)
    cp_global, cp_global_ll = changepoint_global_fit(data)
    hmm_global, hmm_global_ll, gammas_global, hmm_global_trace = hmm_em_global(data, max_iter=500, tol=1e-9)
    hsmm_global, hsmm_global_ll = hsmm_global_fit(data)
    has_z = all("z" in d for d in data)

    # Run LOOCV
    loocv_table = run_loocv_comparison(data)
    print("LOOCV Model Comparison (Predictive Log-Likelihood):")
    print(loocv_table)
    print()

    # Compute global AICs (total log-likelihood across animals)
    global_rows = []
    global_rows.append({"Model": "IID_global", "k": 1, "LL": iid_global_ll, "AIC": 2*1 - 2*iid_global_ll})
    global_rows.append({"Model": "Trend_global", "k": 2, "LL": trend_global_ll, "AIC": 2*2 - 2*trend_global_ll})
    global_rows.append({"Model": "CP_global", "k": 3, "LL": cp_global_ll, "AIC": 2*3 - 2*cp_global_ll})
    global_rows.append({"Model": "HMM_global", "k": 4, "LL": hmm_global_ll, "AIC": 2*4 - 2*hmm_global_ll})
    global_rows.append({"Model": "HSMM_global", "k": 4, "LL": hsmm_global_ll, "AIC": 2*4 - 2*hsmm_global_ll})
    global_aic_table = pd.DataFrame(global_rows).sort_values("AIC")
    print("Global (pooled) model comparison:")
    print(global_aic_table)
    print()

    # -----------------------------
    # 5) Posterior-predictive-like diagnostics using global fits
    # -----------------------------
    def align_to_first_SH(y, max_left=20, max_right=60):
        idx = np.where(y==1)[0]
        k_counts = defaultdict(int); k_denoms = defaultdict(int)
        if len(idx)==0:
            return k_counts, k_denoms
        t0 = idx[0]
        for t in range(max(0, t0-max_left), min(len(y), t0+max_right+1)):
            k = t - t0
            k_counts[k] += int(y[t]==1)
            k_denoms[k] += 1
        return k_counts, k_denoms

    # empirical aligned curve
    max_left, max_right = 20, 55
    emp_counts = defaultdict(int); emp_denoms = defaultdict(int)
    for d in data:
        kc, kd = align_to_first_SH(np.asarray(d["y"]).astype(int), max_left, max_right)
        for k,v in kc.items(): emp_counts[k]+=v
        for k,v in kd.items(): emp_denoms[k]+=v
    ks = [k for k in range(-max_left, max_right+1)]


    emp_curve = np.array([emp_counts.get(k,0)/emp_denoms.get(k,1) for k in ks])

    # Simulation helpers using given params
    def sim_from_IID_param(p, T):
        return rng.binomial(1, p, size=T)

    def sim_from_Trend_param(a_d, T):
        a, dlt = a_d
        t = np.arange(1, T+1).astype(float)
        t_scaled = (t - t.min())/(t.max()-t.min()+1e-12)
        p = 1/(1+np.exp(-(a + dlt*t_scaled)))
        return rng.binomial(1, p)

    def sim_from_CP_param(tau, p0, p1, T, idx=None):
        # Handle per-trial tau if provided (either as list or scalar)
        t_i = tau[idx] if idx is not None and isinstance(tau, (list, np.ndarray)) else tau
        p = np.where(np.arange(T) < t_i, p0, p1)
        return rng.binomial(1, p)

    def sim_from_HMM_param(params, T):
        q = params["q"]; r = params["r"]; pi = params["pi"]; eps = params["eps"]
        z = np.zeros(T, dtype=int); y = np.zeros(T, dtype=int)
        for tt in range(T):
            if tt==0: z[tt]=0
            else:
                if z[tt-1]==0:
                    z[tt] = 1 if rng.random() < pi else 0
                else:
                    z[tt] = 0 if rng.random() < eps else 1
            y[tt] = rng.binomial(1, r if z[tt]==1 else q)
        return y

    def sim_from_HSMM_param(params, T):
        q, r, lam0, lam1 = params["q"], params["r"], params["lam0"], params["lam1"]
        y = np.zeros(T, dtype=int)
        t = 0
        curr_state = 0
        while t < T:
            lam = lam0 if curr_state == 0 else lam1
            bout = 1 + rng.poisson(lam)
            end = min(t + bout, T)
            prob = q if curr_state == 0 else r
            y[t:end] = rng.binomial(1, prob, size=end - t)
            t += bout
            curr_state = 1 - curr_state
        return y

    def sim_from_HMM_biv_param(params, T):
        qy, ry, pi, eps = params["qy"], params["ry"], params["pi"], params["eps"]
        s = np.zeros(T, dtype=int); y = np.zeros(T, dtype=int)
        for tt in range(T):
            if tt==0: s[tt]=0
            else:
                if s[tt-1]==0:
                    s[tt] = 1 if rng.random() < pi else 0
                else:
                    s[tt] = 0 if rng.random() < eps else 1
            y[tt] = rng.binomial(1, ry if s[tt]==1 else qy)
        return y

    def predictive_band_global(model_name, n_rep=200):
        sims = []
        for rep in range(n_rep):
            counts = defaultdict(int); denoms = defaultdict(int)
            for i, d in enumerate(data):
                T = len(d["y"])
                if model_name == "IID":
                    y_sim = sim_from_IID_param(iid_global["p"], T)
                elif model_name == "Trend":
                    y_sim = sim_from_Trend_param((trend_global["a"], trend_global["dlt"]), T)
                elif model_name == "CP":
                    y_sim = sim_from_CP_param(cp_global["tau"], cp_global["p0"], cp_global["p1"], T)
                elif model_name == "HMM":
                    y_sim = sim_from_HMM_param(hmm_global, T)
                elif model_name == "HSMM":
                    y_sim = sim_from_HSMM_param(hsmm_global, T)
                kc, kd = align_to_first_SH(y_sim, max_left, max_right)
                for k,v in kc.items(): counts[k]+=v
                for k,v in kd.items(): denoms[k]+=v
            curve = np.array([counts.get(k,0)/denoms.get(k,1) for k in ks])
            sims.append(curve)
        sims = np.stack(sims, axis=0)
        return sims.mean(axis=0), np.percentile(sims, 2.5, axis=0), np.percentile(sims, 97.5, axis=0)

    models_to_run = ["IID","Trend","CP","HMM","HSMM"]

    bands_global = {}
    for m in models_to_run:
        bands_global[m] = predictive_band_global(m, n_rep=120)

    palette = {
        "IID": "#748067",     
        "Trend": "#386C0B",   
        "CP": "#7AE582",      
        "HMM": "#72A1E5",     
        "HSMM": "#9C27B0",    
        "Real data": "#000000"  
    }

    # Plot Aligned SH (Figure 1)
    plt.figure(figsize=(10, 8))
    plt.plot(ks, emp_curve, label="Real data", linewidth=3, color="#730071")

    for m in models_to_run:
        mean, lo, hi = bands_global[m]
        color = palette[m]
        plt.plot(ks, mean, label=f"{m}", color=color, linewidth=3)
        # plt.fill_between(ks, lo, hi, alpha=0.2, color=color)

    plt.axvline(0, linestyle="--", linewidth=2, color="black")

    # Labels and title
    plt.xlabel("Epoch relative to first scatter-hoarding", fontsize=14)
    plt.ylabel("Probability of scatter-hoarding", fontsize=14)

    # Styling
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", fontsize=14, frameon=False)
    plt.tight_layout()
    
    if save_path_first_plot:
        plt.savefig(save_path_first_plot, dpi=300, bbox_inches='tight', transparent=True)
        print(f"First plot saved to {save_path_first_plot}")
        
    plt.show()

    # Post-switch run-length survival
    def post_switch_run_length(y):
        idx = np.where(y==1)[0]
        if len(idx)==0:
            return 0
        t0 = idx[0]
        L = 0
        for t in range(t0, len(y)):
            if y[t]==1: L += 1
            else: break
        return L

    def survival_from_lengths(Ls, Lmax=30):
        S = []
        for ell in range(1, Lmax+1):
            S.append(np.mean([1 if L>=ell else 0 for L in Ls]))
        return np.array(S)

    emp_L = [post_switch_run_length(np.asarray(d["y"]).astype(int)) for d in data if np.any(np.asarray(d["y"])==1)]
    Lmax = 30
    emp_S = survival_from_lengths(emp_L, Lmax=Lmax)

    def predictive_run_band_global(model_name, n_rep=120):
        all_S = []
        for rep in range(n_rep):
            Ls = []
            for i, d in enumerate(data):
                T = len(d["y"])
                if model_name == "IID":
                    y_sim = sim_from_IID_param(iid_global["p"], T)
                elif model_name == "Trend":
                    y_sim = sim_from_Trend_param((trend_global["a"], trend_global["dlt"]), T)
                elif model_name == "CP":
                    y_sim = sim_from_CP_param(cp_global["tau"], cp_global["p0"], cp_global["p1"], T)
                elif model_name == "CP_per_trial":
                    y_sim = sim_from_CP_param(cp_mixed["taus"], cp_mixed["p0"], cp_mixed["p1"], T, idx=i)
                elif model_name == "HMM":
                    y_sim = sim_from_HMM_param(hmm_global, T)
                elif model_name == "HSMM":
                    y_sim = sim_from_HSMM_param(hsmm_global, T)
                elif model_name == "HMM_bivariate":
                    y_sim = sim_from_HMM_biv_param(hmm_biv_global, T)
                if np.any(y_sim==1):
                    Ls.append(post_switch_run_length(y_sim))
            if len(Ls)==0: continue
            all_S.append(survival_from_lengths(Ls, Lmax=Lmax))
        arr = np.stack(all_S, axis=0)
        return arr.mean(axis=0), np.percentile(arr,2.5,axis=0), np.percentile(arr,97.5,axis=0)

    bands_run_global = {}
    for m in models_to_run:
        bands_run_global[m] = predictive_run_band_global(m, n_rep=120)

    palette_survival = {
        "IID": "#1A659E",     
        "Trend": "#713E5A",   
        "CP": "#7AE582",      
        "CP_per_trial": "#2E7D32", 
        "HMM": "#7F636E",     
        "HSMM": "#9C27B0",    
        "HMM_bivariate": "#FF5722",
        "Real data": "#000000"  
    }

    plt.figure(figsize=(10, 6))

    # Empirical survival curve
    plt.plot(
        np.arange(1, Lmax + 1),
        emp_S,
        label="Real data",
        linewidth=3,
        color="#000000"
    )

    # Model predictions + envelopes
    for m in models_to_run:
        mean, lo, hi = bands_run_global[m]
        color = palette_survival[m]
        
        # Mean curve
        plt.plot(
            np.arange(1, Lmax + 1),
            mean,
            label=f"{m}",
            color=color,
            linewidth=3
        )
        
        # Uncertainty band
        plt.fill_between(
            np.arange(1, Lmax + 1),
            lo,
            hi,
            alpha=0.2,
            color=color,
            linewidth=0
        )

    plt.xlabel("Number of scatter-hoarding after switching", fontsize=14)
    plt.ylabel("Probability of post-switch scatter-hoarding", fontsize=14)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", fontsize=14, frameon=False)
    plt.tight_layout()
    plt.show()


    # -----------------------------
    # 6) Pre-switch eating-bout variability (KM) + quantiles using global fits
    # -----------------------------
    def first_sh_time(y):
        idx = np.where(y==1)[0]
        if len(idx)==0: return len(y), 0
        return int(idx[0]), 1

    def km_survival(times, events, Lmax):
        S = [1.0]
        times = np.asarray(times); events = np.asarray(events).astype(int)
        def n_at(ell): return np.sum(times >= ell)
        def d_at(ell): return np.sum((times == ell) & (events==1))
        prod = 1.0
        for ell in range(0, Lmax):
            n = n_at(ell)
            if n == 0:
                S.append(S[-1]); continue
            d = d_at(ell)
            prod *= (1.0 - d/max(n,1))
            S.append(prod)
        ells = np.arange(0, Lmax+1)
        return ells, np.array(S)

    # empirical KM
    times_emp, events_emp = [], []
    for d in data:
        t_ev, ev = first_sh_time(np.asarray(d["y"]).astype(int))
        times_emp.append(t_ev); events_emp.append(ev)
    ell_emp, S_emp = km_survival(times_emp, events_emp, Lmax=max_T)

    # predictive KM bands using global models
    def predictive_km_band_global(model_name, n_rep=120):
        all_S = []
        for rep in range(n_rep):
            times, events = [], []
            for i, d in enumerate(data):
                T = len(d["y"])
                if model_name == "IID":
                    y_sim = sim_from_IID_param(iid_global["p"], T)
                elif model_name == "Trend":
                    y_sim = sim_from_Trend_param((trend_global["a"], trend_global["dlt"]), T)
                elif model_name == "CP":
                    y_sim = sim_from_CP_param(cp_global["tau"], cp_global["p0"], cp_global["p1"], T)
                elif model_name == "CP_per_trial":
                    y_sim = sim_from_CP_param(cp_mixed["taus"], cp_mixed["p0"], cp_mixed["p1"], T, idx=i)
                elif model_name == "HMM":
                    y_sim = sim_from_HMM_param(hmm_global, T)
                elif model_name == "HSMM":
                    y_sim = sim_from_HSMM_param(hsmm_global, T)
                elif model_name == "HMM_bivariate":
                    y_sim = sim_from_HMM_biv_param(hmm_biv_global, T)
                t_ev, ev = first_sh_time(y_sim)
                times.append(t_ev); events.append(ev)
            ell, S = km_survival(times, events, Lmax=max_T)
            all_S.append(S)
        arr = np.stack(all_S, axis=0)
        return ell, arr.mean(axis=0), np.percentile(arr,2.5,axis=0), np.percentile(arr,97.5,axis=0)

    bands_km_global = {}
    for m in models_to_run:
        bands_km_global[m] = predictive_km_band_global(m, n_rep=120)


    plt.figure(figsize=(10, 6))

    # Empirical survival curve (Kaplan-Meier style)
    plt.step(
        ell_emp,
        S_emp,
        where="post",
        label="Real data",
        linewidth=3,
        color="#730071"
    )

    # Model predictions + uncertainty envelopes
    for m in models_to_run:
        ell, mean, lo, hi = bands_km_global[m]
        color = palette[m]
        
        # Mean curve
        plt.plot(ell, mean, label=f"{m}", color=color, linewidth=3)
        
        # Uncertainty band
        plt.fill_between(ell, lo, hi, alpha=0.2, color=color, linewidth=0)

    plt.xlabel("Number of eating before switching", fontsize=14)
    plt.ylabel("Probability of post-switch scatter-hoarding", fontsize=14)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="best", fontsize=14, frameon=False)
    plt.tight_layout()
    plt.show()

    # KM quantiles table
    def km_quantiles(ell, S, qs=(0.25,0.5,0.75)):
        out = {}
        for q in qs:
            target = 1 - q
            idx = np.where(S <= target)[0]
            out[q] = float(ell[idx[0]]) if len(idx) else float(ell[-1])
        return out

    emp_q = km_quantiles(ell_emp, S_emp, qs=(0.25,0.5,0.75))

    def predictive_km_quantile_band_global(model_name, n_rep=150):
        q25s, meds, q75s = [], [], []
        for rep in range(n_rep):
            times, events = [], []
            for i, d in enumerate(data):
                T = len(d["y"])
                if model_name == "IID":
                    y_sim = sim_from_IID_param(iid_global["p"], T)
                elif model_name == "Trend":
                    y_sim = sim_from_Trend_param((trend_global["a"], trend_global["dlt"]), T)
                elif model_name == "CP":
                    y_sim = sim_from_CP_param(cp_global["tau"], cp_global["p0"], cp_global["p1"], T)
                elif model_name == "CP_per_trial":
                    y_sim = sim_from_CP_param(cp_mixed["taus"], cp_mixed["p0"], cp_mixed["p1"], T, idx=i)
                elif model_name == "HMM":
                    y_sim = sim_from_HMM_param(hmm_global, T)
                elif model_name == "HSMM":
                    y_sim = sim_from_HSMM_param(hsmm_global, T)
                elif model_name == "HMM_bivariate":
                    y_sim = sim_from_HMM_biv_param(hmm_biv_global, T)
                t_ev, ev = first_sh_time(y_sim)
                times.append(t_ev); events.append(ev)
            ell, S = km_survival(times, events, Lmax=max_T)
            qs = km_quantiles(ell, S, qs=(0.25,0.5,0.75))
            q25s.append(qs[0.25]); meds.append(qs[0.5]); q75s.append(qs[0.75])
        def stats(arr): return float(np.mean(arr)), float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))
        return {"Q25": stats(q25s), "Med": stats(meds), "Q75": stats(q75s)}

    rows = [{"Model":"Empirical",
             "KM_Q25": emp_q[0.25],
             "KM_Median": emp_q[0.5],
             "KM_Q75": emp_q[0.75]}]
    for m in models_to_run:
        s = predictive_km_quantile_band_global(m, n_rep=150)
        q25m,q25l,q25h = s["Q25"]; medm,medl,medh = s["Med"]; q75m,q75l,q75h = s["Q75"]
        rows.append({"Model": f"{m} (global pred)",
                     "KM_Q25": f"{q25m:.1f} [{q25l:.1f},{q25h:.1f}]",
                     "KM_Median": f"{medm:.1f} [{medl:.1f},{medh:.1f}]",
                     "KM_Q75": f"{q75m:.1f} [{q75l:.1f},{q75h:.1f}]"})
    summ_df = pd.DataFrame(rows)
    print("KM quantiles (empirical vs global predictive intervals):")
    print(summ_df.to_string(index=False))

    global_params = {
        'IID': iid_global, 
        'Logistic': trend_global, 
        'Change-point': cp_global, 
        'HMM': hmm_global,
        'HSMM': hsmm_global
    }

    return global_aic_table, global_params

# =====================================================================
# --- NEW INTEGRATOR CHANGE-POINT MODELS ---
# =====================================================================

def map_a_to_tau(a_i, b, T, min_seg=3):
    if a_i >= -b * (min_seg - 1):
        return min_seg
    for k in range(min_seg + 1, T):
        if a_i >= -b * (k - 1):
            return k
    return T

# 1. Deterministic Integrator Model


def deterministic_integrator_cp_emit(data, params, rng=None):
    if rng is None: rng = np.random.default_rng()
    mu_a = params["mu_a"]
    sigma_a = params["sigma_a"]
    b = params["b"]
    p0 = params["p0"]
    p1 = params["p1"]
    min_seg = 3
    
    out_y, out_p, out_tau = [], [], []
    for d in data:
        T = len(d["y"])
        a_i = rng.normal(mu_a, sigma_a)
        tau_i = map_a_to_tau(a_i, b, T, min_seg)
        
        p = np.concatenate([
            np.repeat(p0, tau_i),
            np.repeat(p1, max(0, T - tau_i))
        ])
        y_sim = (rng.random(T) < p).astype(int)
        
        out_y.append(y_sim)
        out_p.append(p)
        out_tau.append(tau_i)
    return out_y, out_p, out_tau

# 2. Stochastic Integrator Model



def stochastic_integrator_cp_emit(data, params, rng=None):
    from scipy.special import expit
    if rng is None: rng = np.random.default_rng()
    b = params["b"]
    p0 = params["p0"]
    p1 = params["p1"]
    a_list = params["a_list"]
    min_seg = 3
    
    out_y, out_p, out_tau = [], [], []
    for i, d in enumerate(data):
        T = len(d["y"])
        a_i = a_list[i]
        
        y_sim = np.zeros(T, dtype=int)
        p_sim = np.zeros(T, dtype=float)
        
        tau_sim = T
        in_state_1 = False
        
        for t in range(T):
            if in_state_1:
                p_sim[t] = p1
                y_sim[t] = 1 if rng.random() < p1 else 0
            else:
                p_sim[t] = p0
                y_sim[t] = 1 if rng.random() < p0 else 0
                
                if t >= min_seg - 1:
                    C_next = np.sum(1 - y_sim[:t+1])
                    lambda_next = expit(a_i + b * C_next)
                    if rng.random() < lambda_next:
                        in_state_1 = True
                        tau_sim = t + 1
                        
        out_y.append(y_sim)
        out_p.append(p_sim)
        out_tau.append(tau_sim)
    return out_y, out_p, out_tau

# 3. New Cross-Validation comparison including integrator models

# 4. Full Pipeline v2 including the new models

# --- EXECUTION ---
# run_full_pipeline_v2(structured_data) # Uncomment to run




def fit_bivariate_hmm(path_to_csv, max_iter=500, tol=1e-9):
    """
    Fits a 2-state Bivariate HMM emitting outcome (y) and rotation (z).
    Returns a dictionary containing the 6 fitted probabilities.
    """
    # 1. Load and filter data for non-missing scatter-hoard and rotation trials
    SH_data = pd.read_csv(path_to_csv)
    SH_data['trial'] = SH_data['Date'] + SH_data['ID']
    
    structured_data = []
    for t in np.unique(SH_data['trial']):
        sub = SH_data[SH_data['trial'] == t]
        mask = (~sub['scatter-hoard'].isna()) & (~sub['rotation'].isna())
        if mask.sum() > 0:
            y = sub.loc[mask, 'scatter-hoard'].values.astype(int)
            z = sub.loc[mask, 'rotation'].values.astype(int)
            structured_data.append({"y": y, "z": z})
            
    # 2. Forward-Backward algorithm helper
    def forward_backward(y, z, qy, qz, ry, rz, pi, eps):
        T = len(y)
        py_eat = np.where(y == 1, qy, 1.0 - qy)
        pz_eat = np.where(z == 1, qz, 1.0 - qz)
        p_eat = py_eat * pz_eat
        py_sh = np.where(y == 1, ry, 1.0 - ry)
        pz_sh = np.where(z == 1, rz, 1.0 - rz)
        p_sh = py_sh * pz_sh
        T00 = (1.0 - pi); T01 = pi
        T10 = eps;        T11 = (1.0 - eps)
        alpha = np.zeros((T, 2))
        alpha[0, 0] = np.log(p_eat[0] + 1e-12)
        alpha[0, 1] = -1e12 + np.log(p_sh[0] + 1e-12)
        for t in range(1, T):
            a0 = alpha[t-1, 0] + np.log(T00 + 1e-12)
            a1 = alpha[t-1, 1] + np.log(T10 + 1e-12)
            alpha[t, 0] = np.log(p_eat[t] + 1e-12) + np.logaddexp(a0, a1)
            b0 = alpha[t-1, 0] + np.log(T01 + 1e-12)
            b1 = alpha[t-1, 1] + np.log(T11 + 1e-12)
            alpha[t, 1] = np.log(p_sh[t] + 1e-12) + np.logaddexp(b0, b1)
        ll = float(np.logaddexp(alpha[-1, 0], alpha[-1, 1]))
        beta = np.zeros((T, 2))
        for t in range(T-2, -1, -1):
            term0 = np.log(T00 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
            term1 = np.log(T01 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
            beta[t, 0] = np.logaddexp(term0, term1)
            term0 = np.log(T10 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
            term1 = np.log(T11 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
            beta[t, 1] = np.logaddexp(term0, term1)
        gamma_log = alpha + beta
        m = np.max(gamma_log, axis=1, keepdims=True)
        gamma = np.exp(gamma_log - m)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((T-1, 2, 2))
        for t in range(T-1):
            M = np.zeros((2, 2))
            M[0, 0] = alpha[t, 0] + np.log(T00 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
            M[0, 1] = alpha[t, 0] + np.log(T01 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
            M[1, 0] = alpha[t, 1] + np.log(T10 + 1e-12) + np.log(p_eat[t+1] + 1e-12) + beta[t+1, 0]
            M[1, 1] = alpha[t, 1] + np.log(T11 + 1e-12) + np.log(p_sh[t+1] + 1e-12) + beta[t+1, 1]
            m2 = np.max(M); Z = np.exp(M - m2).sum()
            xi[t, :, :] = np.exp(M - m2) / Z
        return ll, gamma, xi
    # 3. Expectation-Maximization loop
    qy, qz, ry, rz, pi, eps = 0.01, 0.1, 0.9, 0.8, 0.05, 0.01
    prev_ll = -np.inf
    
    for it in range(max_iter):
        num_qy = den_qy = num_qz = den_qz = 0.0
        num_ry = den_ry = num_rz = den_rz = 0.0
        num_pi = den_pi = num_eps = den_eps = 0.0
        total_ll = 0.0
        for d in structured_data:
            y = d["y"]; z = d["z"]
            ll_i, gamma_i, xi_i = forward_backward(y, z, qy, qz, ry, rz, pi, eps)
            total_ll += ll_i
            ge, gs = gamma_i[:, 0], gamma_i[:, 1]
            num_qy += (ge * y).sum(); den_qy += ge.sum()
            num_qz += (ge * z).sum(); den_qz += ge.sum()
            num_ry += (gs * y).sum(); den_ry += gs.sum()
            num_rz += (gs * z).sum(); den_rz += gs.sum()
            if len(xi_i) > 0:
                num_pi += xi_i[:, 0, 1].sum(); den_pi += gamma_i[:-1, 0].sum()
                num_eps += xi_i[:, 1, 0].sum(); den_eps += gamma_i[:-1, 1].sum()
        qy_new = float(np.clip(num_qy / max(den_qy, 1e-12), 1e-6, 1.0 - 1e-6))
        qz_new = float(np.clip(num_qz / max(den_qz, 1e-12), 1e-6, 1.0 - 1e-6))
        ry_new = float(np.clip(num_ry / max(den_ry, 1e-12), 1e-6, 1.0 - 1e-6))
        rz_new = float(np.clip(num_rz / max(den_rz, 1e-12), 1e-6, 1.0 - 1e-6))
        pi_new = float(np.clip(num_pi / max(den_pi, 1e-12), 1e-6, 1.0 - 1e-6))
        eps_new = float(np.clip(num_eps / max(den_eps, 1e-12), 1e-6, 1.0 - 1e-6))
        if abs(total_ll - prev_ll) < tol:
            break
        qy, qz, ry, rz, pi, eps = qy_new, qz_new, ry_new, rz_new, pi_new, eps_new
        prev_ll = total_ll
    # Calculate hypothetical AIC if model was only fitted to outcomes of decisions (y)
    def forward_y_only(y, qy, ry, pi, eps):
        T = len(y)
        py_eat = np.where(y == 1, qy, 1.0 - qy)
        py_sh = np.where(y == 1, ry, 1.0 - ry)
        T00 = (1.0 - pi); T01 = pi
        T10 = eps;        T11 = (1.0 - eps)
        alpha = np.zeros((T, 2))
        alpha[0, 0] = np.log(py_eat[0] + 1e-12)
        alpha[0, 1] = -1e12 + np.log(py_sh[0] + 1e-12)
        for t in range(1, T):
            a0 = alpha[t-1, 0] + np.log(T00 + 1e-12)
            a1 = alpha[t-1, 1] + np.log(T10 + 1e-12)
            alpha[t, 0] = np.log(py_eat[t] + 1e-12) + np.logaddexp(a0, a1)
            b0 = alpha[t-1, 0] + np.log(T01 + 1e-12)
            b1 = alpha[t-1, 1] + np.log(T11 + 1e-12)
            alpha[t, 1] = np.log(py_sh[t] + 1e-12) + np.logaddexp(b0, b1)
        return float(np.logaddexp(alpha[-1, 0], alpha[-1, 1]))

    hypothetical_ll_y = 0.0
    for d in structured_data:
        hypothetical_ll_y += forward_y_only(d["y"], qy, ry, pi, eps)
        
    hypothetical_aic_y = 2 * 4 - 2 * hypothetical_ll_y

    # Store all 6 fitted probabilities in a dictionary with both key styles
    fitted_probabilities = {
        "qy": qy, "q_y": qy,
        "qz": qz, "q_z": qz,
        "ry": ry, "r_y": ry,
        "rz": rz, "r_z": rz,
        "pi": pi, "eps": eps,
        "log_likelihood": prev_ll,
        "hypothetical_aic_y": hypothetical_aic_y
    }
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 6))
    plt.bar(["HMM (Outcomes only)"], [hypothetical_aic_y], color="#72A1E5", width=0.4)
    plt.ylabel("AIC", fontsize=14)
    plt.title("Hypothetical AIC", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(r"figures/hypothetical_hmm_aic.svg", format="svg", bbox_inches="tight")
    plt.show()

    return fitted_probabilities


def hmm_bivariate_emit(data, params, force_emit=True, max_tries=5000, rng=None):
    """Simulate trajectories for the 2-state bivariate HMM emitting outcomes (y) and rotations (z)."""
    if rng is None: rng = np.random.default_rng()
    qy = params.get("qy", params.get("q_y"))
    qz = params.get("qz", params.get("q_z"))
    ry = params.get("ry", params.get("r_y"))
    rz = params.get("rz", params.get("r_z"))
    pi = params["pi"]
    eps = params["eps"]
    out_y, out_z, out_s = [], [], []
    for d in data:
        n = len(d["y"])
        for _ in range(max_tries):
            s = np.zeros(n, dtype=int)
            for t in range(1, n):
                if s[t - 1] == 0:
                    s[t] = 1 if rng.random() < pi else 0
                else:
                    s[t] = 0 if rng.random() < eps else 1
            y_sim = np.zeros(n, dtype=int)
            z_sim = np.zeros(n, dtype=int)
            for t in range(n):
                y_sim[t] = 1 if rng.random() < (ry if s[t] == 1 else qy) else 0
                z_sim[t] = 1 if rng.random() < (rz if s[t] == 1 else qz) else 0
            if not force_emit or np.any(y_sim == 1):
                out_y.append(y_sim)
                out_z.append(z_sim)
                out_s.append(s)
                break
        else:
            raise RuntimeError(f"No emission=1 after {max_tries} tries for sequence length {n}")
    return out_y, out_z, out_s


def simulate_bivariate_hmm_predictions(arg1, arg2, seed=None):
    """Wrapper matching the structure of run_models_with_global_params that takes bivariate HMM parameters and simulates trajectories for outcomes (y), rotations (z), and hidden states (s)."""
    if isinstance(arg1, dict) and isinstance(arg2, (list, tuple)):
        bivariate_params = arg1
        data = arg2
    else:
        data = arg1
        bivariate_params = arg2
        
    rng = np.random.default_rng(seed)
    outputs = {}
    outputs["HMM_bivariate_y"], outputs["HMM_bivariate_z"], outputs["HMM_bivariate_s"] = hmm_bivariate_emit(data, bivariate_params, force_emit=True, rng=rng)
    return outputs


def hmm_emit(data, params, force_emit=False, max_tries=1000):
    """
    Simulate 2-state absorbing HMM sequences.

    Parameters
    ----------
    data : list of dicts
        Each dict must contain key "y" whose length sets the sequence length.
    params : dict
        Must contain "q", "r", "pi", "eps".
    force_emit : bool, optional (default=True)
        If True, re-simulates a sequence until at least one 1 is emitted.
    max_tries : int, optional (default=1000)
        Maximum number of retries per sequence to avoid infinite loops.

    Returns
    -------
    out_y : list of np.ndarray
        Simulated emissions (0/1).
    out_p : list of np.ndarray
        Emission probabilities for each timepoint.
    out_s : list of np.ndarray
        Hidden states (0 or 1).
    """

    q, r, pi, eps = params["q"], params["r"], params["pi"], params["eps"]
    out_y, out_p, out_s = [], [], []

    for d in data:
        n = len(d["y"])

        for _ in range(max_tries):
            # --- simulate states ---
            s = np.zeros(n, dtype=int)
            for t in range(1, n):
                if s[t - 1] == 0:
                    s[t] = np.random.rand() < pi
                else:
                    s[t] = 0 if np.random.rand() < eps else 1

            # --- emissions ---
            y_sim = np.zeros(n, dtype=int)
            for t in range(n):
                y_sim[t] = np.random.rand() < (q if s[t] == 0 else r)

            if not force_emit or np.any(y_sim == 1):
                p_emit = np.array([q if state == 0 else r for state in s])
                out_y.append(y_sim)
                out_s.append(s)
                out_p.append(p_emit)
                break
        else:
            raise RuntimeError(f"No emission=1 after {max_tries} tries for sequence of length {n}")

    return out_y, out_p, out_s
