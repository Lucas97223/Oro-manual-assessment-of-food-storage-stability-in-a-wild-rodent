# Cleaned and Reorganized modeling.py
# Contains only the trial-alignment and bar-plotting functions called in the notebook.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

def find_first_transition1(trial):
    """Return the index of the first 0→1 transition in a single trial, or None."""
    for i in range(1, len(trial)):
        if trial[i-1] == 0 and trial[i] == 1:
            return i
    return None

def align_trials1(trials):
    """
    Align trials by their first 0→1 transition.
    
    Parameters
    ----------
    trials : list of np.ndarray
        Behavioral sequences (e.g. eat=0, scatter-hoard=1).
        
    Returns
    -------
    aligned_matrix : np.ndarray
        Aligned 2D array of sequences.
    max_pre : int
        The zero alignment index (number of elements before the transition).
    """
    transition_indices = [find_first_transition1(t) for t in trials]
    print(transition_indices)
    transition_indices = [idx if idx is not None else 0 for idx in transition_indices]
    print(transition_indices)

    max_pre = max(transition_indices)
    max_post = max(len(t) - idx for t, idx in zip(trials, transition_indices))

    aligned_length = max_pre + max_post
    aligned_matrix = np.full((len(trials), aligned_length), np.nan)

    for i, (trial, idx) in enumerate(zip(trials, transition_indices)):
        start_pos = max_pre - idx
        aligned_matrix[i, start_pos:start_pos+len(trial)] = trial

    return aligned_matrix, max_pre

def find_first_transition2(trial):
    """Return the index of the first 0→1 transition, or None if no transition."""
    for i in range(1, len(trial)):
        if trial[i-1] == 0 and trial[i] == 1:
            return i
    return None

def align_trials2(trials, values_to_insert):
    """
    Align trials by the first transition (0→1) and fill the aligned matrix
    with values from `values_to_insert` that have the same structure.

    Parameters
    ----------
    trials : list of np.ndarray
        Binary arrays used to detect transitions.
    values_to_insert : list of np.ndarray
        Arrays of equal lengths to `trials`, used to fill the aligned matrix.

    Returns
    -------
    aligned_matrix : np.ndarray
        Aligned 2D array filled with values from `values_to_insert`.
    max_pre : int
        Number of samples before the transition in the alignment.
    """
    transition_indices = [find_first_transition2(t) for t in trials]
    transition_indices = [idx if idx is not None else 0 for idx in transition_indices]

    # compute padding sizes
    max_pre = max(transition_indices)
    max_post = max(len(t) - idx for t, idx in zip(trials, transition_indices))
    aligned_length = max_pre + max_post

    # initialize
    aligned_matrix = np.full((len(trials), aligned_length), np.nan)

    # fill with corresponding values
    for i, (vals, idx) in enumerate(zip(values_to_insert, transition_indices)):
        start_pos = max_pre - idx
        aligned_matrix[i, start_pos:start_pos+len(vals)] = vals

    return aligned_matrix, max_pre

def plot_aligned_bars(aligned_matrix, zero_idx, 
                       plot_probability=True, 
                       fit_sigmoid=False,
                       plot_pretransition_zeros=False, rotation = False,
                      metric = 'outcome', rotation_matrix=None):
    """
    Visualizes trial-aligned behavioral states using horizontal bars,
    overlays a probability trace, and optionally shows:
      - a sigmoid fit,
      - a trace of the proportion of 0's before the first transition.
    """
    n_trials, n_time = aligned_matrix.shape
    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = {0: '#1A897B', 1: '#FFB722'}
    if rotation == True:
        colors = {0: '#36454F', 1: '#8075C0'}
        prob_before = np.nanmean(aligned_matrix[:, :zero_idx])
        prob_after = np.nanmean(aligned_matrix[:, zero_idx:])
        print(f"Probability of rotation before 0: {prob_before:.4f}")
        print(f"Probability of rotation after 0: {prob_after:.4f}")

    # --- Draw horizontal bars for each trial ---
    y_vals_0, xmin_0, xmax_0 = [], [], []
    y_vals_1, xmin_1, xmax_1 = [], [], []
    
    for i in range(n_trials):
        y = 1 - (i / (n_trials - 1)) if n_trials > 1 else 0.5
        for t in range(n_time):
            val = aligned_matrix[i, t]
            if np.isnan(val):
                continue
            if val == 0:
                y_vals_0.append(y)
                xmin_0.append(t - zero_idx - 0.5)
                xmax_0.append(t - zero_idx + 0.5)
            elif val == 1:
                y_vals_1.append(y)
                xmin_1.append(t - zero_idx - 0.5)
                xmax_1.append(t - zero_idx + 0.5)

    if y_vals_0:
        ax1.hlines(y=y_vals_0, xmin=xmin_0, xmax=xmax_0, color=colors[0], linewidth=22, alpha=1)
    if y_vals_1:
        ax1.hlines(y=y_vals_1, xmin=xmin_1, xmax=xmax_1, color=colors[1], linewidth=22, alpha=1)

    # Vertical alignment line
    ax1.axvline(-0.5, color='red', linestyle='--', linewidth=3, label='First 0→1 transition')

    # --- Axis and label formatting ---
    ax1.set_ylim(-.05, 1.05)
    ax1.set_xlim(-zero_idx - 5, n_time - zero_idx + 5)
    ax1.set_xlabel('Peanut order relative to first Scatter-hoarding', fontsize=24)
    ax1.set_ylabel('Trials', fontsize=24)
    ax1.set_yticks([])

    # Remove top/right spines, thicken bottom/left
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_linewidth(2.5)

    # Tick label size and line width
    ax1.tick_params(axis='x', labelsize=24, width=2)
    ax1.tick_params(axis='y', labelsize=24, width=2)

    # Legend
    if rotation == False:
        custom_legend = [
            Patch(facecolor=colors[0], label='Eat'),
            Patch(facecolor=colors[1], label='Scatter-hoard')
        ]
    elif rotation == True:
        custom_legend = [
            Patch(facecolor=colors[0], label='No Rotation'),
            Patch(facecolor=colors[1], label='Rotation')
        ]
        
    # --- Probability trace ---
    if plot_probability or plot_pretransition_zeros:
        x_vals = np.arange(n_time) - zero_idx
        ax2 = ax1.twinx()
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel('Probability', fontsize=14)
        ax2.tick_params(axis='y', labelsize=14, width=2)
        for spine in ['top', 'right']:
            ax2.spines[spine].set_linewidth(2.5)

    if plot_probability:
        prob_1 = []
        for t in range(n_time):
            col = aligned_matrix[:, t]
            valid = ~np.isnan(col)
            prob_1.append(np.nan if valid.sum() == 0 else np.sum(col[valid] == 1) / valid.sum())
        prob_1 = np.array(prob_1)
        prob_line, = ax2.plot(x_vals, prob_1, color='black', linewidth=4, label='P(Scatter Hoard)')
        custom_legend.append(prob_line)

        # --- Sigmoid fitting ---
        if fit_sigmoid:
            def sigmoid(x, L, x0, k, b):
                return L / (1 + np.exp(-k * (x - x0))) + b

            mask = ~np.isnan(prob_1)
            try:
                popt, _ = curve_fit(sigmoid, x_vals[mask], prob_1[mask],
                                    p0=[1, 0, 1, 0], maxfev=5000)
                y_fit = sigmoid(x_vals, *popt)
                sig_line, = ax2.plot(x_vals, y_fit, '-', color='green', linewidth=3, label='Sigmoid fit')
                custom_legend.append(sig_line)
            except RuntimeError:
                print("⚠️ Sigmoid fit failed.")

    # --- Pre-transition proportion of 0s ---
    if plot_pretransition_zeros:
        prop_zeros = np.zeros(n_time)
        for t in range(n_time):
            count_zeros = 0
            for i in range(n_trials):
                row = aligned_matrix[i, :]
                if np.all(np.isnan(row)):
                    continue
                ones = np.where(row == 1)[0]
                if len(ones) > 0 and t >= ones[0]:
                    continue  # after first 1, stop counting
                if not np.isnan(row[t]) and row[t] == 0:
                    count_zeros += 1
            prop_zeros[t] = count_zeros / n_trials  # include NaN trials in denominator

        zero_line, = ax2.plot(x_vals, prop_zeros, color='blue', linewidth=3, linestyle='-')
        custom_legend.append(zero_line)

    ax1.legend(handles=custom_legend, loc='lower right', fontsize=20, frameon=False)
    plt.tight_layout()

    save_path = r'figures/SH_mode_' + metric + '.svg'
    print(save_path)
    plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)
    
    plt.show()

    if rotation:
        from scipy.stats import ttest_ind
        vals_before = aligned_matrix[:, :zero_idx].flatten()
        vals_before = vals_before[~np.isnan(vals_before)]
        vals_after = aligned_matrix[:, zero_idx:].flatten()
        vals_after = vals_after[~np.isnan(vals_after)]
        
        t_stat, p_val1 = ttest_ind(vals_before, vals_after, equal_var=False) if len(vals_before) > 0 and len(vals_after) > 0 else (0, 1)
        
        plt.figure(figsize=(5, 6))
        plt.bar(['Before', 'After'], [prob_before, prob_after], color=['#36454F', '#8075C0'], width=0.5)
        plt.ylabel('Probability of Rotation', fontsize=14)
        plt.title('Rotation Before vs After First SH', fontsize=14)
        
        y_max = max(prob_before, prob_after) if not np.isnan(prob_before) and not np.isnan(prob_after) else 0.5
        h = max(y_max * 0.05, 0.02)
        plt.plot([0, 0, 1, 1], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], lw=1.5, c='k')
        
        p_str = f"p = {p_val1:.4g}"
        if p_val1 < 0.001: p_str += " ***"
        elif p_val1 < 0.01: p_str += " **"
        elif p_val1 < 0.05: p_str += " *"
        else: p_str += " (ns)"
        
        plt.text(0.5, y_max + 2.5*h, p_str, ha='center', va='bottom', color='k', fontsize=12)
        plt.ylim(0, max(1.15, y_max + 6*h))
        
        ax_rot = plt.gca()
        ax_rot.spines['top'].set_visible(False)
        ax_rot.spines['right'].set_visible(False)
        ax_rot.spines['left'].set_linewidth(2.0)
        ax_rot.spines['bottom'].set_linewidth(2.0)
        plt.xticks(fontsize=14)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
        plt.ylim(0, 1)
        plt.tight_layout()
        rot_save_path = r'figures/SH_mode_' + metric + '_rotation_before_after.svg'
        plt.savefig(rot_save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

    if rotation_matrix is not None:
        after_outcomes = aligned_matrix[:, zero_idx:]
        after_rotations = rotation_matrix[:, zero_idx:]
        
        valid = ~np.isnan(after_outcomes) & ~np.isnan(after_rotations)
        
        outcomes_valid = after_outcomes[valid]
        rotations_valid = after_rotations[valid]
        
        eat_mask = (outcomes_valid == 0)
        sh_mask = (outcomes_valid == 1)
        
        prob_rot_eat = np.mean(rotations_valid[eat_mask]) if np.sum(eat_mask) > 0 else 0
        prob_rot_sh = np.mean(rotations_valid[sh_mask]) if np.sum(sh_mask) > 0 else 0
        
        print(f"Probability of rotation given Eat (after 1st SH): {prob_rot_eat:.4f}")
        print(f"Probability of rotation given Scatter-hoard (after 1st SH): {prob_rot_sh:.4f}")
        
        from scipy.stats import ttest_ind
        t_stat2, p_val2 = ttest_ind(rotations_valid[eat_mask], rotations_valid[sh_mask], equal_var=False) if np.sum(eat_mask) > 0 and np.sum(sh_mask) > 0 else (0, 1)
        
        plt.figure(figsize=(5, 6))
        plt.bar(['Eat', 'Scatter-hoard'], [prob_rot_eat, prob_rot_sh], color=['#1A897B', '#FFB722'], width=0.5)
        plt.ylabel('Probability of Rotation', fontsize=14)
        plt.title('Rotation given Outcome (After 1st SH)', fontsize=14)
        
        y_max2 = max(prob_rot_eat, prob_rot_sh) if not np.isnan(prob_rot_eat) and not np.isnan(prob_rot_sh) else 0.5
        h2 = max(y_max2 * 0.05, 0.02)
        plt.plot([0, 0, 1, 1], [y_max2 + h2, y_max2 + 2*h2, y_max2 + 2*h2, y_max2 + h2], lw=1.5, c='k')
        
        p_str2 = f"p = {p_val2:.4g}"
        if p_val2 < 0.001: p_str2 += " ***"
        elif p_val2 < 0.01: p_str2 += " **"
        elif p_val2 < 0.05: p_str2 += " *"
        else: p_str2 += " (ns)"
        
        plt.text(0.5, y_max2 + 2.5*h2, p_str2, ha='center', va='bottom', color='k', fontsize=12)
        plt.ylim(0, max(1.15, y_max2 + 6*h2))
        
        ax_cond = plt.gca()
        ax_cond.spines['top'].set_visible(False)
        ax_cond.spines['right'].set_visible(False)
        ax_cond.spines['left'].set_linewidth(2.0)
        ax_cond.spines['bottom'].set_linewidth(2.0)
        plt.xticks(fontsize=14)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
        plt.ylim(0, 1)
        plt.tight_layout()
        cond_save_path = r'figures/SH_mode_' + metric + '_rotation_given_outcome.svg'
        plt.savefig(cond_save_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
