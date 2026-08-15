# Coherently organized Agouti behavioral statistics and ethogram analysis toolkit.

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro, levene, ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, linregress, kruskal, norm, t
from scipy.optimize import curve_fit
from statsmodels.stats.proportion import proportions_ztest
from collections import defaultdict
from itertools import groupby, combinations
import os
import networkx as nx
from statsmodels.formula.api import mixedlm
from scipy.stats import kstest
from pandas.api.types import CategoricalDtype
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from statsmodels.formula.api import ols
from PIL import Image
from scipy.ndimage import binary_erosion
import statsmodels.formula.api as smf
import itertools
from scipy.stats import fisher_exact, chi2_contingency
from itertools import chain
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# --- ENFORCE GLOBAL CLEAN STYLE ---
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_style("ticks", {"axes.edgecolor": "black"})



    
def plot_sh_mode(SH_data, sh_or_rotation, test = 'mann-whitney', legend_xy=(0.7, 0.5)):
    # Organize into dictionary
    dict_sh = {
        animal: SH_data.loc[SH_data['ID'] == animal, 'scatter-hoard'].tolist()
        for animal in SH_data['ID'].unique()
    }

    if sh_or_rotation == 'rotation':
        dict_rotate = {
        animal: SH_data.loc[SH_data['ID'] == animal, 'rotation'].tolist()
        for animal in SH_data['ID'].unique()
    }

    number_animals = len(dict_sh.keys())
    
    # Pad to equal length
    max_len = max(len(v) for v in dict_sh.values())
    df_sh = pd.DataFrame({k: pd.Series(v) for k, v in dict_sh.items()}, index=range(max_len))
    if sh_or_rotation == 'rotation':
        df_rotate = pd.DataFrame({k: pd.Series(v) for k, v in dict_rotate.items()}, index=range(max_len))

    # Align each series to its first 1.0
    aligned_sh = {}
    aligned_rotate = {}
    for animal, series in df_sh.items():
        first_1_idx = series[series == 1.0].first_valid_index()
        if first_1_idx is not None:
            shifted_index = np.arange(len(series)) - first_1_idx
            aligned_sh[animal] = pd.Series(series.values, index=shifted_index)
            if sh_or_rotation == 'rotation':
                aligned_rotate[animal] = pd.Series(df_rotate[animal].values, index=shifted_index)

    # Combine into a single DataFrame
    if sh_or_rotation == 'scatter-hoard':
        aligned_df = pd.DataFrame(aligned_sh).sort_index()
    elif sh_or_rotation == 'rotation':
        aligned_df = pd.DataFrame(aligned_rotate).sort_index()
    
    # Compute probability of 1
    mean_prob = aligned_df.mean(axis=1)

    if sh_or_rotation == 'scatter-hoard':
        lab_not = 'Eat'
        col_not = '#1A897B'
        lab_yes = 'Scatter-Hoard'
        col_yes = '#FFB722'
        
    elif sh_or_rotation == 'rotation':
        lab_not = 'No rotation'
        col_not = '#36454F'
        lab_yes = 'Rotation'
        col_yes = '#8075C0'
        
    # Plot
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Left axis for individual traces
    y_offset = 0.99
    spacing = -1 / number_animals

    for i, (animal, series) in enumerate(aligned_df.items()):
        for t, val in series.items():
            if not np.isnan(val):
                color = col_yes if val == 1 else col_not
                ax1.hlines(
                    y=y_offset + i * spacing,
                    xmin=t - 0.5, xmax=t + 0.5,
                    color=color,
                    linewidth=15,
                    alpha=1
                )
    
    ax1.set_ylabel('Individuals', fontsize=16)
    ax1.set_yticks([])  # Hide y-ticks for individuals
    ax1.set_xlabel('Epoch relative to First Scatter Hoarding', fontsize=16)

    # Right axis for probability
    #ax2 = ax1.twinx()
    #ax2.plot(mean_prob.index, mean_prob.values, label='Mean Probability', color='black', linewidth=3)
    #ax2.set_ylabel('Probability of Scatter Hoarding', fontsize=16)
    #ax2.set_ylim(-0.05, 1.05)

    # Vertical line at time zero
    ax1.axvline(0, color='red', linestyle='--', linewidth=3, label='Time of first SH')
        
    custom_legend = [
        Patch(facecolor=col_not, label=lab_not),
        Patch(facecolor=col_yes, label=lab_yes),
        Line2D([0], [0], color='black', lw=3, label='Mean Probability'),
        Line2D([0], [0], color='red', lw=3, linestyle='--', label='Time of first SH'),
    ]
    #ax2.legend(handles=custom_legend, loc='upper left', bbox_to_anchor=legend_xy, fontsize=13)

    ax1.set_title('Probability of Scatter-Hoarding Over Time (Centered around first SH)', fontsize=16)
    ax1.tick_params(axis='x', labelsize=12)
    #ax2.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.show()

    if test == 'wilcoxon':
        # Wilcoxon test: compare pre vs post SH per individual
        before_vals = []
        after_vals = []
        for animal, series in aligned_df.items():
            before = series[series.index < 0].mean()
            after = series[series.index > 0].mean()
            if not np.isnan(before) and not np.isnan(after):
                before_vals.append(before)
                after_vals.append(after)
    
        if len(before_vals) > 0:
            stat, p = wilcoxon(after_vals, before_vals)
            print(f"Wilcoxon signed-rank test: stat = {stat:.3f}, p = {p:.4g}")
        else:
            print("Not enough data for Wilcoxon test.")

    elif test == 'mann-whitney':
        # Mann–Whitney U test: compare pre vs post SH across individuals
        before_vals = []
        after_vals = []
        for animal, series in aligned_df.items():
            before = series[series.index < 0].mean()
            after = series[series.index > 0].mean()
            if not np.isnan(before) and not np.isnan(after):
                before_vals.append(before)
                after_vals.append(after)
        
        if len(before_vals) > 0:
            stat, p = mannwhitneyu(after_vals, before_vals, alternative='two-sided')
            print(f"Mann–Whitney U test: stat = {stat:.3f}, p = {p:.4g}")
        else:
            print("Not enough data for Mann–Whitney U test.")
    return aligned_df


def plot_mean_rotations_by_decision(SH_data, rotation_col='rotation', decision_col='scatter-hoard', fruit = None, after_first_sh = False):
    """
    Plot mean rotations for scatter-hoard vs eat decisions with minimalist styling.
    """
    if after_first_sh:
        SH_data = SH_data.copy()
        if 'ID' in SH_data.columns:
            filtered_dfs = []
            for animal, group in SH_data.groupby('ID'):
                sh_mask = (group[decision_col] == 1.0)
                if sh_mask.any():
                    first_sh_idx = sh_mask.values.argmax()
                    filtered_dfs.append(group.iloc[first_sh_idx + 1:])
            if filtered_dfs:
                SH_data = pd.concat(filtered_dfs)
            else:
                SH_data = SH_data.iloc[0:0]
        else:
            sh_mask = (SH_data[decision_col] == 1.0)
            if sh_mask.any():
                first_sh_idx = sh_mask.values.argmax()
                SH_data = SH_data.iloc[first_sh_idx + 1:]
            else:
                SH_data = SH_data.iloc[0:0]

    # Compute means and counts for stats
    grouped = SH_data.groupby(decision_col)[rotation_col]
    means = grouped.mean().reindex([0, 1], fill_value=0.0)
    counts = grouped.sum().reindex([0, 1], fill_value=0.0)
    nobs = grouped.count().reindex([0, 1], fill_value=0.0)

    # Define labels
    labels = ['Eat', 'Scatter-hoard']

    # --- Statistical Test (Proportions Z-test) ---
    # We expect decision_col to be 0 for Eat and 1 for Scatter-hoard
    # Match the order of labels [0, 1]
    if (nobs.values > 0).all():
        stat, pval = proportions_ztest(count=counts.values, nobs=nobs.values)
        print(f"--- Stat Test: Proportions Z-test (Eat vs Scatter-hoard) ---")
        print(f"Counts: {counts.values}, Totals: {nobs.values}")
        print(f"Z-stat: {stat:.3f}, p-val: {pval:.4g}")
    else:
        stat, pval = np.nan, np.nan
        print(f"--- Stat Test: Proportions Z-test (Eat vs Scatter-hoard) ---")
        print(f"Counts: {counts.values}, Totals: {nobs.values}")
        print(f"Insufficient data for proportions z-test (one or both categories have zero observations).")

    def get_stars(p):
        if pd.isna(p): return 'n.s.'
        if p < 0.001: return '***'
        if p < 0.01: return '**'
        if p < 0.05: return '*'
        return 'n.s.'

    # --- Plot setup ---
    plt.figure(figsize=(4,5))
    sns.set_style("white")  # no grid background

    # Control spacing manually
    x_positions = np.arange(len(labels)) * 1  # <--- increase value for more spacing
    bars = plt.bar(
        x_positions, means,
        color=['#1A897B', '#FFB722'], width=0.7
    )

    # Annotate means
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{means.iloc[i]:.2f}",
            ha='center', va='bottom',
            fontsize=17
        )

    # --- Add Significance Bracket ---
    if pval < 0.05 or True: # Force show n.s. if requested, but standard is p < 0.05
        y_max = means.max() + 0.15
        h = 0.05
        plt.plot([x_positions[0], x_positions[0], x_positions[1], x_positions[1]], 
                 [y_max, y_max+h, y_max+h, y_max], lw=1.5, c='black')
        plt.text((x_positions[0] + x_positions[1]) * .5, y_max + h, get_stars(pval), 
                 ha='center', va='bottom', color='black', fontsize=18)

    # --- Styling ---
    plt.ylabel('Probability of rotation', fontsize=17)
    plt.title('Probability of rotation\ngiven decision', fontsize=17)
    plt.xticks(x_positions, labels, fontsize=17)
    plt.yticks(fontsize=17)

    # Make ticks visible and slightly thicker
    plt.tick_params(axis='both', width=2, length=6, direction='out', color='black')

    plt.tight_layout()

    # Remove all spines except x and y
    sns.despine(top=True, right=True)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().tick_params(which='both', bottom=True, left=True)

    plt.ylim(0, 1.2) # Adjusted to fit bracket

    if fruit:
        save_path = r'figures/rotation_prob_' + str(fruit) + '.svg'
        plt.savefig(
            save_path, 
            dpi=300,               # High resolution (perfect for publications)
            bbox_inches='tight',    # Ensures labels/titles aren't cut off
            transparent=True,      # Set to True if you want a transparent background
            facecolor='white'       # Ensures background is solid white
        )
        print(f"Plot saved to: {save_path}")
    else:
        print("No fruit name provided, skipping save.")
        
    plt.show()

    

def compute_props(df, proba):
    # Initialize all potential outputs to nan
    sh = eat = number_sh = number_eat = np.nan
    mea_jaw = med_jaw = mea_rotation = med_rotation = np.nan

    # Convert to numeric safely
    if "jaws" in df.columns:
        df["jaws"] = pd.to_numeric(df["jaws"], errors='coerce')
    if "rotation" in df.columns:
        df["rotation"] = pd.to_numeric(df["rotation"], errors='coerce')
    if "scatter-hoard" in df.columns:
        df["scatter-hoard"] = pd.to_numeric(df["scatter-hoard"], errors='coerce')
    if "eaten" in df.columns:
        df["eaten"] = pd.to_numeric(df["eaten"], errors='coerce')
    if "teeth-hole contact" in df.columns:
        df["teeth-hole contact"] = pd.to_numeric(df["teeth-hole contact"], errors='coerce')

    peanut_num = df.shape[0]

    # Case: P(behavior)
    if proba == 'behavior':
        subset = [c for c in ["scatter-hoard", "eaten"] if c in df.columns]
        valid_rows = df.dropna(subset=subset).shape[0] if subset else 0
        
        if valid_rows == 0:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num,
                    'mean_num_jaw': np.nan, 'median_num_jaw': np.nan, 'mean_num_rot': np.nan, 'median_num_rot': np.nan}

        if "scatter-hoard" in df.columns:
            number_sh = df[df["scatter-hoard"] == 1].shape[0]
            sh = number_sh / valid_rows
        
        if "eaten" in df.columns:
            number_eat = df[df["eaten"] == 1].shape[0]
            eat = number_eat / valid_rows

        if "jaws" in df.columns:
            mea_jaw = np.nanmean(df['jaws'])
            med_jaw = np.nanmedian(df['jaws'])
        
        if "rotation" in df.columns:
            mea_rotation = np.nanmean(df['rotation'])
            med_rotation = np.nanmedian(df['rotation'])
        


    # Case: P(behavior | rotation)
    elif proba == 'given rotate':
        if "rotation" not in df.columns:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num,
                    'mean_num_jaw': np.nan, 'median_num_jaw': np.nan, 'mean_num_rot': np.nan, 'median_num_rot': np.nan}

        df_rotate = df[df["rotation"] == 1]
        peanut_num = df_rotate.shape[0]

        if df_rotate.empty:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        subset = [c for c in ["scatter-hoard", "eaten"] if c in df_rotate.columns]
        valid_rows = df_rotate.dropna(subset=subset).shape[0] if subset else 0
        if valid_rows == 0:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        if "scatter-hoard" in df_rotate.columns:
            number_sh = df_rotate[df_rotate["scatter-hoard"] == 1].shape[0]
            sh = number_sh / valid_rows
        
        if "eaten" in df_rotate.columns:
            number_eat = df_rotate[df_rotate["eaten"] == 1].shape[0]
            eat = number_eat / valid_rows

        number_sh = df_rotate[df_rotate["scatter-hoard"] == 1].shape[0]
        number_eat = df_rotate[df_rotate["eaten"] == 1].shape[0]

        sh = number_sh / valid_rows
        eat = number_eat / valid_rows

    # Case: P(behavior | not rotate)
    elif proba == 'given not rotate':
        if "rotation" not in df.columns:
             return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        df_rotate = df[df["rotation"] == 0]
        peanut_num = df_rotate.shape[0]

        if df_rotate.empty:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        subset = [c for c in ["scatter-hoard", "eaten"] if c in df_rotate.columns]
        valid_rows = df_rotate.dropna(subset=subset).shape[0] if subset else 0
        if valid_rows == 0:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        if "scatter-hoard" in df_rotate.columns:
            number_sh = df_rotate[df_rotate["scatter-hoard"] == 1].shape[0]
            sh = number_sh / valid_rows
        
        if "eaten" in df_rotate.columns:
            number_eat = df_rotate[df_rotate["eaten"] == 1].shape[0]
            eat = number_eat / valid_rows

    # Case: P(rotation | behavior)
    elif proba == 'given behavior':
        subset = [c for c in ["rotation", "scatter-hoard", "eaten"] if c in df.columns]
        valid_rows = df.dropna(subset=subset).shape[0] if subset else 0
        if valid_rows == 0:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        if "scatter-hoard" in df.columns:
            number_sh = df[df["scatter-hoard"] == 1].shape[0]
            if "rotation" in df.columns:
                sh = (df[(df["rotation"] == 1) & (df["scatter-hoard"] == 1)].shape[0] / number_sh 
                      if number_sh != 0 else np.nan)
        
        if "eaten" in df.columns:
            number_eat = df[df["eaten"] == 1].shape[0]
            if "rotation" in df.columns:
                eat = (df[(df["rotation"] == 1) & (df["eaten"] == 1)].shape[0] / number_eat 
                       if number_eat != 0 else np.nan)

    # Case: P(behavior | teeth-hole contact)
    elif proba == 'given teeth-hole':
        if "teeth-hole contact" not in df.columns:
             return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        df_teeth = df[df["teeth-hole contact"] == 1]
        peanut_num = df_teeth.shape[0]

        if df_teeth.empty:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        subset = [c for c in ["scatter-hoard", "eaten"] if c in df_teeth.columns]
        valid_rows = df_teeth.dropna(subset=subset).shape[0] if subset else 0
        if valid_rows == 0:
            return {'Scatter hoard': [np.nan, np.nan], 'Eat': [np.nan, np.nan], 'number of peanuts': peanut_num}

        if "scatter-hoard" in df_teeth.columns:
            number_sh = df_teeth[df_teeth["scatter-hoard"] == 1].shape[0]
            sh = number_sh / valid_rows
        
        if "eaten" in df_teeth.columns:
            number_eat = df_teeth[df_teeth["eaten"] == 1].shape[0]
            eat = number_eat / valid_rows

    else:
        raise ValueError("Invalid value for 'proba'. Must be one of: 'behavior', 'given rotate', 'given behavior', 'given teeth-hole'.")

    return {
        'Scatter hoard': [sh, number_sh],
        'Eat': [eat, number_eat],
        'number of peanuts': peanut_num,
        'mean_num_jaw': mea_jaw,
        'median_num_jaw': med_jaw,
        'mean_num_rot': mea_rotation,
        'median_num_rot': med_rotation
    }



    



def create_violin_df(arena_data, proba, order, group_by):
    records = []
    N_peanuts = []
    # Gather proportions by peanut type and category
    for peanut_type in order:
        df_type = arena_data[arena_data["type"] == peanut_type]
        for group_value in df_type[group_by].dropna().unique():
            df_group = df_type[df_type[group_by] == group_value]
            props = compute_props(df_group, proba)
            #print(props)
            for cat in props:
                if cat != 'number of peanuts' and cat != 'mean_num_rots' and cat != 'mean_num_jaw' and cat!= 'median_num_jaw' and cat!= 'mean_num_rot' and cat!= 'median_num_rot':
                    records.append({
                        "type": peanut_type,
                        group_by: group_value,
                        "category": cat,
                        "proportion": props[cat][0]
                    })
                if cat == 'mean_num_rots' or cat == 'mean_num_jaw' or cat== 'median_num_jaw' or cat== 'mean_num_rot' or cat== 'median_num_rot':
                    records.append({
                        "type": peanut_type,
                        group_by: group_value,
                        "category": cat,
                        f"mean or median": props[cat]
                    })

            N_peanuts.append({"type": peanut_type, group_by: group_value, 'N_tot': props['number of peanuts']})

    N_df = pd.DataFrame(N_peanuts)

    violin_df = pd.DataFrame(records)
    violin_df_rotations = violin_df.copy()
    violin_df = violin_df.dropna(subset=['proportion'])
    #print(violin_df[violin_df['type'] == 'empty'])
    
    num_types = len(np.unique(violin_df['type']))
    return violin_df, N_df, num_types, violin_df_rotations


def filter_violin_plot(violin_df, group_by, N_df, num_types, min_N_peanuts_per_violin, max_N_missing_violins):
    for group in np.unique(violin_df[group_by]):
        for tp in np.unique(violin_df['type'][violin_df[group_by] == group]):
            typ, grp = list(violin_df[['type', group_by]][(violin_df[group_by] == group) & (violin_df['type'] == tp)].iloc[0,:])
            N_typ_grp = N_df['N_tot'][(N_df['type'] == typ) & (N_df[group_by] == grp)]

            val = int(N_typ_grp.iloc[0]) if not N_typ_grp.empty else 0
            if val <= min_N_peanuts_per_violin:
                violin_df = violin_df[~((violin_df['type'] == typ) & (violin_df[group_by] == grp))]
                
        num_missing_types = num_types - len(np.unique(violin_df['type'][violin_df[group_by] == group]))
        if num_missing_types > max_N_missing_violins:
            violin_df = violin_df[violin_df[group_by] != group]
            
    return violin_df


def estimate_mannwhitney_power(data, control_group='control', category='Scatter hoard', 
                                alpha=0.05, n_sim=1000, alternative='two-sided'):
    """
    Estimate power for Mann-Whitney U test between control and other peanut types for Scatter hoard data.
    
    Parameters:
        data (pd.DataFrame): DataFrame with columns ['type', 'category', 'proportion']
        control_group (str): Label for control group in 'type' column
        category (str): Category to filter by (e.g., 'Scatter hoard')
        alpha (float): Significance level
        n_sim (int): Number of simulations per comparison
        alternative (str): 'two-sided', 'less', or 'greater'
    
    Returns:
        pd.DataFrame: Power estimates for each group vs. control
    """
    power_results = []
    
    # Filter to Scatter hoard data
    df = data[data['category'] == category]
    
    # Get control group data
    control_data = df[df['type'] == control_group]['proportion'].values
    
    for group in df['type'].unique():
        if group == control_group:
            continue
        
        test_data = df[df['type'] == group]['proportion'].values
        n1, n2 = len(control_data), len(test_data)
        
        if n1 < 3 or n2 < 3:
            print(f"Not enough data for group {group}. Skipping...")
            continue

        count_significant = 0
        for _ in range(n_sim):
            sample1 = np.random.choice(control_data, n1, replace=True)
            sample2 = np.random.choice(test_data, n2, replace=True)
            stat, p = mannwhitneyu(sample1, sample2, alternative=alternative)
            if p < alpha:
                count_significant += 1

        power = count_significant / n_sim
        power_results.append({
            'comparison': f'{group} vs {control_group}',
            'n_control': n1,
            'n_group': n2,
            'power': round(power, 3)
        })

    return pd.DataFrame(power_results)


def plot_violin_and_scatter(violin_df, order, hue_order, width, violin_width=0.6, x_spacing=1.0, fig_width=12):
    """
    Same as before, but when '0,3' is in violin_df['type'],
    violins are spaced at numeric x positions (0, 0.3, 0.5, 0.7, 1, 1.5, 2)
    and automatically adjust width/spacing to fit the figure.
    The violin shape is truncated at the min/max of the data (cut=0 equivalent).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(fig_width, 8))
    sns.set_style("white")

    # === CASE 1: numeric spacing (integrities) ===
    if '0,3' in violin_df['type'].unique():
        mapping = {'control': 0, '0,3': 0.3, '0,5': 0.5, '0,7': 0.7,
                   '1': 1.0, '1,5': 1.5, '2': 2.0}
        violin_df = violin_df.copy()
        violin_df['x_val'] = violin_df['type'].map(mapping)

        colors = ["#FFB722", "#1A897B"]

        # automatically scale violin width and offset to fit plot
        x_vals_unique = sorted(violin_df['x_val'].dropna().unique())
        x_range = max(x_vals_unique) - min(x_vals_unique)
        violin_width = x_range * 0.02  # dynamic width (4% of x-range)
        offset = violin_width * 1   # separation between hues

        for j, category in enumerate(hue_order):
            subset = violin_df[violin_df["category"] == category]
            for x_val, group in subset.groupby("x_val"):
                vals = group["proportion"].dropna().values
                if len(vals) < 2:
                    continue

                # --- compute KDE shape for violin ---
                if np.std(vals) == 0:
                    y = np.linspace(vals[0] - 0.01, vals[0] + 0.01, 200)
                    v = np.ones_like(y) * violin_width
                else:
                    kde = gaussian_kde(vals)
                    y = np.linspace(vals.min(), vals.max(), 200)  # cut=0 equivalent
                    v = kde(y)
                    v = v / v.max() * violin_width
                x_shift = x_val + (j - (len(hue_order)-1)/2) * 2 * offset if len(hue_order) > 1 else x_val

                # --- violin body ---
                ax.fill_betweenx(y, x_shift - v, x_shift + v,
                                 facecolor=colors[j],
                                 alpha=1,
                                 linewidth=1.2,
                                 edgecolor="black")

                # --- box interior: median & IQR ---
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                print(category, med)
                min_val = np.min(vals)
                max_val = np.max(vals)
                ax.plot([x_shift - violin_width * 0.1, x_shift + violin_width * 0.1],
                        [med, med],
                        color="white", lw=1.5, zorder=10)
                ax.plot([x_shift, x_shift], [q1, q3],
                        color="black", lw=5, zorder=5)
                ax.plot([x_shift, x_shift], [min_val, max_val], color = 'black',
                            lw = 1, zorder = 5)

                # --- raw data scatter (jittered) ---
                jitter = np.random.uniform(-violin_width * 0.7, violin_width * 0.7, size=len(vals))
                ax.scatter(np.full_like(vals, x_shift) + jitter,
                           vals, color='black', s=28,
                           edgecolor='black', linewidth=0.5,
                           alpha=1, zorder=10)

        # --- axis setup ---
        ax.set_xticks([0, 0.3, 0.5, 0.7, 1, 1.5, 2])
        ax.set_xticklabels(['control', '0.3', '0.5', '0.7', '1', '1.5', '2'])
        ax.set_xlim(min(x_vals_unique) - violin_width * 3,
                    max(x_vals_unique) + violin_width * 3)
        ax.set_ylim(-0.05, 1.05)

    # === CASE 2: default categorical spacing ===
    else:
        # Create numeric mapping for custom spacing (used for scatters/lines)
        x_map = {t: i * x_spacing for i, t in enumerate(order)}
        
        for tp in np.unique(violin_df['type']):
            print(tp, np.median(violin_df['proportion'][violin_df['type'] == tp]))
            #print('number of individuals:', len(np.unique(violin_df['ID'])))
        
        sns.violinplot(
            data=violin_df,
            x="type",
            y="proportion",
            hue="category",
            hue_order=hue_order,
            order=order,
            palette=["#FFB722", "#1A897B"],
            cut=0,
            scale="width",
            inner="box",
            dodge=True,
            alpha=0.9,
            width=violin_width,
            linewidth=1.2,
            ax=ax
        )

        # remove exterior outlines
        for artist in ax.collections:
            if isinstance(artist, plt.Polygon):
                artist.set_edgecolor("none")

        # overlay scatter
        for i, peanut_type in enumerate(order):
            for j, category in enumerate(hue_order):
                if category == 'Scatter hoard':
                    offset = 1.3
                if category == 'Eat':
                    offset = 0.635
                subset = violin_df[
                    (violin_df["type"] == peanut_type) &
                    (violin_df["category"] == category)
                ]
                # Center scatters on the i * x_spacing coordinate
                if len(hue_order) == 1:
                    x_center = x_map[peanut_type]
                else:
                    x_center = x_map[peanut_type] - width / 2 + j * offset + offset / 2
                
                jitter_strength = 0.06
                x_vals = x_center + np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
                ax.scatter(
                    x_vals,
                    subset["proportion"],
                    color='black',
                    s=28,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=1,
                    zorder=10
                )

        # Update ticks to respect x_spacing if used for other components
        ax.set_xticks([x_map[t] for t in order])
        ax.set_xticklabels(order)

    # --- universal styling ---
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', labelsize=16, width=2)
    ax.set_ylabel("Conditional probability", fontsize=18)
    ax.set_xlabel("", fontsize=18)
    #ax.get_legend().remove()
    
    plt.tick_params(axis='both', width=2, length=6, color='black', direction='out')
    plt.gca().tick_params(which='both', bottom=True, left=True)
    plt.tight_layout()
    return fig, ax


# ===========================================================
# === 2. SCATTER CONNECTIONS ================================
# ===========================================================
def plot_scatter_connections(fig, ax, violin_df, group_by, hue_order, connect_scatter, order, width, x_spacing=1.0):
    colors = ['red', 'black', "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
              "#D55E00", "#CC79A7", "#999999", "#1B9E77", "#D95F02", "#7570B3",
              "#E7298A", "#66A61E", "#E6AB02", "#A6761D"]

    n_hue = len(hue_order)
    offset = width / n_hue
    
    if connect_scatter:
        unique_groups = sorted(violin_df[group_by].dropna().unique())
        color_map = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        linestyles = ['-', '--']

        for x, category in enumerate(hue_order):
            cat_df = violin_df[violin_df["category"] == category]
            style = linestyles[x]

            for group_value in unique_groups:
                subset = cat_df[cat_df[group_by] == group_value]
                if len(subset) < 2:
                    continue

                x_coords, y_coords = [], []
                for i, peanut_type in enumerate(order):
                    point = subset[subset["type"] == peanut_type]
                    if len(point) == 0:
                        continue
                    j = hue_order.index(category)
                    x_center = i * x_spacing - width / 2 + j * offset + offset / 2
                    x_coords.append(x_center)
                    y_coords.append(point["proportion"].values[0])

                if len(x_coords) >= 2:
                    ax.plot(
                        x_coords, y_coords,
                        color=color_map[group_value],
                        alpha=1 if connect_scatter is True else 0.5,
                        linewidth=2.5,
                        linestyle=style,
                        zorder=5
                    )

    return fig, ax


# ===========================================================
# === 3. STATS ACROSS & WITHIN ==============================
# ===========================================================
def print_stats_across_peanut_types(violin_df, order, control, hue_order):
    treatments = [t for t in order if t != control]
    p_list_overall = []

    for category in hue_order:
        p_list = []
        for treatment in treatments:
            control_vals = violin_df[
                (violin_df['type'] == control) & (violin_df['category'] == category)
            ]['proportion'].dropna().values
            treatment_vals = violin_df[
                (violin_df['type'] == treatment) & (violin_df['category'] == category)
            ]['proportion'].dropna().values
            if len(control_vals) > 1 and len(treatment_vals) > 1:
                p = mannwhitneyu(control_vals, treatment_vals, alternative='two-sided')[1]
            else:
                p = np.nan
            p_list.append(p)
        p_list_overall.extend(p_list)

    rejected, pvals_corrected, _, _ = multipletests(p_list_overall, method='holm')
    print("\nCorrected Holm–Bonferroni p-values across types:")
    for i, (p, rej) in enumerate(zip(pvals_corrected, rejected)):
        print(f"{i+1}: p={p:.4f} ({'*' if rej else 'n.s.'})")


def print_stats_within_peanut_types(violin_df, order, show_stats):
    if not show_stats or len(violin_df['category'].unique()) < 2:
        return
    pvals, comparisons = [], []
    for peanut_type in order:
        sh_vals = violin_df[
            (violin_df['type'] == peanut_type) & (violin_df['category'] == "Scatter hoard")
        ]['proportion'].dropna().values
        eat_vals = violin_df[
            (violin_df['type'] == peanut_type) & (violin_df['category'] == "Eat")
        ]['proportion'].dropna().values

        if len(sh_vals) > 1 and len(eat_vals) > 1:
            p = mannwhitneyu(sh_vals, eat_vals, alternative='two-sided')[1]
        else:
            p = np.nan

        pvals.append(p)
        comparisons.append(peanut_type)

    rejected, pvals_corrected, _, _ = multipletests(pvals, method='holm')
    print("\nWithin-type corrected p-values:")
    for t, p, rej in zip(comparisons, pvals_corrected, rejected):
        print(f"{t}: p={p:.4f} ({'*' if rej else 'n.s.'})")


def violin_esthetics(fig, ax, violin_df, proba, group_by, hue_order,
                     legend_pos=(0.5, 0.5)):
    """Finalize axis style and labeling for violin plot."""
    # --- Legend (safe remove)
    #handles, labels = ax.get_legend_handles_labels()
    #if handles:
        # optional: you can comment this out if you prefer no legend ever
        #ax.legend(handles[:len(hue_order)], labels[:len(hue_order)],
                  #fontsize=14, loc='lower center',
                  #bbox_to_anchor=legend_pos, frameon=False, ncol=1)
    #else:
        # safely skip when no legend was generated
        #if getattr(ax, "legend_", None) is not None:
            #ax.legend_.remove()

    # --- Titles and labels
    title_map = {
        'behavior': 'Probability of decision across peanut types',
        'given behavior': 'Probability of rotating given SH or Eat',
        'given rotate': 'Probability of SH or Eat given rotated',
        'given teeth-hole': 'Probability of SH or Eat given teeth-hole contact'
    }
    ax.set_title(title_map.get(proba, 'Proportion by peanut type'), fontsize=20)
    ax.set_ylabel("Conditional probability", fontsize=20)
    ax.set_xlabel("", fontsize=20)

    # --- Axis tick labels depending on type set
    if '1cm_hole' in violin_df['type'].unique():
        ax.set_xticklabels(["control", "1cm hole", "2cm hole", "crushed"])
    elif 'hole' in violin_df['type'].unique():
        ax.set_xticklabels(["control", "1cm hole", "crushed"])
    elif '0,3' in violin_df['type'].unique():
        ax.set_xticks([0, 0.3, 0.5, 0.7, 1, 1.5, 2])
        ax.set_xticklabels(["control", "0.3", "0.5", "0.7", "1", "1.5", "2"])

    # --- Aesthetics
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=20, width=2, length=6, color='black', direction='out')
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    fig.tight_layout()
    return violin_df


# ===========================================================
# === 2. ADAPTIVE REGRESSION LINE HANDLER ===================
# ===========================================================
def plot_regression_lines(fig, ax, violin_df, order, hue_order, width, n_std, show_reg_lines, x_spacing=1.0, fit_type='linear'):
    """Plot category-wise regression lines, adapted for numeric or categorical x."""
    if not show_reg_lines:
        return fig, ax

    print('\nStats regression lines:')
    n_hue = len(hue_order)
    offset = width / n_hue

    # Check if we have numeric integrity values (e.g. '0,3')
    use_numeric_x = '0,3' in violin_df['type'].unique()
    if use_numeric_x:
        mapping = {'control': 0, '0,3': 0.3, '0,5': 0.5, '0,7': 0.7,
                   '1': 1.0, '1,5': 1.5, '2': 2.0}

    for cat_i, cat in enumerate(hue_order):
        cat_df = violin_df[violin_df['category'] == cat]

        # Use numeric x if applicable
        if use_numeric_x:
            cat_df = cat_df.assign(x_val=cat_df['type'].map(mapping))
            if len(hue_order) == 1:
                x = cat_df['x_val'].values
            else:
                if list(cat_df['category'])[0] == 'Scatter hoard':
                    x = cat_df['x_val'].values - 0.025
                else:
                    x = cat_df['x_val'].values + 0.025
        else:
            if len(hue_order) == 1:
                x = cat_df['type'].map({v: i * x_spacing for i, v in enumerate(order)}).values
            else:
                # Maintain compatibility with the asymmetrical offsets in categorical scatter
                if cat == 'Scatter hoard':
                    hue_shift = -0.15 # Derived from i - width/2 + 0 + 1.3/2
                else:
                    hue_shift = 0.8175 # Derived from i - width/2 + 1.3 + 0.635/2
                x = cat_df['type'].map({v: i * x_spacing for i, v in enumerate(order)}).values + hue_shift

        y = cat_df['proportion'].values
        if len(x) < 2:
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_vals = np.linspace(min(x), max(x), 100)
        x_shift = 0  # we no longer dodge here, regression lines overlay both categories
        
        if fit_type == 'sigmoid':
            def sigmoid(x, L, x0, k, b):
                return L / (1 + np.exp(-k*(x-x0))) + b
            
            p0 = [max(y), np.median(x), 1, min(y)] # Initial guess
            try:
                popt, pcov = curve_fit(sigmoid, x, y, p0, maxfev=10000)
                y_pred = sigmoid(x_vals, *popt)
                fit_label = 'sigmoid'
                
                # Parameters and errors
                L_val, x0_val, k_val, b_val = popt
                L_err, x0_err, k_err, b_err = np.sqrt(np.diag(pcov))
                          # p-value for k (slope) using Wald test
                p_value = 2 * norm.sf(np.abs(k_val/k_err)) 
                
                # Calculate R^2 for the sigmoid fit
                y_fit = sigmoid(x, *popt)
                ss_res = np.sum((y - y_fit)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_value = np.sqrt(max(0, 1 - ss_res / ss_tot)) if ss_tot > 0 else 0
                
                print(f"  {cat} {fit_label}:")
                print(f"    L (scale):    {L_val:.4f} ± {L_err:.4f}")
                print(f"    x0 (midpoint): {x0_val:.4f} ± {x0_err:.4f}")
                print(f"    k (steepness): {k_val:.4f} ± {k_err:.4f}")
                print(f"    b (offset):   {b_val:.4f} ± {b_err:.4f}")
                print(f"    p-value (k):  {p_value:.10f}")
                print(f"    R² (sigmoid): {r_value**2:.4f}")
            except Exception as e:
                print(f"  {cat} sigmoid fit failed: {e}")
                y_pred = slope * x_vals + intercept
                fit_label = 'linear (fallback)'
                print(f"  {cat} {fit_label}: p = {p_value:.10f}")
        elif fit_type == 'logistic':
            def logistic(x, k, x0):
                exp_val = -k * (x - x0)
                exp_val = np.clip(exp_val, -500, 500)
                return 1.0 / (1.0 + np.exp(exp_val))
            
            p0 = [1.0, np.median(x)] # Initial guess
            try:
                popt, pcov = curve_fit(logistic, x, y, p0, maxfev=10000)
                y_pred = logistic(x_vals, *popt)
                fit_label = 'logistic'
                
                # Parameters and errors
                k_val, x0_val = popt
                k_err, x0_err = np.sqrt(np.diag(pcov))
                p_value = 2 * norm.sf(np.abs(k_val/k_err))
                
                # Calculate R^2 for the logistic fit
                y_fit = logistic(x, *popt)
                ss_res = np.sum((y - y_fit)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_value = np.sqrt(max(0, 1 - ss_res / ss_tot)) if ss_tot > 0 else 0
                
                print(f"  {cat} {fit_label}:")
                print(f"    k (steepness): {k_val:.4f} ± {k_err:.4f}")
                print(f"    x0 (midpoint): {x0_val:.4f} ± {x0_err:.4f}")
                print(f"    p-value (k):  {p_value:.10f}")
                print(f"    R² (logistic): {r_value**2:.4f}")
            except Exception as e:
                print(f"  {cat} logistic fit failed: {e}")
                y_pred = slope * x_vals + intercept
                fit_label = 'linear (fallback)'
                print(f"  {cat} {fit_label}: p = {p_value:.10f}")
        else:
            y_pred = slope * x_vals + intercept
            fit_label = 'linear'
            print(f"  {cat} {fit_label}: p = {p_value:.10f}")

        # Bootstrap envelope
        n_boot = 1000
        y_boot = np.zeros((n_boot, len(x_vals)))
        rng = np.random.default_rng(42)
        for i in range(n_boot):
            idx = rng.choice(len(x), size=len(x), replace=True)
            x_s, y_s = x[idx], y[idx]
            try:
                if fit_label == 'sigmoid':
                    popt_b, _ = curve_fit(sigmoid, x_s, y_s, p0, maxfev=5000)
                    y_boot[i, :] = sigmoid(x_vals, *popt_b)
                elif fit_label == 'logistic':
                    popt_b, _ = curve_fit(logistic, x_s, y_s, p0, maxfev=5000)
                    y_boot[i, :] = logistic(x_vals, *popt_b)
                else:
                    m, b_lin = np.polyfit(x_s, y_s, 1)
                    y_boot[i, :] = m * x_vals + b_lin
            except:
                y_boot[i, :] = np.nan
        std_env = np.nanstd(y_boot, axis=0)
        y_upper = y_pred + n_std * std_env
        y_lower = y_pred - n_std * std_env

        color = "#FFB722" if cat == "Scatter hoard" else "#1A897B"
        ax.plot(x_vals + x_shift, y_pred, color=color, lw=2.5,
                ls='--', label=f'{cat} {fit_label}', zorder=8)
        ax.fill_between(x_vals + x_shift, y_lower, y_upper,
                        color=color, alpha=0.2, zorder=2)

        print(f'{cat}: {fit_label} R²={r_value**2:.4f}, p={p_value:.4f}')

    return fig, ax
    

def plot_arena(arena_data, proba, order, group_by, min_N_peanuts_per_violin, max_N_missing_violins = 10, 
               control='control', connect_scatter=False, show_stats=False, show_reg_lines = False, 
               n_std = 1, save_path = None, show = True, only_scatter_hoard = False, 
               violin_width = 0.6, x_spacing = 1.0, fig_width = 12, comparisons='control', fit_type='linear'): 
    """
    comparisons: 'control' (Mann-Whitney vs control) or 'all' (Omnibus Chi2 + pairwise Fisher)
    """
    hue_order = ["Scatter hoard", "Eat"] if not only_scatter_hoard else ["Scatter hoard"]
    width = 1.6 # Alignment width for scatter points
    
    violin_df, N_df, num_types, violin_df_rotations = create_violin_df(arena_data, proba, order, group_by) 
    
    if only_scatter_hoard:
        violin_df = violin_df[violin_df["category"] == "Scatter hoard"].copy()
        violin_df_rotations = violin_df_rotations[violin_df_rotations["category"] == "Scatter hoard"].copy()
    
    violin_df = filter_violin_plot(violin_df, group_by, N_df, num_types, min_N_peanuts_per_violin, max_N_missing_violins) 
    
    violin_df_rotations = filter_violin_plot(violin_df_rotations, group_by, N_df, num_types, min_N_peanuts_per_violin, max_N_missing_violins) 
    power_df = estimate_mannwhitney_power(violin_df) 
    print(power_df) 
    
    fig, ax = plot_violin_and_scatter(violin_df, order, hue_order, width, 
                                      violin_width=violin_width, x_spacing=x_spacing, fig_width=fig_width) 
    fig, ax = plot_scatter_connections(fig, ax, violin_df, group_by, hue_order, connect_scatter, order, width, x_spacing=x_spacing) 
    
    if comparisons == 'control':
        print_stats_across_peanut_types(violin_df, order, control, hue_order) 
    print_stats_within_peanut_types(violin_df, order, show_stats) 
    
    fig, ax = plot_regression_lines(fig, ax, violin_df, order, hue_order, width, n_std, show_reg_lines, x_spacing=x_spacing, fit_type=fit_type) 
    
    # --- SIGNIFICANCE BRACKETS ---
    if comparisons:
        from statsmodels.stats.multitest import multipletests
        from itertools import combinations
        from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
        
        x_map = {t: i * x_spacing for i, t in enumerate(order)}
        
        for k, category in enumerate(hue_order):
            pvals = []
            comparisons_list = []
            
            if comparisons == 'control':
                treatments = [t for t in order if t != control]
                for treatment in treatments:
                    c_vals = violin_df[(violin_df['type'] == control) & (violin_df['category'] == category)]['proportion'].dropna().values
                    t_vals = violin_df[(violin_df['type'] == treatment) & (violin_df['category'] == category)]['proportion'].dropna().values
                    if len(c_vals) > 0 and len(t_vals) > 0:
                        p = mannwhitneyu(c_vals, t_vals, alternative='two-sided')[1]
                    else: p = 1.0
                    pvals.append(p)
                    comparisons_list.append((control, treatment))
                    
            elif comparisons == 'all':
                # Build Rx2 table for Omnibus
                table = []
                for t in order:
                    sub = arena_data[arena_data['type'] == t]
                    # We need to compute success counts. Use compute_props logic.
                    # Simplified here: assuming binary data in columns named after category
                    cat_col = 'scatter-hoard' if category == 'Scatter hoard' else 'eaten'
                    if cat_col in arena_data.columns:
                        success = arena_data[arena_data['type'] == t][cat_col].sum()
                        total = arena_data[arena_data['type'] == t][cat_col].count()
                        table.append([success, total - success])
                
                if len(table) > 1:
                    _, p_omnibus, _, _ = chi2_contingency(table)
                    print(f"\nOmnibus Chi2 for {category}: p={p_omnibus:.10g}")
                    
                    if p_omnibus < 0.05:
                        for (i, t1), (j, t2) in combinations(enumerate(order), 2):
                            _, p = fisher_exact([table[i], table[j]])
                            pvals.append(p)
                            comparisons_list.append((t1, t2))
                
            if pvals:
                rejected, p_adj, _, _ = multipletests(pvals, method='holm')
                
                if comparisons == 'all':
                   print(f"\nPairwise comparisons for {category} (Holm-corrected Fisher):")
                   for (g1, g2), padj, rej in zip(comparisons_list, p_adj, rejected):
                       print(f"  {g1} vs {g2}: p={padj:.10g} {'*' if rej else 'n.s.'}")
                
                # Draw Brackets
                y_base = violin_df[violin_df['category'] == category]['proportion'].max() + 0.05 + (k * 0.1)
                for (g1, g2), padj, rej in zip(comparisons_list, p_adj, rejected):
                    if rej:
                        x1, x2 = x_map[g1], x_map[g2]
                        if len(hue_order) > 1:
                            offset = -width/2 + k * 0.965 + 0.965/2 # match scatter offset
                            x1 += offset
                            x2 += offset
                        
                        y_pos = y_base + (pvals.index(min(pvals)) * 0.05) # basic stacking
                        h = 0.02
                        ax.plot([x1, x1, x2, x2], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, c='black')
                        stars = '***' if padj < 0.001 else '**' if padj < 0.01 else '*'
                        ax.text((x1+x2)/2, y_pos+h, stars, ha='center', va='bottom', fontsize=12)
                        y_base += 0.1 # stack next bracket

    violin_esthetics(fig, ax, violin_df, proba, group_by, hue_order) 
    
    if save_path:
        fig.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True
        )
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()

    return violin_df_rotations




def plot_compute_single_prob(
    data,
    prob_to_compute,
    order,
    color,
    show_scatter=False,
    scatter_alpha=0.6,
    scatter_size=40,
    jitter=0.08,
    # ---- stats args ----
    test=None,                    # None, "fisher", "chi2"
    correction="holm",            # for post-hoc when chi2 & >2 groups
    annotate_test=True,
    fisher_alternative="two-sided",
    # ---- export args ----
    save_path=None,
    show=True
):
    """
    Bar plot of probabilities per type with optional scatter overlay.

    Statistical behavior
    --------------------
    - test="fisher": requires len(order)==2 → Fisher exact test
    - test="chi2":
        - len(order)==2 → chi-square (2x2)
        - len(order)>2  → chi-square omnibus + post-hoc pairwise Fisher tests

    Returns
    -------
    results_df : pd.DataFrame or None
        Statistical results table (None if test is None)
    """

    results_df = None

    # ------------------
    # Compute means + counts
    # ------------------
    rotation_probabilities = {
        g: [
            np.nanmean(data.loc[data["type"] == g, prob_to_compute]),
            data.loc[data["type"] == g, prob_to_compute].dropna().shape[0]
        ]
        for g in order
    }

    labels = list(rotation_probabilities.keys())
    proportions = [v[0] for v in rotation_probabilities.values()]
    counts = [v[1] for v in rotation_probabilities.values()]

    # ------------------
    # Plot bars
    # ------------------
    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, proportions, color=color, zorder=2)

    # ------------------
    # Optional scatter
    # ------------------
    if show_scatter:
        for i, g in enumerate(order):
            vals = data.loc[data["type"] == g, prob_to_compute].dropna()
            if not vals.empty:
                x = np.random.normal(i, jitter, size=len(vals))
                plt.scatter(x, vals, s=scatter_size,
                            color="black", alpha=scatter_alpha, zorder=3)

    ax = plt.gca()

    # ------------------
    # Count annotations
    # ------------------
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"N={int(n)}",
            ha="center", va="bottom", fontsize=18
        )

    # ------------------
    # Statistics
    # ------------------
    if test is not None:
        test = test.lower()

        # build Rx2 table
        present = []
        total = []
        for g in order:
            vals = data.loc[data["type"] == g, prob_to_compute].dropna().astype(int)
            present.append(int(vals.sum()))
            total.append(int(vals.shape[0]))

        table = np.array([[x, n - x] for x, n in zip(present, total)])

        if np.any(table < 0):
            raise ValueError(f"`{prob_to_compute}` must be binary (0/1).")

        # ---- Fisher: only 2 groups ----
        if test == "fisher":
            if len(order) != 2:
                raise ValueError("Fisher exact test requires exactly 2 groups.")

            _, p = fisher_exact(table, alternative=fisher_alternative)

            results_df = pd.DataFrame([{
                "test": "fisher_exact",
                "group1": order[0],
                "group2": order[1],
                "x1": present[0],
                "n1": total[0],
                "x2": present[1],
                "n2": total[1],
                "p_value": p
            }])

            if annotate_test:
                ax.set_title(f"Fisher exact: p={p:.3g}", fontsize=20)
            print(f"\n--- Fisher Exact Test ({prob_to_compute}) ---")
            print(f"Groups: {order[0]} vs {order[1]}")
            print(f"Frequencies: {present[0]}/{total[0]} vs {present[1]}/{total[1]}")
            print(f"p-value: {p:.10g}")

        # ---- Chi-square: 2+ groups ----
        elif test in ["chi2", "chi"]:
            from scipy.stats import chi2_contingency
            chi2, p_global, dof, _ = chi2_contingency(table, correction=False)

            rows = [{
                "test": "chi_square_global",
                "groups": order,
                "chi2": chi2,
                "dof": dof,
                "p_value": p_global
            }]
            print(f"\n--- Chi-square Omnibus Test ({prob_to_compute}) ---")
            print(f"Groups: {order}")
            print(f"Chi2: {chi2:.3f}, df: {dof}, p-value: {p_global:.10g}")

            # ---- post-hoc if >2 groups ----
            if len(order) > 2:
                from statsmodels.stats.multitest import multipletests
                pairwise_rows = []
                for i in range(len(order)):
                    for j in range(i + 1, len(order)):
                        g1, g2 = order[i], order[j]
                        x1, n1 = present[i], total[i]
                        x2, n2 = present[j], total[j]

                        _, p = fisher_exact(
                            [[x1, n1 - x1], [x2, n2 - x2]],
                            alternative="two-sided"
                        )

                        pairwise_rows.append({
                            "test": "pairwise_fisher",
                            "group1": g1,
                            "group2": g2,
                            "x1": x1,
                            "n1": n1,
                            "x2": x2,
                            "n2": n2,
                            "p_raw": p
                        })

                pairwise_df = pd.DataFrame(pairwise_rows)

                reject, p_adj, _, _ = multipletests(
                    pairwise_df["p_raw"], method=correction
                )
                pairwise_df["p_adj"] = p_adj
                pairwise_df["reject"] = reject

                print(f"Post-hoc Pairwise Fisher tests ({correction} corrected):")
                for _, row in pairwise_df.iterrows():
                    sig_str = '*' if row['reject'] else 'ns'
                    print(f"  {row['group1']} vs {row['group2']}: p={row['p_adj']:.10g} ({sig_str})")

                results_df = pd.concat(
                    [pd.DataFrame(rows), pairwise_df],
                    ignore_index=True
                )
            else:
                results_df = pd.DataFrame(rows)
    
    # ------------------
    # Axis styling
    # ------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    ax.tick_params(axis="both", which="both",
                   bottom=True, left=True,
                   width=2, length=6, labelsize=20)

    ax.set_xlabel(ax.get_xlabel().replace("_", " "), fontsize=20)
    ax.set_ylabel(ax.get_ylabel().replace("_", " "), fontsize=20)

    ax.set_xticklabels(
        [lbl.get_text().replace("_", " ") for lbl in ax.get_xticklabels()],
        fontsize=20
    )

    ax.grid(False)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)

    if show:
        plt.show()

    return results_df








    

    









    

        

            




#ff7f0e good orange

color_dict = {
    #'chomping': ['black', 'cho'],
    'jaw movement': ['#9F2B68','j'],         # blue
    'assessing': ['#d62728','A'],            # red
    'sniffing': ['#EF1FD1','s'],             # green
    'horizontal rotation': ['#8075C0','hR'],             # violet
    'vertical rotation': ['#8075C0','vR'],             # violet
    'yaw rotation': ['#8075C0','yR'],             # violet
    'eating': ['#1A897B','E'],               # greenish
    'chewing': ['#8c564b','C'],              # brown
    'bending': ['#e377c2','B'],              # pink
    'peeling': ['#7f7f7f','P'],              # gray
    'target': ['#bcbd22','T'],               # olive
    'explore': ['#17becf','EX'],              # teal
    'grabbing peanut': ['red','g'],      # red
    'licking': ['#FF7F0E','l'],              # light orange
    'mouth': ['orange','m'],                # light green
    'teeth-hole contact': ['red','t'],    # beige/pinkish
    'scatter-hoarding': ['#FFB722', 'S'],    # yellowish
    'lifting': ['gray', 'L'],
    'pause': ['gray', 'Pa'],
    'digging': ['gray', 'D'],
    'pushing': ['gray', 'Pu'],
    'translational movement': ['#9F2B68', 'tr'],
    'dropping peanut': ['gray', 'd'],
    'covering with soil': ['gray', 'C'],
    'translational jaw movement': ['#9F2B68', 'Trj'],
    'ear clapping': ['lightgray', 'e']
}

color_dict1 = {
    'chomping': ['black', 'cho'],
    'jaw movement': ['blue','j'],         # blue
    'assessing': ['#d62728','A'],            # red
    'sniffing': ['green','s'],             # green
    'rotation': ['#8075C0','hR'],
    'horizontal rotation': ['#8075C0','hR'],             # violet
    'vertical rotation': ['#8075C0','vR'],             # violet
    'yaw rotation': ['#8075C0','yR'],             # violet
    'eating': ['#1A897B','E'],               # greenish
    'chewing': ['#8c564b','C'],              # brown
    'bending': ['#e377c2','B'],              # pink
    'peeling': ['#7f7f7f','P'],              # gray
    'target': ['#bcbd22','T'],               # olive
    'explore': ['#17becf','EX'],              # teal
    'grabbing peanut': ['red','g'],      # red
    'licking': ['#ffbb78','l'],              # light orange
    'mouth': ['orange','m'],                # light green
    'teeth-hole contact': ['red','t'],    # beige/pinkish
    'scatter-hoarding': ['#FFB722', 'S'],    # yellowish
    'lifting': ['gray', 'L'],
    'pause': ['gray', 'Pa'],
    'digging': ['gray', 'D'],
    'pushing': ['gray', 'Pu'],
    'translational movement': ['gray', 'tr'],
    'dropping peanut': ['gray', 'd'],
    'covering with soil': ['gray', 'C'],
    'ear clapping': ['gray', 'e']
}

level_1_behaviors = {
        'assessing', 'bending', 'explore', 'eating', 'grabbing peanut', 'mouth', 'target', 'scatter-hoarding', 'pause', 'lifting', 'dropping peanut'
    }
level_2_behaviors = {
        'jaw movement', 'sniffing', 'rotation', 'teeth-hole contact', 'licking', 'peeling', 'chewing', 'chomping', 'digging', 'pushing', 'translational jaw movement', 'covering with soil', 'ear clapping'}


def turn_target_into_sh(df):
    df = df.copy()

    first_is_target = df.iloc[0]['Behavior'] == 'target'
    target_indices = df.index[df['Behavior'] == 'target']

    for idx in target_indices:
        # Skip the first 'target' if it's the first behavior
        if first_is_target and idx == df.index[0]:
            continue

        # Check if any behavior after this row is 'assessment'
        after_behaviors = df.loc[idx + 1:, 'Behavior']
        if 'assessing' in after_behaviors.values:
            continue  # Don't convert this 'target'

        # Convert to 'scatter-hoarding' if no 'assessment' follows
        df.at[idx, 'Behavior'] = 'scatter-hoarding'

    return df


def handle_eating_chewing_alternance(df):
    df = df.reset_index(drop=True)
    to_drop = []
    i = 0

    while i < len(df) - 1:
        if df.loc[i, 'Behavior'] == 'eating':
            # Start of pattern
            start_idx = i
            chunk_indices = [i]

            # Continue collecting alternating pairs
            current = 'eating'
            while i + 1 < len(df):
                next_behavior = df.loc[i + 1, 'Behavior']
                if next_behavior == current:
                    chunk_indices.append(i + 1)
                    i += 1
                elif next_behavior in ['eating', 'chewing'] and next_behavior != current:
                    # switch behavior
                    current = next_behavior
                    chunk_indices.append(i + 1)
                    i += 1
                else:
                    break

            # Now check if chunk is strictly alternating in even-length pairs
            chunk_behaviors = df.loc[chunk_indices, 'Behavior'].tolist()
            if (
                len(chunk_behaviors) >= 6 and  # at least 3 pairs
                all(chunk_behaviors[k] == chunk_behaviors[k+1] for k in range(0, len(chunk_behaviors), 2))
            ):
                # Identify eating indices in chunk except first and last
                eating_in_chunk = [idx for idx in chunk_indices if df.loc[idx, 'Behavior'] == 'eating']
                if len(eating_in_chunk) >= 2:
                    to_drop.extend(eating_in_chunk[1:-1])
            else:
                i += 1
        else:
            i += 1

    df = df.drop(index=to_drop).reset_index(drop=True)
    return df


def get_active_intervals(df, behavior):
    """Return list of (start, stop) time tuples for a given behavior."""
    starts = df[(df['Behavior'] == behavior) & (df['Behavior type'] == 'START')]['Time'].tolist()
    stops = df[(df['Behavior'] == behavior) & (df['Behavior type'] == 'STOP')]['Time'].tolist()
    return list(zip(starts, stops))



def segment_times(times, step):
    segments = []
    for _, g in groupby(enumerate(times), lambda ix: round(ix[1] - ix[0] * step, 6)):
        group = list(g)
        segments.append((group[0][1], group[-1][1]))
    return segments
    

def add_chomping_rows(df, step=0.01):
    if 'eating' in np.unique(df['Behavior']):
        df = df.copy()
        time_col = 'Time'
        
        # Step 1: Get all intervals
        eating_intervals = get_active_intervals(df, 'eating')
        chewing_intervals = get_active_intervals(df, 'chewing')
        #print(eating_intervals)
        # Step 2: Build full time range using np.arange for fractional steps
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        timeline = np.arange(min_time, max_time + step, step)
                
        # Step 3: Create boolean mask of chomping = eating but not chewing
        def in_any_interval(t, intervals):
            return any(start <= t <= stop for start, stop in intervals)
        
        chomping_times = [t for t in timeline if in_any_interval(t, eating_intervals) and not in_any_interval(t, chewing_intervals)]
        
        min_chomp, max_chomp = min(chomping_times)+step, max(chomping_times)-step
    
        # Step 4: Add chomping START and STOP events
        segments = segment_times(chomping_times, step)
        chomping_rows = pd.DataFrame([
            {'Behavior': 'chomping', 'Behavior type': 'START', 'Time': start, 'Image index': None, 'Media file name': df['Media file name'][0]}
            for start, _ in segments
        ] + [
            {'Behavior': 'chomping', 'Behavior type': 'STOP', 'Time': stop, 'Image index': None, 'Media file name': df['Media file name'][0]}
            for _, stop in segments
        ])
        
        df = pd.concat([df, chomping_rows], ignore_index=True)
        df = df.sort_values('Time').reset_index(drop=True)
        #print(df[['Behavior', 'Behavior type', 'Time', 'Image index']])

    return df


def add_some_sh(df):
    df = df.copy()
    for sub_behav in ['digging', 'pushing', 'covering with soil']:
        if sub_behav in np.unique(df['Behavior']):
            sub_behav_interval = get_active_intervals(df, sub_behav)
            for interv in sub_behav_interval:
                start_sh = df[df['Time'] == interv[0]]
                end_sh = df[df['Time'] == interv[1]]
        
                start_sh['Behavior'] = 'scatter-hoarding'
                start_sh['Time'] = start_sh['Time']
                start_sh['Image index'] = start_sh['Image index']
        
                end_sh['Behavior'] = 'scatter-hoarding'
                end_sh['Time'] = end_sh['Time']
                end_sh['Image index'] = end_sh['Image index']
        
                df = pd.concat([df, start_sh, end_sh])
    return df


def remove_pause_overlaps(df):
    df = df.copy()

    if 'pause' in df['Behavior'].unique():
        pause_intervals = get_active_intervals(df, 'pause')

        for start_time, end_time in pause_intervals:
            # Get pause start/end indices
            pause_start_row = df[(df['Time'] == start_time) & (df['Behavior'] == 'pause')]
            pause_start_index = pause_start_row.index[0] if not pause_start_row.empty else None

            pause_end_row = df[(df['Time'] == end_time) & (df['Behavior'] == 'pause')]
            pause_end_index = pause_end_row.index[0] if not pause_end_row.empty else None

            # Only proceed if valid indices were found
            if pause_start_index is not None and pause_start_index > 0:
                row_before = df.loc[[pause_start_index - 1]].copy()
                row_before['Behavior type'] = 'STOP'
                row_before['Time'] = start_time - 0.01
                row_before['Image index'] = pause_start_row['Image index'].values[0] - 1
                df = pd.concat([df, row_before])

            if pause_end_index is not None and pause_end_index < df.index.max():
                try:
                    row_after = df.loc[[pause_end_index + 1]].copy()
                    row_after['Behavior type'] = 'START'
                    row_after['Time'] = end_time + 0.01
                    row_after['Image index'] = pause_end_row['Image index'].values[0] + 1
                    df = pd.concat([df, row_after])
                except KeyError:
                    pass  # skip if pause ends at the very end of the dataframe

        # Sort again after all modifications
        df = df.sort_values(by='Time').reset_index(drop=True)

    return df


def add_rotation_modifier(df):
    # Only modify rows where Behavior is 'rotation'
    mask = df['Behavior'] == 'rotation'
    if True in np.unique(mask):
    
        # Get the relevant modifiers and uncapitalize them
        modifiers = df.loc[mask, 'Modifier #1'].astype(str).str.lower()
        
        # Prepend 'rotation_' to each
        df.loc[mask, 'Behavior'] = modifiers + ' rotation'

    return df


def turn_last_peel_to_eat(df):
    if df['Behavior'].iloc[-1] == 'peeling':
        last_peeling_row = df.iloc[-1:]
        
        added_eat_start = last_peeling_row.copy()
        added_eat_stop = last_peeling_row.copy()
        
        added_eat_start['Behavior'] = 'eating'
        added_eat_start['Behavior type'] = 'START'
        added_eat_start['Time'] = last_peeling_row['Time'] - 0.1
        added_eat_start['Image index'] = last_peeling_row['Image index'] - 1

        added_eat_stop['Behavior'] = 'eating'
        
        df = pd.concat([df, added_eat_start, added_eat_stop])
    return df

def pool_translational_movements(df):
    for row in range(df.shape[0]):
        if df.iloc[row, 10] == 'translational movement':
            df.iloc[row, 10] = 'translational jaw movement' 
    return df
        


def correct_for_go_pros(df):
    df = df.copy()

    peanut_map = {
        '0.3': 'three',
        '0.5': 'five',
        '0.7': 'seven',
        '1': 'one',
        '1.5': 'onehalf',
        '2': 'two',
        '2+': 'twoplus',
        'control': 'control'
    }

    name_indiv = df['Observation id'][0].split('_')[0]+'_'+df['Observation id'][0].split('_')[1]+df['Observation id'][0].split('_')[2]
    
    # Step 2: Extract the video name from rows starting with 'startpeanut'
    start_mask = df['Behavior'].str.startswith('startpeanut', na=False)
    df.loc[start_mask, 'video_name'] = df.loc[start_mask, 'Behavior'].str.replace('startpeanut', '', regex=False)

    df['video_id'] = start_mask.cumsum()
    
    # Step 3: Forward-fill video_name
    df['video_name'] = df['video_name'].ffill()
    df['video_name'] = [
    el.replace(',', '.') if pd.notna(el) else el 
    for el in df['video_name']
]
    
    # Step 4: Remove the rows that were used to define video names
    df = df[~start_mask].reset_index(drop=True)
    
    df['video_name'] = [
    el.replace(',', '.') if pd.notna(el) else el 
    for el in df['video_name']
]
    df['video_name'] = df['video_name'].map(peanut_map)
    
    df['Media file name'] = df.apply(
            lambda row: f"{name_indiv}_{row['video_id']:04d}_{row['video_name']}.mp4",
            axis=1
        )

    df = df.drop(['video_name', 'video_id'], axis=1)

    return df
    

def apply_teeth_contact_expansion(df):
    """
    Replaces each 'teeth-hole contact' event with a sequence: 'jaw movement' followed 
    by 'teeth-hole contact'. This triggers a transition from jaw movement to contact.
    Works for both 'START'/'STOP' intervals and 'POINT' events.
    """
    if 'teeth-hole contact' not in df['Behavior'].values:
        return df

    df = df.copy()
    new_rows = []
    
    # Process each row that is a teeth-hole contact
    # We target START (for intervals) and POINT (for single events)
    mask = (df['Behavior'] == 'teeth-hole contact') & (df['Behavior type'].isin(['START', 'POINT']))
    
    for idx, row in df[mask].iterrows():
        # Create 'jaw movement' version of this row
        jm_row = row.copy()
        jm_row['Behavior'] = 'jaw movement'
        
        if row['Behavior type'] == 'START':
            # For intervals: Jaw movement START (orig time) -> STOP (tiny bit later) -> Contact START (shifted later)
            # Use 0.15s offset to be very safe for 0.1s sampling bins in reconstruct_timelines
            jm_stop = row.copy()
            jm_stop['Behavior'] = 'jaw movement'
            jm_stop['Behavior type'] = 'STOP'
            jm_stop['Time'] = row['Time'] + 0.14  # JM ends just before Contact starts
            
            # Shift the original contact START so it follows the JM sequence by more than one 'step'
            df.at[idx, 'Time'] = row['Time'] + 0.15
            
            new_rows.append(jm_row) # jm_start
            new_rows.append(jm_stop)
        else:
            # For POINT events: Insert JM point 0.15s before original contact
            # This ensures they fall into different 0.1s bins for transition analysis
            jm_row['Time'] = row['Time'] - 0.15
            new_rows.append(jm_row)
        
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values(by=['Time', 'Behavior type'], ascending=[True, False]).reset_index(drop=True)
        
    return df


def process_data(ethogram_df, expand_teeth_contact=False):
    ethogram_df = ethogram_df.copy()  # Avoid modifying original df

    if len(np.unique(ethogram_df['Media file name'])) <= 2:
        ethogram_df = correct_for_go_pros(ethogram_df)        
        
    processed_chunks = []

    for vid, group in ethogram_df.groupby('Media file name'):
        sub_df = group.copy()
        sub_df = turn_target_into_sh(sub_df)
        sub_df = handle_eating_chewing_alternance(sub_df)
        sub_df = add_chomping_rows(sub_df)
        sub_df = add_some_sh(sub_df)
        sub_df = remove_pause_overlaps(sub_df)
        sub_df = add_rotation_modifier(sub_df)  
        sub_df = turn_last_peel_to_eat(sub_df)
        sub_df = pool_translational_movements(sub_df)
        
        if expand_teeth_contact:
            sub_df = apply_teeth_contact_expansion(sub_df)
            
        processed_chunks.append(sub_df)

    processed_df = pd.concat(processed_chunks).sort_index().reset_index(drop=True)
    processed_df['Media file name'] = processed_df['Media file name'].apply(format_video_name)

    # Sort key for correct order
    processed_df['sort_key'] = processed_df['Media file name'].apply(
        lambda x: int(os.path.splitext(x)[0].split('_')[2])
    )
        
    processed_df = processed_df.sort_values(by='sort_key').reset_index(drop=True)
    processed_df = processed_df.drop(columns='sort_key')

    return processed_df


def import_ethograms(ethogram_folder, exp_name, expand_teeth_contact=False):
    dict_exp = {}

    if exp_name == 'ALL':
        for filename in os.listdir(ethogram_folder):
            path_exp = os.path.join(ethogram_folder, filename)
            if not filename.endswith('.csv'):
                continue
            for sep in [',', ';', '\t']:
                try:
                    import_exp = pd.read_csv(path_exp, sep=sep)
                    if len(import_exp.columns) >= 2:
                        dict_exp[filename.rsplit('.', 1)[0]] = import_exp
                        break  # Valid separator found, stop trying others
                except Exception:
                    continue
    else:
        path_exp = os.path.join(ethogram_folder, exp_name + '.csv')
        for sep in [',', ';', '\t']:
            try:
                import_exp = pd.read_csv(path_exp, sep=sep)
                if len(import_exp.columns) >= 2:
                    dict_exp[exp_name] = import_exp
                    break
            except Exception:
                continue

    dict_ethograms = {}
    for name_exp, ethogram in dict_exp.items():
        #if name_exp not in ['baby_1405', 'Boit_1205', 'dest_1405', 'griz_1305', 'Pepp_1305', 'Side_1205', 'Side_1305', 'whit_1305', 'whit_1405']:   # these are the ones witht he correct rotation annotations
        print(f"Processing: {name_exp}")
        ethogram_processed = process_data(ethogram, expand_teeth_contact=expand_teeth_contact)

        # Reinitialize timelines per experiment
        timelines = {}
        for exclusion in ['level1', 'level2', None]:
            key = str(exclusion) if exclusion is not None else 'None'
            timelines[key] = reconstruct_timelines(
                ethogram_processed,
                exclude_behaviors=exclusion,
                fill_gaps_with_last=True
            )

        dict_ethograms[name_exp] = {
            'processed': ethogram_processed,
            'timelines': timelines
        }

    return dict_ethograms

    



    








        





def extract_timeline_data(df, color_dict, level_1_behaviors):
    """
    Extract aligned behavioral timeline data for one video,
    aligned to assessing start = 0s and ending at assessing stop.
    """
    df = df.sort_values(by='Time').reset_index(drop=True)

    segments = {1: [], 2: []}
    point_events = {1: [], 2: []}
    active_behaviors = {}

    for _, row in df.iterrows():
        b, t, btype = row['Behavior'], row['Time'], row['Behavior type']
        if btype == 'START':
            active_behaviors[b] = t
        elif btype == 'STOP' and b in active_behaviors:
            start_time = active_behaviors.pop(b)
            level = 1 if b in level_1_behaviors else 2
            segments[level].append((b, start_time, t))
        elif btype == 'POINT':
            level = 1 if b in level_1_behaviors else 2
            point_events[level].append((b, t))

    # find assessing window
    assessing = [[s, e] for b, s, e in segments[1] if b == 'assessing']
    if not assessing:
        return None

    start_real, end_real = assessing[0]
    duration = end_real - start_real

    # restrict to events that fall inside assessing window
    aligned_segments = []
    for b, s, e in segments[2]:
        if b == 'chewing':
            continue
        if e < start_real or s > end_real:
            continue
        s_clipped = max(s, start_real)
        e_clipped = min(e, end_real)
        aligned_segments.append((b, s_clipped - start_real, e_clipped - start_real))

    aligned_points = [
        (b, t - start_real) for b, t in point_events[2]
        if start_real <= t <= end_real
    ]

    return {
        "segments": aligned_segments,
        "points": aligned_points,
        "duration": duration,
        "color_dict": color_dict
    }


def plot_multi_behavior_timelines(
    dict_ethograms,
    media_list,
    figsize=(12, 8),
    row_spacing=0.8,
    tick_spacing=2,
    show_points=True,
    font_size=13,
    clip_to_assessment=True,
    ordering=False,
    ordering_outcome=False,
    normalize_time=False,
    align_outcomes=None,
    align_end=False,  # NEW OPTION
    save_path=None,
    show=True
):
    """
    Plot all videos' behavior timelines aligned to assessing start = 0 by default.

    If normalize_time=True, each timeline is rescaled so assessing duration = 1.
    If align_end=True (and normalize_time=False), all rows are right-aligned
    so the end of assessing coincides, and x-axis shows seconds *before end*.
    """
    aligned = []
    longest_duration = 0
    rows = []

    # === Gather aligned data ===
    for media_file in media_list:
        mapping = {
            'griz_1905bloc2': 'griz_1905_bloc2',
            'repo_1905bloc0': 'repo_1905_bloc0',
            'repo_1905bloc1': 'repo_1905_bloc1',
            'repo_1905bloc2': 'repo_1905_bloc2',
            'repo_1905bloc2bis': 'repo_1905_bloc2bis',
            'griz_1905bloc1': 'griz_1905_bloc1',
            'ligh_1605bloc1': 'ligh_1605_bloc1',
            'griz_1905bloc3': 'griz_1905_bloc3'
        }
        exp_name = next((v for k, v in mapping.items() if k in media_file), None)
        if not exp_name:
                base = media_file.split('_00')[0]
                exp_name = base[0].lower() + base[1:]
                if '(1)' in media_file:
                    exp_name += '_bloc1'
        if exp_name == 'side_1305':
            exp_name = 'Side_1305'

        df = dict_ethograms[exp_name]['processed']
        sub = df[df['Media file name'] == media_file]

        # Detect outcome type and color
        behavs = [str(b).lower() for b in np.unique(sub['Behavior'].astype(str))]
        if any('scatter' in b for b in behavs):
            outcome = 'scatter'
            outcome_color = '#FFB722'
        elif any('eat' in b for b in behavs):
            outcome = 'eat'
            outcome_color = '#1A897B'
        else:
            outcome = 'unknown'
            outcome_color = 'gray'

        data = extract_timeline_data(sub, color_dict, level_1_behaviors)
        if not data:
            continue

        longest_duration = max(longest_duration, data['duration'])

        raw_type = media_file.split('_')[-1].split('.')[0].split('(1)')[0]
        peanut_type_mapping = {
            'control': 'control',
            'three': 0.3,
            'five': 0.5,
            'seven': 0.7,
            'one': 1,
            'onehalf': 1.5,
            'two': 2,
            'twoplus': '2+'
        }
        type_label = peanut_type_mapping.get(raw_type, np.nan)
        if pd.isna(type_label):
            continue

        rows.append({
            "media": media_file,
            "data": data,
            "raw_type": raw_type,
            "type_label": type_label,
            "outcome": outcome,
            "outcome_color": outcome_color
        })

    if not rows:
        print("No valid assessing intervals found.")
        return

    # === Ordering ===
    if ordering:
        type_order = ['control', 'three', 'five', 'seven', 'one', 'onehalf', 'two', 'twoplus']
        rows = [r for r in rows if r["raw_type"] in type_order]
        if ordering_outcome:
            def outcome_rank(o):
                if o == 'eat': return 0
                elif o == 'scatter': return 1
                else: return 2
            rows.sort(key=lambda r: (type_order.index(r["raw_type"]), outcome_rank(r["outcome"])))
        else:
            rows.sort(key=lambda r: type_order.index(r["raw_type"]))

    # === Figure setup ===
    n = len(rows)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    y_positions = np.arange(n)[::-1] * row_spacing
    bar_height = 0.97 * row_spacing

    # === Draw timelines ===
    for row, y in zip(rows, y_positions):
        data = row["data"]
        outcome_col = row["outcome_color"]
        max_dur = data["duration"]

         # === Grey background for assessment window ===
        start_time = 0 if not align_end else longest_duration - max_dur
        ax.barh(
            y - 0.05,                      # vertical position
            max_dur,                       # width = duration of assessment
            left=start_time,               # start at assessment start
            height=bar_height+0.01,             # same height as behaviors
            color="grey",             # background color
            alpha=0.3,                     # semi-transparent
            edgecolor="none",
            zorder=0                       # draw beneath everything else
        )
        
        # Normalization factor
        scale = 1 / max_dur if (normalize_time and max_dur > 0) else 1

        # Shift for align_end mode (only if not normalized)
        shift = longest_duration - max_dur if (align_end and not normalize_time) else 0

        # --- Draw duration segments ---
        for b, s, e in data["segments"]:
            s_plot = s * scale + shift
            e_plot = e * scale + shift

            if normalize_time:
                if s_plot > 1:
                    continue
                e_plot = min(e_plot, 1)
            else:
                if s_plot > max_dur + shift:
                    continue
                e_plot = min(e_plot, max_dur + shift)

            color, label = data['color_dict'].get(b, ['white', ''])
            if color != 'white':
                ax.barh(
                    y - 0.05,
                    e_plot - s_plot,
                    left=s_plot,
                    height=bar_height,
                    color=color,
                    edgecolor='grey',
                    linewidth=1
                )

        # --- Point events ---
        if show_points:
            for b, t in data["points"]:
                t_plot = t * scale + shift
                if normalize_time:
                    if 0 <= t_plot <= 1:
                        color, label = data['color_dict'].get(b, ['white', ''])
                        ax.vlines(t_plot, y - 0.3, y + 0.2, color=color, linewidth=2.5, alpha=1)
                else:
                    if 0 <= t_plot <= max_dur + shift:
                        color, label = data['color_dict'].get(b, ['white', ''])
                        ax.vlines(t_plot, y - 0.3, y + 0.2, color=color, linewidth=2.5, alpha=1)

        # --- Fading outcome rectangle ---
        if normalize_time:
            base_end_x = 1
        elif align_outcomes is not None:
            base_end_x = align_outcomes
        else:
            base_end_x = max_dur if not align_end else longest_duration

        end_x = base_end_x
        fade_width = (0.5 if normalize_time else 1)
        fade_steps = 200
        gradient = np.linspace(1, 0, fade_steps).reshape(1, -1)
        extent = [end_x, end_x + fade_width,
                  y - 0.05 - bar_height / 2,
                  y - 0.055 + bar_height / 2]

        from matplotlib.colors import to_rgba
        r, g, b, _ = to_rgba(outcome_col)
        rgba = np.zeros((1, fade_steps, 4))
        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = gradient
        ax.imshow(rgba, extent=extent, aspect='auto', origin='lower', zorder=2)

    # === Axis setup ===
    if normalize_time:
        ax.set_xlim(0, 1.5)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xticklabels([f"{t:.1f}" for t in np.arange(0, 1.1, 0.2)], fontsize=18)
        ax.set_xlabel("Normalized time (0 = start, 1 = end of assessing)", fontsize=20)

    elif align_end:
        # Axis flipped: 0 at end, increasing absolute seconds to the left
        max_extent = longest_duration
        ax.set_xlim(max_extent, 0)  # reverse axis
        tick_positions = np.arange(0, max_extent + 1e-9, tick_spacing)
        ax.set_xticks(max_extent - tick_positions)
        ax.set_xticklabels([str(int(t)) for t in tick_positions], fontsize=22)
        ax.set_xlabel("Time to decision (s)", fontsize=22)

    else:
        new_ticks = np.arange(0, longest_duration + 1e-9, tick_spacing)
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks], fontsize=22)
        ax.set_xlim(0, longest_duration)
        ax.set_xlabel("Time (s)", fontsize=22)

    # === Y-axis setup ===
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['' for _ in y_positions])
    ax.invert_yaxis()

    ax.set_title("Assessment ethogram", fontsize=22)
    sns.despine(ax=ax, top=True, right=True, left=False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', width=2, length=6, direction='out', color='black')
    ax.tick_params(axis='y', length=6, direction='out', color='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set_ylim(y_positions[-1] - 0.5, y_positions[0] + 0.8)

    # === Grouped vertical peanut-type labels ===
    from collections import defaultdict
    groups = defaultdict(list)
    for y, row in zip(y_positions, rows):
        groups[row["type_label"]].append(y)

    for label, ys in groups.items():
        y_top, y_bottom = max(ys), min(ys)
        x_span = (1.5 if normalize_time else longest_duration)

        if align_end:
            # Move to the RIGHT side
            x_pos = x_span + 0.16 * x_span
            text_offset = 0.02 * x_span
            ax.vlines(x_pos,
                      y_bottom - row_spacing / 2 + 0.15,
                      y_top + row_spacing / 2 - 0.15,
                      color='black', linewidth=2.5, clip_on=False)
            ax.text(x_pos + text_offset,
                    (y_top + y_bottom) / 2,
                    label,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=22)
        else:
            # Default: labels on the LEFT
            x_pos = -0.01 * x_span
            ax.vlines(x_pos,
                      y_bottom - row_spacing / 2 + 0.15,
                      y_top + row_spacing / 2 - 0.15,
                      color='black', linewidth=2.5, clip_on=False)
            ax.text(x_pos - 0.01 * x_span,
                    (y_top + y_bottom) / 2,
                    label,
                    rotation=90,
                    va='center', ha='center',
                    fontsize=22)

    # === Final adjustments ===
    if normalize_time:
        zero_pos = 0
    else:
        zero_pos = longest_duration
    
    plt.xlim(-0.5, zero_pos + 1)

    ax.axvline(
    x=zero_pos,
    color='black',
    linestyle='--',
    linewidth=2,
    alpha=0.9,
    zorder=5
    )
    
    plt.grid(False)
    plt.tick_params(axis='both', width=1.5, length=5, color='black', direction='out')
    ax.tick_params(which='both', bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True
        )
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return rows
    

def reconstruct_timelines(df, exclude_behaviors=None, step=0.1, fill_gaps_with_last=False):
    #level_1_behaviors = {'assessing', 'bending', 'explore', 'eating', 'grabbing peanut', 'mouth', 'target', 'scatter-hoarding'}
    #level_2_behaviors = {'jaw movement', 'sniffing', 'rotation', 'teeth-hole contact', 'licking', 'peeling', 'chewing', 'chomping'}

    # Interpret exclusion argument
    if exclude_behaviors == 'level1':
        exclude_behaviors = level_1_behaviors
    elif exclude_behaviors == 'level2':
        exclude_behaviors = level_2_behaviors
    elif exclude_behaviors is None:
        exclude_behaviors = set()
    else:
        exclude_behaviors = set(exclude_behaviors)

    all_segments = []

    for media_file, group in df.groupby('Media file name'):
        group = group[~group['Behavior'].isin(exclude_behaviors)]
        if not group.empty:
            group = group.sort_values('Time').reset_index(drop=True)
    
            timeline = []
            active = {}  # Active behaviors
            last_behavior = 'no behavior'
            min_time = group['Time'].min()
            max_time = group['Time'].max()
            time_points = np.arange(min_time, max_time + step, step)
                
            # Separate POINT events for quick lookup
            point_events = group[group['Behavior type'] == 'POINT']
    
            for t in time_points:
                current_events = group[np.isclose(group['Time'], t, atol=step/2)]
    
                for _, row in current_events.iterrows():
                    behavior = row['Behavior']
                    btype = row['Behavior type']
                    image_index = row['Image index']
                    
                    if btype == 'START':
                        active[behavior] = True
                    elif btype == 'STOP' and behavior in active:
                        del active[behavior]
    
                # Check for POINT event at this time
                point_match = point_events[np.isclose(point_events['Time'], t, atol=step/2)]
                if not point_match.empty:
                    label = point_match['Behavior'].iloc[-1]
                    last_behavior = label
                else:
                    current_behavior = list(active.keys())
                    if current_behavior:
                        label = current_behavior[-1]
                        last_behavior = label
                    else:
                        label = last_behavior if fill_gaps_with_last else 'no behavior'

                timeline.append({'Media file name': media_file, 'Time': round(t, 3), 'Behavior': label, 'Image index': image_index})

            all_segments.append(pd.DataFrame(timeline))

    return pd.concat(all_segments).reset_index(drop=True)


        






    



def behaviors_between_first_and_last_assessing(df, time_col='Time', behavior_col='Behavior'):
    """
    Returns behaviors that occur between the first and last assessing block.

    Parameters:
    - df: DataFrame with time and behavior columns.
    - time_col: name of the time column.
    - behavior_col: name of the behavior column.

    Returns:
    - List of behaviors (excluding 'assessing') that occur between the first and last assessing blocks.
    """
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

    if 'assessing' not in df_sorted[behavior_col].values:
        return []

    # Get indices of all 'assessing' rows
    assessing_idx = df_sorted[df_sorted[behavior_col] == 'assessing'].index.to_list()
    if not assessing_idx:
        return []

    # Group assessing indices into contiguous blocks
    assessing_blocks = []
    block = [assessing_idx[0]]
    for i in range(1, len(assessing_idx)):
        if assessing_idx[i] == assessing_idx[i - 1] + 1:
            block.append(assessing_idx[i])
        else:
            assessing_blocks.append(block)
            block = [assessing_idx[i]]
    assessing_blocks.append(block)

    # If fewer than 2 blocks, nothing to do
    if len(assessing_blocks) < 2:
        return []

    # Get the end of the first and start of the last block
    end_of_first_block = assessing_blocks[0][-1]
    start_of_last_block = assessing_blocks[-1][0]

    # Extract behaviors in between
    in_between = df_sorted.iloc[end_of_first_block + 1 : start_of_last_block]
    behaviors = in_between[behavior_col]
    behaviors = behaviors[behaviors != 'assessing'].tolist()

    return behaviors


def compute_transition_probabilities(timeline_dict, 
                                     name_exp, 
                                     exclude_behavior,
                                     min_count, 
                                     show_matrix=True, 
                                     self_transitions=True,
                                     only_assessing = False,
                                     no_display_behaviors = None,
                                     pull_rotations = True,
                                     by_condition = False,
                                     by_session = False,
                                     include_outcome = False,
                                    add_grabbing = False):
    """
    Compute transition probabilities from ethogram timeline DataFrames.
    
    Args:
        timeline_dict (dict): dict[name_exp]['timelines'][exclude_behavior] -> DataFrame
        name_exp (str): Specific experiment to compute ('ALL' for aggregate without cross-exp transitions)
        exclude_behavior (str/list): Key for timeline variant OR a custom list for on-the-fly collapse
        min_count (int): Minimum number of total transitions from a behavior to include
        show_matrix (bool): Whether to plot the matrix
        self_transitions (bool): Whether to include self-transitions (A → A)
        by_condition (bool): Group results by experimental condition (parsed from filename)
        by_session (bool): Group results by experiment session name
        include_outcome (bool): Append final trial outcome (eating/hoarding) to sequences

    Returns:
        prob_matrix (pd.DataFrame or dict): Normalized probabilities OR nested dicts based on flags
    """
    if exclude_behavior is None:
        exclude_behavior = 'None'
    
    # Helper to parse condition from filename
    def _get_cond_name(media_file):
        filename = os.path.basename(media_file)
        cond_name = os.path.splitext(filename)[0].split('_')[-1]
        
        # Standardize known condition variations
        cond_map = {
            'seven(1)': 'seven', 'five(1)': 'five', 'five (2)': 'five', 'five(2)': 'five',
            'three(1)': 'three', 'three(2)': 'three', 'two(1)': 'two', 'two(2)': 'two',
            'one(1)': 'one', 'one (2)': 'one', 'one(2)': 'one',
            'onehalf(1)': 'onehalf', 'onehalf(2)': 'onehalf'
        }
        cond_name = cond_map.get(cond_name, cond_name)
        return cond_name if cond_name not in ['twoplus', 'nan'] else None

    # Helper to process a list of (from, to) transitions into a matrix
    def _process_to_matrix(transitions, current_name):
        if not transitions:
            return pd.DataFrame()
            
        transition_df = pd.DataFrame(transitions, columns=["From", "To"])

        if pull_rotations:
            puller = {'horizontal rotation': 'rotation', 'vertical rotation': 'rotation', 
                      'yaw rotation': 'rotation', 'translational jaw movement': 'rotation'}
            for col in transition_df.columns:
                transition_df[col] = transition_df[col].map(lambda x: puller.get(x, x))

        if no_display_behaviors:
            transition_df = transition_df[~transition_df["From"].isin(no_display_behaviors) & 
                                          ~transition_df["To"].isin(no_display_behaviors)]
            
        if transition_df.empty:
            return pd.DataFrame()
        
        count_matrix = transition_df.pivot_table(index="From", columns="To", aggfunc='size', fill_value=0)
        for target in ["no behavior"]:
            if target in count_matrix.index: count_matrix.drop(index=target, inplace=True)
            if target in count_matrix.columns: count_matrix.drop(columns=target, inplace=True)

        row_sums = count_matrix.sum(axis=1)
        filtered_matrix = count_matrix.loc[row_sums >= min_count]

        if filtered_matrix.empty:
            return pd.DataFrame()

        prob_matrix = filtered_matrix.div(filtered_matrix.sum(axis=1), axis=0).fillna(0).round(3)

        # Plotting (only for aggregate if by_session/by_condition are False to avoid plot flooding)
        if show_matrix and not (by_condition or by_session):
            plt.figure(figsize=(12, 10))
            sns.heatmap(prob_matrix, annot=True, cmap="Blues", fmt=".2f", cbar=True)
            #plt.title(f"Transition Matrix: {current_name}\n(min_count ≥ {min_count}, exclude={exclude_behavior})")
            plt.ylabel("From Behavior"); plt.xlabel("To Behavior")
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout()
            output_file = rf"figures/transition_matrix_parametrics.svg"
            plt.savefig(output_file, format='svg', bbox_inches='tight')
            plt.show()
            
        return prob_matrix

    # 1. Collect all transitions into a grouped map: transitions_map[session][condition]
    selected_exps = [name_exp] if name_exp != 'ALL' else list(timeline_dict.keys())
    transitions_map = {}

    for exp in selected_exps:
        # Get/Reconstruct dataframe
        if isinstance(exclude_behavior, (list, set)):
            df = reconstruct_timelines(timeline_dict[exp]['processed'], 
                                       exclude_behaviors=exclude_behavior, fill_gaps_with_last=True)
        else:
            if exclude_behavior not in timeline_dict[exp]['timelines']:
                if name_exp == 'ALL': continue
                else: raise ValueError(f"'{exclude_behavior}' not in timelines of '{exp}'.")
            df = timeline_dict[exp]['timelines'][exclude_behavior].copy()

        sess_key = exp if by_session else 'ALL_SESSIONS'
        if sess_key not in transitions_map:
            transitions_map[sess_key] = {}

        for media_file, group in df.groupby("Media file name"):
            cond_key = _get_cond_name(media_file) if by_condition else 'ALL_CONDITIONS'
            if cond_key is None: continue
            
            if cond_key not in transitions_map[sess_key]:
                transitions_map[sess_key][cond_key] = []
                
            group = group.sort_values("Time")
            behaviors = behaviors_between_first_and_last_assessing(group) if only_assessing else group["Behavior"].tolist()
            
            if include_outcome:
                counter = Counter(group['Behavior'])
                outcome = 'scatter-hoarding' if 'scatter-hoarding' in counter else ('eating' if 'eating' in counter else None)
                if outcome: behaviors.append(outcome)

            if add_grabbing == True:
                behaviors.insert(0, 'grabbing peanut')
            
            transitions = list(zip(behaviors[:-1], behaviors[1:]))
            if not self_transitions:
                transitions = [(a, b) for a, b in transitions if a != b]
            
            transitions_map[sess_key][cond_key].extend(transitions)

    # 2. Process collected transitions into final output structure
    if by_session and by_condition:
        return {s: {c: _process_to_matrix(t, f"{s}_{c}") for c, t in c_map.items()} 
                for s, c_map in transitions_map.items()}
    elif by_session:
        # Merge all condition lists within each session
        return {s: _process_to_matrix([item for sublist in c_map.values() for item in sublist], s) 
                for s, c_map in transitions_map.items()}
    elif by_condition:
        # Merge same condition lists across all sessions
        cond_merged = {}
        for c_map in transitions_map.values():
            for c, t in c_map.items():
                if c not in cond_merged: cond_merged[c] = []
                cond_merged[c].extend(t)
        return {c: _process_to_matrix(t, c) for c, t in cond_merged.items()}
    else:
        # Single aggregate matrix
        all_trans = [item for s_map in transitions_map.values() for c_map in s_map.values() for item in c_map]
        return _process_to_matrix(all_trans, name_exp)






















grid_pos_level1 = {
    'target': (1, 7),
    'explore': (3, 7),
    'bending': (2, 6),
    'grabbing peanut': (2, 5),
    'lifting': (2, 4),
    'assessing': (2, 3),
    'mouth': (1, 2),
    'scatter-hoarding': (1, 1),
    'eating': (3, 1.5),
    'dropping peanut': (0.5, 0),
    'pause': (1.5, 0)
}


grid_pos_level2 = {
    'jaw movement': (0, 0),
    'sniffing': (-1, -1),
    'rotation': (-1, 1),
    'horizontal rotation': (1.85, 5),
    'vertical rotation': (1.7, 5),
    'yaw rotation': (1.5, 5),
    'teeth-hole contact': (0, -2),
    'licking': (1, 1),
    'translational jaw movement': (1, -1),
    'digging': (1, 4),
    'pushing': (1.3, 3),
    'covering with soil': (0.7, 3),
    
    'peeling': (1.85, 1),
    'chewing': (1.55, 0),
    'chomping': (2.15, 0),

    'ear clapping': (2.5,1.5)
}








    #return counts, bin_edges



    

def delay_a_post_b(df, a, b):
    """
    For each 'rotation' block followed by a 'jaw movement' (before another 'rotation'),
    return the block's start index and the delay (in rows) to the next 'jaw movement'.
    If no jaw movement occurs before the next 'rotation' or end, delay is None.

    Parameters:
    - df: DataFrame with a behavior column.

    Returns:
    - List of tuples: (start_index_of_rotation_block, delay_to_jaw_movement or None)
    """
    behavior_series = df.reset_index(drop=True)
    results = []
    i = 0

    while i < len(behavior_series):
        if behavior_series[i] == b:
            start_idx = i

            # Move to end of the current rotation block
            while i < len(behavior_series) and behavior_series[i] == b:
                i += 1
            end_idx = i  # first index after rotation block

            # Look ahead for next 'jaw movement' before another 'rotation'
            delay = None
            j = end_idx
            while j < len(behavior_series):
                if behavior_series[j] == b:
                    break  # next rotation block begins
                if behavior_series[j] == a:
                    delay = j - end_idx
                    break
                j += 1

            results.append(delay)
        else:
            i += 1

    return results


def plot_jaw_delay_probabilities(delay_list, max_delay=12, plot_type='bar', metric = 'jaw movement'):
    """
    Plots the probability of jaw movement occurring at each delay step after a rotation.

    Parameters:
    - delay_list (list): List of delays (integers) or None (if no jaw movement followed).
    - max_delay (int): Maximum delay (in steps) to include on the x-axis.
    - plot_type (str): 'bar' or 'line' to control plot style.
    """
    delay_list = [el + 1 if el is not None else None for el in delay_list]
    # Filter valid delays and count them
    valid_delays = [d for d in delay_list if d is not None]
    total = len(delay_list)  # Includes rotations with no jaw movement

    delay_counts = Counter(valid_delays)

    x_vals = list(range(1, max_delay + 1))
    probabilities = [delay_counts.get(d, 0) / total for d in x_vals]

    # --- PLOT ---
    plt.figure(figsize=(8, 5))

    if plot_type == 'bar':
        plt.bar(
            x_vals, 
            probabilities, 
            color='grey', 
            edgecolor='black',
            linewidth=1.5
        )
    elif plot_type == 'line':
        plt.plot(
            x_vals, 
            probabilities, 
            marker='o', 
            linestyle='-', 
            color='black', 
            linewidth=2,
            markersize=8
        )
    else:
        raise ValueError("plot_type must be either 'bar' or 'line'")

    # --- STYLE ---
    #plt.title("Probability of Jaw Movement per Delay Step", fontsize=20, pad=15)
    plt.xlabel("Number of behaviors post-rotation", fontsize=18, labelpad=10)
    plt.ylabel(f"Probability of {metric}", fontsize=18, labelpad=10)

    #plt.axvline(0, linestyle = '--', color = 'red', linewidth = 4)
    plt.xticks(x_vals, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylim(0, 1)

    # Remove grid
    plt.grid(False)

    ax = plt.gca()

    # Thicker axes
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color("black")

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick marks
    ax.tick_params(axis='x', width=2, length=6, direction='out', color='black')
    ax.tick_params(axis='y', width=2, length=6, direction='out', color='black')

    plt.gca().tick_params(which='both', bottom=True, left=True)

    plt.tight_layout()
    plt.show()


def plot_jaw_delay_after_rotation(dict_ethograms, metric, plotting = False):
    rotation_puller = {'horizontal rotation': 'rotation', 'vertical rotation': 'rotation', 'yaw rotation': 'rotation'}
    
    jaw_movement_delay = []
    for id_sess in dict_ethograms.keys():
        level1_timeline = dict_ethograms[id_sess]['timelines']['level1']
        for vid in np.unique(level1_timeline['Media file name']):
            vid_df = level1_timeline[level1_timeline['Media file name'] == vid]
            vid_df = vid_df['Behavior'].map(lambda x: rotation_puller.get(x, x))
            jaw_movement_delay.append(delay_a_post_b(vid_df, metric, 'rotation'))
    
    jaw_movement_delay = [item for sublist in jaw_movement_delay for item in sublist]
    
    if plotting == True:
        plot_jaw_delay_probabilities(jaw_movement_delay, metric = metric)
    else: 
        return jaw_movement_delay


def plot_overlayed_delay_probabilities(
    delays_dict,
    max_delay=12,
    order=['jaw movement', 'licking', 'sniffing'],
    alpha=0.8,
    save_path=None,
    show=True
):

    if order is None:
        order = list(delays_dict.keys())

    colors = {beh: color_dict1[beh][0] for beh in order}

    plt.figure(figsize=(10, 6))
    all_probabilities = {}
    all_counts = {}
    all_totals = {}

    for behavior in order:
        delay_list = delays_dict[behavior]

        # --- SAME PROCESSING ---
        delay_list = [el + 1 if el is not None else None for el in delay_list]
        valid_delays = [d for d in delay_list if d is not None]
        total = len(delay_list)

        delay_counts = Counter(valid_delays)

        x_vals = list(range(1, max_delay + 1))
        probabilities = [delay_counts.get(d, 0) / total for d in x_vals]

        # --- OVERLAID BARS ---
        plt.bar(
            x_vals,
            probabilities,
            color=colors.get(behavior, None),
            edgecolor='black',
            linewidth=1.5,
            alpha=alpha,
            label=behavior
        )
        all_probabilities[behavior] = probabilities
        all_counts[behavior] = delay_counts
        all_totals[behavior] = total

    # --- Kruskal-Wallis Global Test ---
    list_of_valid_delays = [ [d for d in delays_dict[beh] if d is not None] for beh in order ]
    # Filters out groups with no data to avoid errors
    list_of_valid_delays = [group for group in list_of_valid_delays if len(group) > 0]
    
    if len(list_of_valid_delays) > 1:
        stat, pval = kruskal(*list_of_valid_delays)
        print(f"\n--- Statistical Test: Kruskal-Wallis ---")
        print(f"H-statistic: {stat:.3f}, p-value: {pval:.4g}")
        ax = plt.gca()
        ax.set_title(f"Kruskal-Wallis: p={pval:.3g}", fontsize=18, fontweight='bold')
        
        # --- Post-hoc Pairwise comparisons ---
        if pval < 0.05 and len(order) > 1:
            from itertools import combinations
            from scipy.stats import mannwhitneyu
            from statsmodels.stats.multitest import multipletests
            
            p_raw = []
            pairs = []
            for (i, b1), (j, b2) in combinations(enumerate(order), 2):
                v1 = [d for d in delays_dict[b1] if d is not None]
                v2 = [d for d in delays_dict[b2] if d is not None]
                if len(v1) > 0 and len(v2) > 0:
                    _, p = mannwhitneyu(v1, v2, alternative='two-sided')
                    p_raw.append(p)
                    pairs.append((b1, b2))
            
            if p_raw:
                rejected, p_adj, _, _ = multipletests(p_raw, method='holm')
                print(f"--- Post-hoc Pairwise Comparisons (Holm-corrected MWU) ---")
                for (b1, b2), padj, rej in zip(pairs, p_adj, rejected):
                    print(f"  {b1} vs {b2}: p={padj:.4f} {'*' if rej else 'n.s.'}")

    # --- Pointwise Significance along X-axis ---
    p_bins = []
    for x in x_vals:
        obs = []
        for beh in order:
            c = all_counts[beh].get(x, 0)
            t = all_totals[beh]
            obs.append([c, t - c])
        
        # Rx2 Chi-square: must have non-zero column/row sums
        col_sum_success = sum(row[0] for row in obs)
        col_sum_failure = sum(row[1] for row in obs)
        row_sums = [sum(row) for row in obs]
        
        if col_sum_success > 0 and col_sum_failure > 0 and all(r > 0 for r in row_sums):
            try:
                _, p, _, _ = chi2_contingency(obs)
                p_bins.append(p)
            except ValueError:
                p_bins.append(1.0)
        else:
            p_bins.append(1.0)
            
    if p_bins:
        # Filter p_bins to remove 1.0 (non-tested bins) for fairer correction?
        # Standard to keep them if they were part of the hypothesis.
        rej_bins, padj_bins, _, _ = multipletests(p_bins, method='holm')
        for i, (rej, p) in enumerate(zip(rej_bins, padj_bins)):
            if rej:
                x_pos = x_vals[i]
                # Find max prob in this bin to place star
                max_p = max([all_probabilities[beh][i] for beh in order])
                plt.text(x_pos, max_p + 0.01, '*', ha='center', va='bottom', fontsize=20, color='red', fontweight='bold')

    # --- SAME STYLE ---
    plt.xlabel("Number of behaviors post-rotation", fontsize=18, labelpad=10)
    plt.ylabel("Probability", fontsize=18, labelpad=10)

    plt.xticks(x_vals, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1)

    plt.grid(False)

    ax = plt.gca()

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color("black")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='x', width=2, length=6, direction='out', color='black')
    ax.tick_params(axis='y', width=2, length=6, direction='out', color='black')

    plt.gca().tick_params(which='both', bottom=True, left=True)
    plt.legend(frameon=False, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True
        )
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    
def plot_combined_delays_after_rotation(dict_ethograms, save_path=None):
    dict_delays = {}
    for metric in ['jaw movement', 'sniffing', 'licking']:
        dict_delays[metric] = plot_jaw_delay_after_rotation(dict_ethograms, metric)
    plot_overlayed_delay_probabilities(dict_delays, save_path=save_path)
    


    
    





    



    






def prepare_behavior_df(output, behavior_to_remove):
    """
    Prepares feature matrix X and target vector y from raw output data.

    Parameters:
        output (list of dicts): Raw data.
        behavior_to_remove (str, list of str, or None): Behavior(s) to exclude from features.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Binary outcome (1 = scatter-hoarding, 0 = eating).
        df (pd.DataFrame): Cleaned DataFrame with all original columns.
    """

    def clean_behavior_list(beh_list):
        if not isinstance(beh_list, list):
            return []
        if behavior_to_remove is None:
            return beh_list
        # Convert single string to list for uniformity
        to_remove = [behavior_to_remove] if isinstance(behavior_to_remove, str) else behavior_to_remove
        return [b for b in beh_list if b not in to_remove]

    df = pd.DataFrame(output)

    # Binary outcome
    df['y'] = df['final_behavior'].map({'scatter-hoarding': 1, 'eating': 0})
    df = df.dropna(subset=['y'])

    # Handle 'previous outcome' if not excluded
    remove_prev = (
        (isinstance(behavior_to_remove, str) and behavior_to_remove == 'previous outcome') or
        (isinstance(behavior_to_remove, list) and 'previous outcome' in behavior_to_remove)
    )

    if not remove_prev:
        df['previous outcome'] = df['previous outcome'].map({'scatter-hoarding': 1, 'eating': 0})
        df = df.dropna(subset=['previous outcome'])
    else:
        df['previous outcome'] = None

    # Map peanut_type to numeric values
    peanut_map = {
        'three': 0.3,
        'five': 0.5,
        'seven': 0.7,
        'one': 1,
        'onehalf': 1.5,
        'two': 2,
        'control': 0
    }
    if 'peanut_type' not in behavior_to_remove:
        df['peanut_type'] = df['peanut_type'].map(peanut_map)
    
    # Drop rows with missing peanut_type if it's not excluded
    if 'peanut_type' not in behavior_to_remove:
        df = df.dropna(subset=['peanut_type'])

    # Drop rows with missing teeth double checked if specified
    if 'teeth from sheet' in behavior_to_remove:
        df = df.dropna(subset=['teeth double checked'])

    df = df.reset_index(drop=True)

    # Cleaned behavior list
    df['cleaned_behaviors_before'] = df['behaviors_before'].apply(clean_behavior_list)

    # Compute behavior counts
    behavior_counts = df['cleaned_behaviors_before'].apply(Counter)
    behavior_df = pd.DataFrame(behavior_counts.tolist()).fillna(0).astype(int).reset_index(drop=True)

    # Combine features
    timings = df[['assessing time', 'rotation time']].reset_index(drop=True)
    columns_to_concat = [behavior_df, timings]

    if 'teeth from sheet' in behavior_to_remove:
        columns_to_concat.append(df['teeth double checked'].reset_index(drop=True))

    if 'peanut_type' not in behavior_to_remove:
        peanut_col = df[['peanut_type']].reset_index(drop=True)
        columns_to_concat.append(peanut_col)

    if 'previous outcome' in df.columns and df['previous outcome'].notna().any():
        columns_to_concat.append(df['previous outcome'].reset_index(drop=True).to_frame())

    X = pd.concat(columns_to_concat, axis=1)

    y = df['y'].reset_index(drop=True)

    return X, y, df


import re

def extract_type_from_filename1(fname):
    """
    Extract peanut type robustly from filenames like:
    Baby_1405_0001_three.mp4
    Baby_1405_0001_seven(1).mp4
    Griz_1905_002_two+.mp4
    """
    base = fname.split('/')[-1].split('.')[0]
    parts = base.split('_')

    # peanut is last part
    raw = parts[-1]

    # remove (1)
    raw = re.sub(r'\(.*?\)', '', raw)

    # drop trailing numbers if any
    raw = re.sub(r'\d+$', '', raw)

    # map variations
    mapping = {
        'onehalf': 'onehalf',
        'one': 'one',
        'two': 'two',
        'twoplus': 'two',
        'five': 'five',
        'seven': 'seven',
        'three': 'three',
        'control': 'control'
    }

    raw = raw.lower().strip()
    return mapping.get(raw, raw)

    
def build_output(dict_ethograms):
    output = []

    rotation_mapping = {
        'horizontal rotation': 'rotation',
        'vertical rotation': 'rotation',
        'yaw rotation': 'rotation'
    }

    removal_list = set(['assessing'])  # remove assessing inside the window

    for id_ in dict_ethograms.keys():
        processed = dict_ethograms[id_]['processed']
        for file in np.unique(processed['Media file name']):

            #print(file)
            file_df = processed[processed['Media file name'] == file]
            assessing_intervals = get_active_intervals(file_df, 'assessing')

            # skip files with no assessing → desired behavior
            if not assessing_intervals:
                continue

            # build assess_df (handles multiple intervals)
            if len(assessing_intervals) == 1:
                t0, t1 = assessing_intervals[0]
                assess_df = file_df[(file_df['Time'] > t0) & (file_df['Time'] < t1)]
            else:
                dfs = []
                for t0, t1 in assessing_intervals:
                    dfs.append(file_df[(file_df['Time'] > t0) & (file_df['Time'] < t1)])
                assess_df = pd.concat(dfs)

            # remap rotations
            assess_df['Behavior'] = assess_df['Behavior'].replace(rotation_mapping)

            assess_df['Behavior'] = assess_df['Behavior'].replace({'translational jaw movement': 'jaw movement'})            

            # remove assessing from behaviors_before
            behaviors_before = []
            for b in assess_df['Behavior']:
                if isinstance(b, str) and b.lower() not in removal_list:
                    behaviors_before.append(b)

            # compute rotation duration
            rotation_intervals = get_active_intervals(assess_df, 'rotation')
            rotation_time = np.sum([end - start for start, end in rotation_intervals])

            # compute assess duration
            assessing_duration = np.sum([end - start for start, end in assessing_intervals])

            # final outcome
            cnt = Counter(file_df['Behavior'])
            if 'scatter-hoarding' in cnt:
                outcome = 'scatter-hoarding'
            elif 'eating' in cnt:
                outcome = 'eating'
            else:
                outcome = np.nan

            #print(outcome)
            output.append({
                'file name': file,
                'animal_id': id_,
                'peanut_type': extract_type_from_filename1(file),
                'behaviors_before': behaviors_before,
                'final_behavior': outcome,
                'assessing time': assessing_duration,
                'rotation time': rotation_time
            })

    return output
    


    
def plot_GLM_coeffs(result, output_file=None):
    """
    Plot GLM coefficients with 95% CI and significance stars.
    Clean aesthetic: grey tones, no grid, no top/right spines.
    """
    coefs = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    conf.columns = ['lower', 'upper']

    # Combine into summary DataFrame
    summary_df = (
        coefs.to_frame(name='coef')
        .join(conf)
        .join(pvals.to_frame(name='pval'))
        .drop(index='Intercept', errors='ignore')
    )

    # Sort by coefficient value (descending)
    summary_df = summary_df.sort_values('coef', ascending=False)

    # Compute asymmetric error bars
    yerr = np.array([
        summary_df['coef'] - summary_df['lower'],
        summary_df['upper'] - summary_df['coef']
    ])

    # --- Plot setup ---
    sns.set_style("white")
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    x = np.arange(len(summary_df))

    # Grey error bars and points
    ax.errorbar(
        x,
        summary_df['coef'],
        yerr=yerr,
        fmt='o',
        color='gray',
        ecolor='gray',
        elinewidth=2,
        capsize=5,
        markersize=7
    )

    # Baseline line at 0
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)

    # --- Significance stars ---
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
        
    stars = summary_df['pval'].apply(get_stars)
    for i, (y_val, star) in enumerate(zip(summary_df['coef'], stars)):
        if star:
            ax.text(
                x[i],
                y_val + yerr[1][i] + 0.02,
                star,
                ha='center',
                va='bottom',
                fontsize=20,
                fontweight='bold'
            )

    # (already defined above)
    behav_mapping = {'sniffing': 'Sniffing', 'Group Var': 'Group variation', 'assessing_time': 'Assessment time', 'rotation_time': 'Rotation time', 'rotation': 'Number of rotations',
       'licking': 'Licking', 'jaw_movement': 'Jaw clamping', 'teeth_double_checked': 'Teeth-hole contact', 'peanut_type': 'Hole size (cm)',
       'Group x peanut_type Cov': 'Group x Hole size Cov', 'peanut_type Var': 'Hole size variation'}

    print(summary_df.index)
    list_xlabels = [behav_mapping.get(b, b) for b in summary_df.index]
    
    # --- Styling ---
    ax.set_xticks(x)
    ax.set_xticklabels(list_xlabels, rotation=45, ha='right', fontsize=20)
    ax.set_ylabel("Coefficient", fontsize=20)
    ax.set_title("GLM Coefficient Estimates (95% CI)", fontsize=16, fontweight='bold')
    
    # Make y-axis labels & ticks bigger
    ax.tick_params(
        axis='y',
        labelsize=20,   # bigger font
        width=2,        # thicker ticks
        length=8        # longer ticks
    )
    
    # Make x-axis ticks thicker too (optional)
    ax.tick_params(
        axis='x',
        width=2,
        length=8
    )
    
    # Remove grid
    ax.grid(False)
    
    # Remove top/right spines
    sns.despine(ax=ax, top=True, right=True)
    
    # Thicken left/bottom spines
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.gca().tick_params(which='both', bottom=True, left=True)
    ax.margins(y=0.2)
    plt.subplots_adjust(bottom=0.2, right=0.95)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    plt.show()
    plt.close()




fruit_palette = {
    'grape': '#6D4E85',
    'banana': '#E0B751',
    'apple': '#B31A1A',
    'orange': '#DE6E09',
    'peanut': '#995240'
}
default_fruit_color = 'gray'

order_fruit = ['banana', 'orange', 'apple', 'grape', 'peanut']

def show_fruit_stats(df):
    df_fruit = df#df_fruit = df[df['Fruit'] != 'peanut']
    data = Counter(df_fruit['Fruit'])

    values = [data[item] for item in order_fruit]
    
    # Map colors
    colors = [fruit_palette.get(item, default_fruit_color) for item in order_fruit] 
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(order_fruit, values, color=colors)
    
    # Remove frame (top and right)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Keep only y and x axes, and make them thicker
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    # Force ticks to appear prominent (thicker and longer)
    ax.tick_params(axis='both', which='major', width=3, length=10, labelsize=20, bottom=True, left=True)
    
    # All fonts 20 size
    ax.set_xlabel("Fruit", fontsize=20, labelpad=15)
    ax.set_ylabel("Count", fontsize=20, labelpad=15)
    ax.set_title("Fruit Counts", fontsize=20, pad=20)
    
    plt.tight_layout()
    plt.show()


def compute_fruit_behav_prob(fruit_data, behav, correction='bonferroni', save_path = False):
    fruits_data = fruit_data.copy()
    
    if behav == 'unstem':
        fruits_data = fruits_data[fruits_data['Fruit'] != 'peanut']
        fruits_data = fruits_data[fruits_data['Fruit'] != 'orange']
        
    df = fruits_data[['Fruit', behav]].dropna()

    # Aggregate counts
    fruit_counts = df.groupby('Fruit')[behav].agg(['sum', 'count'])
    fruit_props = fruit_counts['sum'] / fruit_counts['count']
    dict_avg_behav = fruit_props.to_dict()

    # Pairwise statistical comparisons
    fruits = fruit_counts.index.tolist()
    pairs = list(itertools.combinations(fruits, 2))
    pvals = []
    tests = []
    stats = []

    for f1, f2 in pairs:
        count = [fruit_counts.loc[f1, 'sum'], fruit_counts.loc[f2, 'sum']]
        nobs = [fruit_counts.loc[f1, 'count'], fruit_counts.loc[f2, 'count']]
        stat, pval = proportions_ztest(count, nobs)
        pvals.append(pval)
        stats.append(stat)
        tests.append((f1, f2))

    # Multiple testing correction
    reject, pvals_corrected, _, _ = multipletests(pvals, method=correction)

    # Store significant results
    sig_results = {}
    for idx, (f1, f2) in enumerate(tests):
        if reject[idx]:
            sig_results[(f1, f2)] = pvals_corrected[idx]

    if save_path == True:
        save_path = r'figures/fruits_' + behav + '.svg'
    else:
        save_path = None
    show_fruit_behav_stats(dict_avg_behav, behav, sig_results, save_path)
    print('proportion z test:', sig_results, 'statistiques:', stats)



def show_fruit_behav_stats(dict_data, behav, sig_results, save_path = None):
    # Determine the fruits to show (exclude irrelevant ones for 'unstem')
    current_order = [f for f in order_fruit]
    if behav == 'unstem':
        current_order = [f for f in current_order if f not in ['peanut', 'orange']]

    values = [dict_data.get(item, 0) for item in current_order]
    colors = [fruit_palette.get(item, default_fruit_color) for item in current_order] 

    fig, ax = plt.subplots(figsize=(5, 6))
    bar_width = 0.9
    bars = ax.bar(current_order, values, color=colors, width=bar_width)

    ax.set_ylabel("Probability", fontsize=20)
    ax.set_xticks(range(len(current_order)))
    ax.set_xticklabels(current_order, fontsize=20, rotation=45, horizontalalignment='right')
    ax.tick_params(axis='y', labelsize=16)

    ax.tick_params(axis='both', which='major', width=2, length=10, labelsize=20, bottom=True, left=True)

    if behav == 'peel':
        title = 'Peeling'
    elif behav == 'rotate':
        title = 'Rotation'
    elif behav == 'transported':
        title = 'Transport'
    elif behav == 'scatter.hoarded':
        title = 'Scatter-hoarding'
    elif behav == 'eaten.lifted':
        title = 'Lifting'
    elif behav == 'unstem':
        title = 'Unsteming'
    ax.set_title(title, fontsize = 20) 
    
    # Remove top and right spines (frame)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Thicken x and y axis lines
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)

    # Add value labels above bars
    for i, val in enumerate(values):
        ax.text(i, val + 0.015, f"{val:.2f}", ha='center', va='bottom', fontsize=20)

    bar_locs = {fruit: i for i, fruit in enumerate(current_order)}

    # Fixed desired top-to-bottom order
    fixed_order = [
        ('banana', 'orange'), ('banana', 'apple'), ('banana', 'grape'), ('banana', 'peanut'),
        ('orange', 'apple'), ('orange', 'grape'), ('orange', 'peanut'),
        ('apple', 'grape'), ('apple', 'peanut'), ('grape', 'peanut')
    ]

    # Compute top height
    valid_sig_results = {pair: p for pair, p in sig_results.items() 
                        if pair[0] in current_order and pair[1] in current_order}
    
    max_val = max(values) if values else 1.0
    total_bars = len(valid_sig_results)
    top_height = max_val + 0.05 + total_bars * 0.05
    step = 0.05
    plotted = 0

    for pair in fixed_order:
        if pair not in valid_sig_results and tuple(reversed(pair)) not in valid_sig_results:
            continue

        pval = sig_results.get(pair, sig_results.get(tuple(reversed(pair))))
        x1, x2 = sorted([bar_locs[pair[0]], bar_locs[pair[1]]])
        h = top_height - step * plotted

        ax.plot([x1, x2], [h, h], lw=1.5, c='black')
        ax.plot([x1, x1], [h - 0.005, h], lw=1.5, c='black')
        ax.plot([x2, x2], [h - 0.005, h], lw=1.5, c='black')

        if pval < 0.001:
            star = '***'
        elif pval < 0.01:
            star = '**'
        elif pval < 0.05:
            star = '*'
        else:
            continue

        ax.text((x1 + x2) / 2, h-0.02, star, ha='center', va='bottom', fontsize=17)
        plotted += 1

    # Force y-axis labels to stop at 1.0 for probability
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=20)

    # Extend the plot area vertically to accommodate the significance brackets
    ax.set_ylim(0, max(1.1, top_height + 0.1))
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, 
            dpi=300,               # High resolution (perfect for publications)
            bbox_inches='tight',    # Ensures labels/titles aren't cut off
            transparent=True,      # Set to True if you want a transparent background
            facecolor='white'       # Ensures background is solid white
        )
        print(f"Plot saved to: {save_path}")
        
    plt.show()






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass # Mocked out  # pip install statannotations



def create_teeth_doublecheck(data_list):
    for entry in data_list:
        behaviors = entry.get("behaviors_before", [])
        teeth_from_sheet = entry.get("teeth from sheet", np.nan)

        if "teeth-hole contact" in behaviors:
            entry["teeth double checked"] = behaviors.count("teeth-hole contact")
        else:
            if teeth_from_sheet == 0:
                entry["teeth double checked"] = 0
            elif np.isnan(teeth_from_sheet):
                entry["teeth double checked"] = np.nan
            else:
                entry["teeth double checked"] = np.nan  # can adjust if you want a fallback
    return data_list





palette_peanut_conditions = {'intact': '#d79d72',
                             'hole': '#ad552e',
                             'crushed': '#715040'}













from scipy.special import digamma
import random










def plot_prob_contact(dict_ethograms, output_file=None):
    peanut_types = ['three', 'five', 'seven', 'one', 'onehalf', 'two']
    peanut_to_num = {'three': 0.3, 'five': 0.5, 'seven': 0.7, 'one': 1, 'onehalf': 1.5, 'two': 2}

    dict_contact = {}
    contact_list = []
    tot_list = []
    for peanut in peanut_types:
        n_contact = 0
        n_tot = 0
        for ID, processed in dict_ethograms.items():
            processed = processed['processed']
            for media in np.unique(processed['Media file name']):
                if peanut in media:
                    counter = dict(Counter(processed['Behavior'][processed['Media file name'] == media]))
                    if 'teeth-hole contact' in counter:
                        n_contact += 1
                    n_tot += 1
        dict_contact[peanut] = n_contact / n_tot if n_tot > 0 else np.nan
        contact_list.append(n_contact)
        tot_list.append(n_tot)

    # Build DataFrame with numeric x values
    df_bar = pd.DataFrame({
        'Peanut Type': list(dict_contact.keys()),
        'Value': list(dict_contact.values())
    })
    df_bar['x_num'] = df_bar['Peanut Type'].map(peanut_to_num)
    df_bar = df_bar.sort_values('x_num')

    # Plot using true numeric scale
    plt.figure(figsize=(6, 4))
    sns.set_style("white")

    ax = plt.gca()
    bar_width = 0.15
    plt.bar(df_bar['x_num'], df_bar['Value'], width=bar_width, color='grey', edgecolor='grey')

    font_size = 14

    # Labels and axes
    ax.set_xlabel('Peanut integrity', fontsize=font_size)
    ax.set_ylabel('Probability of teeth-hole contact', fontsize=font_size)

    # Real numeric ticks
    ax.set_xticks(df_bar['x_num'])
    ax.set_xticklabels([str(x) for x in df_bar['x_num']], fontsize=font_size)
    ax.set_xlim(0.15, 2.2)  # ✅ brings first bar closer to y-axis

    # Add numeric labels above bars
    for x, v in zip(df_bar['x_num'], df_bar['Value']):
        ax.text(x, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=font_size - 2)

    # Style and spines
    sns.despine(top=True, right=True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', width=1.5, length=5, color='black', direction='out')
    ax.tick_params(labelsize=font_size)
    plt.tick_params(axis='both', width=1.5, length=5, color='black', direction='out')
    ax.tick_params(which='both', bottom=True, left=True)
    
    # --- STATS ---
    # 1. Global Chi-square
    table = []
    for c, t in zip(contact_list, tot_list):
        if t > 0:
            table.append([c, t - c])
    
    p_global = 1.0
    if len(table) > 1:
        _, p_global, _, _ = chi2_contingency(table)
        print(f"\n--- Prob Contact Stats ---")
        print(f"Global Chi-square: p={p_global:.4g}")
    
    # 2. Cochran-Armitage Trend Test (via linear regression of proportions)
    # Using the numeric integrity values as scores
    x_scores = df_bar['x_num'].values
    y_props = df_bar['Value'].values
    # Filter out NaNs
    mask = ~np.isnan(y_props)
    p_trend = 1.0
    if sum(mask) > 1:
        slope, intercept, r_value, p_trend, std_err = linregress(x_scores[mask], y_props[mask])
        print(f"Trend Test (Regression): slope={slope:.3f}, p={p_trend:.4g}, R^2={r_value**2:.3f}")

    ax.set_title(f"Global: p={p_global:.3g} | Trend: p={p_trend:.3g}", fontsize=font_size-2, fontweight='bold')
    
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    
    plt.show()
    plt.close()


def plot_prob_contact_rotation(dict_ethograms, output_file=None):
    peanut_types = ['three', 'five', 'seven', 'one', 'onehalf', 'two']
    peanut_to_num = {'three': 0.3, 'five': 0.5, 'seven': 0.7, 'one': 1, 'onehalf': 1.5, 'two': 2}

    dict_contact_rotation = {}
    for peanut in peanut_types:
        n_contact_rotation = n_tot_rotation = 0
        n_contact_no = n_tot_no = 0
        for ID, processed in dict_ethograms.items():
            processed = processed['processed']
            for media in np.unique(processed['Media file name']):
                if peanut in media:
                    counter = dict(Counter(processed['Behavior'][processed['Media file name'] == media]))
                    has_contact = 'teeth-hole contact' in counter
                    has_rotation = any('rotation' in b for b in counter.keys())

                    if has_rotation:
                        n_tot_rotation += 1
                        if has_contact:
                            n_contact_rotation += 1
                    else:
                        n_tot_no += 1
                        if has_contact:
                            n_contact_no += 1

        dict_contact_rotation[peanut] = {
            'rotation': n_contact_rotation / n_tot_rotation if n_tot_rotation > 0 else np.nan,
            'No rotation': n_contact_no / n_tot_no if n_tot_no > 0 else np.nan,
            'n_rot': n_contact_rotation, 'tot_rot': n_tot_rotation,
            'n_no': n_contact_no, 'tot_no': n_tot_no
        }

    # Convert to DataFrame (long form)
    df_plot = (
        pd.DataFrame(dict_contact_rotation)
        .T
        .reset_index()
        .melt(id_vars='index', var_name='Condition', value_name='Value')
        .rename(columns={'index': 'Peanut Type'})
    )

    # Add numeric x positions
    df_plot['x_num'] = df_plot['Peanut Type'].map(peanut_to_num)

    # Define order
    order = ['three', 'five', 'seven', 'one', 'onehalf', 'two']
    df_plot = df_plot[df_plot['Peanut Type'].isin(order)]
    df_plot['Peanut Type'] = pd.Categorical(df_plot['Peanut Type'], categories=order, ordered=True)
    df_plot = df_plot.sort_values('x_num')

    # Create numeric positions for grouped bars
    offset = 0.05
    df_plot['x_plot'] = df_plot['x_num'] + df_plot['Condition'].map({'rotation': -offset, 'No rotation': offset})

    # Plot
    plt.figure(figsize=(6, 5))
    sns.set_style("white")
    ax = plt.gca()

    bar_width = 0.10
    palette = {'rotation': '#8075C0', 'No rotation': '#36454F'}

    for cond in ['rotation', 'No rotation']:
        subset = df_plot[df_plot['Condition'] == cond]
        ax.bar(subset['x_plot'], subset['Value'],
               width=bar_width, color=palette[cond],
               edgecolor=None, label=cond)

    # === Labels and axes ===
    font_size = 14
    ax.set_xlabel('Peanut integrity', fontsize=font_size)
    ax.set_ylabel('Conditional probability of teeth-hole contact', fontsize=font_size)

    # Real numeric x-axis
    ax.set_xticks(df_plot['x_num'].unique())
    ax.set_xticklabels([str(x) for x in df_plot['x_num'].unique()], fontsize=font_size)
    ax.set_xlim(0.15, 2.2)  # ✅ brings first bar closer to y-axis

    # Add value labels
    for _, row in df_plot.iterrows():
        ax.text(row['x_plot'], row['Value'] + 0.02,
                f"{row['Value']:.2f}", ha='center', va='bottom', fontsize=font_size - 4)

    # === Legend and style ===
    ax.legend(title='', fontsize=font_size - 2, frameon=False, loc='upper left')
    sns.despine(top=True, right=True)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # --- STATS (Console Only) ---
    p_vals = []
    peanut_order = df_plot.sort_values('x_num')['Peanut Type'].unique()
    valid_peanuts = []
    for peanut in peanut_order:
        pd_info = dict_contact_rotation[peanut]
        if pd_info['tot_rot'] > 0 and pd_info['tot_no'] > 0:
            count = [pd_info['n_rot'], pd_info['n_no']]
            nobs = [pd_info['tot_rot'], pd_info['tot_no']]
            from statsmodels.stats.proportion import proportions_ztest
            _, p = proportions_ztest(count, nobs)
            p_vals.append(p)
            valid_peanuts.append(peanut)
        else:
            p_vals.append(1.0)
            valid_peanuts.append(peanut)

    if p_vals:
        from statsmodels.stats.multitest import multipletests
        # Using Benjamini-Hochberg (fdr_bh) which is less stringent than Holm
        rejected, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        
        print(f"\n--- Stats: Rotation vs No Rotation ---")
        print(f"  (Method: False Discovery Rate / Benjamini-Hochberg)")
        print(f"  {'Peanut':<12} {'p-raw':<10} {'p-adj':<10} {'Sig'}")
        print(f"  {'-'*42}")
        for peanut, p_raw, padj, rej in zip(valid_peanuts, p_vals, p_adj, rejected):
            sig_str = "*" if rej else "n.s."
            print(f"  {peanut:<12} {p_raw:<10.4f} {padj:<10.4g} {sig_str}")

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    plt.show()
    plt.close()

def get_assess_timing_without_chewing(processed):
    timings_assessment = get_active_intervals(processed, 'assessing')
    chewing_duration = []
    for timings in timings_assessment:
        if 'chewing' in np.unique(processed['Behavior']):
            timings_chewing = get_active_intervals(processed, 'chewing')
            for chewing_timings in timings_chewing:
                if chewing_timings[0] > timings[0] and chewing_timings[1] < timings[1]:
                    chewing_dur = chewing_timings[1] - chewing_timings[0]
                    if chewing_dur >= 0:
                        chewing_duration.append(chewing_dur)
    return timings_assessment, int(sum(chewing_duration))

def get_assessment_timings_dict(dict_ethograms, correct_chewing, overall_or_per_type):
    assessing_time_dict = {}
    peanut_types = ['control', 'three', 'five', 'seven', 'one', 'onehalf', 'two']
    for peanut in peanut_types:
        total_duration = []
        for ID, processed in dict_ethograms.items():
            processed = processed['processed']
            for media in np.unique(processed['Media file name']):
                if peanut in media:
                    if 'scatter-hoarding' in np.unique(processed['Behavior'][processed['Media file name'] == media]):
                        outcome = 'scatter-hoarding'
                    elif 'eating' in np.unique(processed['Behavior'][processed['Media file name'] == media]):
                        outcome = 'eating'
                    else:
                        outcome = np.nan
                    if correct_chewing == False:
                        timings_assessment = get_active_intervals(processed[processed['Media file name'] == media], 'assessing')
                        if timings_assessment:
                            for assess_bout in timings_assessment:
                                bout_duration = assess_bout[1] - assess_bout[0]
                                if bout_duration < 0:
                                    bout_duration = 0
                                total_duration.append([bout_duration, outcome])                        
                        else:
                            total_duration.append([0, outcome])
                    else:
                        timings_assessment, chewing_time = get_assess_timing_without_chewing(processed[processed['Media file name'] == media])
                        if timings_assessment:
                            for assess_bout in timings_assessment:
                                bout_duration = assess_bout[1] - assess_bout[0] - chewing_time
                                if bout_duration < 0:
                                    bout_duration = 0
                                total_duration.append([bout_duration, outcome])                        
                        else:
                            total_duration.append([0, outcome])
        assessing_time_dict[peanut] = total_duration

    if overall_or_per_type == 'overall':
        return [item for sublist in assessing_time_dict.values() for item in sublist]
    else:
        return assessing_time_dict


def plot_overall_assessment_time(assessing_timings_overall, time_or_number):
    df = pd.DataFrame(assessing_timings_overall, columns=["Value", "Outcome"])
    df = df.dropna(subset=["Outcome"])
    df = df[df['Value'] != 0]

    # --- split groups ---
    eat_vals = df.loc[df["Outcome"] == "eating", "Value"].dropna()
    scatter_vals = df.loc[df["Outcome"] == "scatter-hoarding", "Value"].dropna()

    # --- Mann–Whitney U test (nonparametric) ---
    from scipy.stats import mannwhitneyu
    if len(eat_vals) > 0 and len(scatter_vals) > 0:
        stat, p_val = mannwhitneyu(eat_vals, scatter_vals, alternative="two-sided")
    else:
        stat, p_val = np.nan, np.nan

    # --- plot ---
    plt.figure(figsize=(5, 5))
    sns.set_style("white")

    sns.violinplot(
        data=df,
        x="Outcome",
        y="Value",
        order=["eating", "scatter-hoarding"],
        palette={"eating": "#1A897B", "scatter-hoarding": "#FFB722"},
        cut=0,
        inner=None,
        linewidth=1.2,
        alpha=0.9
    )

    sns.stripplot(
        data=df,
        x="Outcome",
        y="Value",
        order=["eating", "scatter-hoarding"],
        color="black",
        size=5,
        jitter=True,
        alpha=0.7
    )

    # --- significance line and stars ---
    ax = plt.gca()
    y_max = df["Value"].max()
    y_line = y_max * 1.05
    y_text = y_max * 1.1

    ax.plot([0, 0, 1, 1], [y_line, y_line + 0.05 * y_max, y_line + 0.05 * y_max, y_line],
            color="black", lw=1.5)

    # significance label
    if p_val < 0.001:
        star = '***'
    elif p_val < 0.01:
        star = '**'
    elif p_val < 0.05:
        star = '*'
    else:
        star = 'ns'

    ax.text(0.5, y_text, star, ha='center', va='bottom', fontsize=16, fontweight='bold')

    # --- style ---
    if time_or_number == 'time':
        ax.set_ylabel("Assessment time (s)", fontsize=14)
    else:
        ax.set_ylabel("Number of behaviors", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Eat", "Scatter-hoard"], fontsize=14)

    sns.despine(top=True, right=True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', width=1.5, length=5, color='black', direction='out')
    ax.tick_params(axis='y', width=1.5, length=5, color='black', direction='out')
    plt.gca().tick_params(which='both', bottom=True, left=True)
    
    plt.tight_layout()
    plt.show()

    print(f"Mann–Whitney U test: U = {stat:.3f}, p = {p_val:.4f} → {star}")


def plot_assessment_time_per_types(dict_data, time_or_number):
    """
    Plot side-by-side violin plots with scatter points for each peanut type
    comparing 'eating' vs 'scatter-hoarding' assessment durations.
    """

    # --- 1. Flatten dictionary into a long DataFrame ---
    rows = []
    for peanut_type, records in dict_data.items():
        for val, outcome in records:
            if pd.notna(outcome) and outcome in ['eating', 'scatter-hoarding']:
                rows.append({
                    'Peanut Type': peanut_type,
                    'Duration': val,
                    'Outcome': outcome
                })
    df = pd.DataFrame(rows)

    if df.empty:
        print("No valid data to plot.")
        return

    # --- 2. Ensure consistent category order ---
    order_types = ['control', 'three', 'five', 'seven', 'one', 'onehalf', 'two']
    df['Peanut Type'] = pd.Categorical(df['Peanut Type'], categories=order_types, ordered=True)

    # --- 3. Define outcome colors ---
    palette = {'eating': '#1A897B', 'scatter-hoarding': '#FFB722'}

    # --- 4. Plot ---
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=df,
        x='Peanut Type',
        y='Duration',
        hue='Outcome',
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.5
    )

    # Add scatter points (with small horizontal jitter)
    sns.stripplot(
        data=df,
        x='Peanut Type',
        y='Duration',
        hue='Outcome',
        dodge=True,
        jitter=0.18,
        size=4,
        color='black',
        edgecolor='black',
        linewidth=0.3,
        ax=ax,
        alpha = 0.5
    )

    # --- 5. Cleanup duplicates in legend ---
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Eating', 'Scatter-hoarding'],
              title='', frameon=False, fontsize=12, loc='upper right')

    # --- 6. Axis formatting ---
    sns.despine(top=True, right=True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    ax.set_xlabel('', fontsize=16, weight='bold')
    if time_or_number == 'time':
        ax.set_ylabel('Assessment duration (s)', fontsize=16)
        plt.title('Assessment duration by peanut type and outcome', fontsize=18, weight='bold')
    else:
        ax.set_ylabel('Number of behaviors', fontsize=16)
        plt.title('Number of behaviors by peanut type and outcome', fontsize=18, weight='bold')
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14, width=2, length=6)

    plt.tick_params(axis='both', width=1.5, length=5, color='black', direction='out')
    ax.tick_params(which='both', bottom=True, left=True)

    plt.tight_layout()
    plt.show()




















    
    
from scipy.stats import chisquare as scipy_chisquare






def plot_weights_empty_vs_full(weights_df, save_path=None, show=True):
    
    weights_df_ = weights_df.melt(
        value_vars=["Control", "Empty"],
        var_name="Condition",
        value_name="Value"
    )

    #for cond in ['Control', 'Empty']:
        #print(np.mean(weights_df_['Value'][weights_df_['Condition'] == cond]))
    #print(weights_df_)
    
    plt.figure(figsize=(5, 6.5))
    
    palette = {
        "Control": "#4C72B0",   # blue
        "Empty":   "#DD8452"    # orange
    }
    
    ax = sns.violinplot(
        data=weights_df_,
        x="Condition",
        y="Value",
        cut=0,
        inner=None,
        linewidth=1,
        palette=palette
    )
    
    sns.stripplot(
        data=weights_df_,
        x="Condition",
        y="Value",
        color="black",
        alpha=0.6,
        jitter=True,
        size=6
    )
    
    # ---- axis limits ----
    ymax = weights_df_["Value"].max()
    plt.ylim(0, ymax + 0.2)
    
    # ---- remove frame, thicken axes ----
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    
    ax.minorticks_off()
    
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=8,
        width=2,
        labelsize=20
    )
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    
    # --- Statistics ---
    from scipy.stats import mannwhitneyu
    g1 = weights_df["Control"].dropna()
    g2 = weights_df["Empty"].dropna()
    if len(g1) > 0 and len(g2) > 0:
        stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
        print(f"Mann-Whitney U test (Control vs Empty): p={pval:.4g}")
    
        # --- Drawing bracket ---
        if pval < 0.05:
            x1, x2 = 0, 1
            y, h, col = ymax + 0.05, 0.02, 'k'
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            ax.text((x1+x2)*.5, y+h, stars, ha='center', va='bottom', color=col, fontsize=20)

    # ---- labels & title ----
    ax.set_ylabel("Peanut weight (g)", fontsize=22)
    ax.set_xlabel("", fontsize=20)
    #ax.set_title("Control vs Empty", fontsize=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Plot saved to: {save_path}")
        

    plt.show()




def compute_prob_outcome_given_thc(dict_ethograms, output_file=None):
    # --- Counters ---
    # THC Yes
    thc_y = 0
    thc_y_sh = 0
    thc_y_eat = 0
    
    # THC No
    thc_n = 0
    thc_n_sh = 0
    thc_n_eat = 0
    
    for id_ in dict_ethograms.keys():
        processed = dict_ethograms[id_]['processed']
        
        for file in np.unique(processed['Media file name']):
            file_df = processed[processed['Media file name'] == file]
            counter = Counter(file_df['Behavior'])
    
            sh_  = 'scatter-hoarding' in counter
            eat_ = 'eating' in counter
            thc_ = 'teeth-hole contact' in counter
    
            if thc_:
                thc_y += 1
                if sh_:  thc_y_sh += 1
                if eat_: thc_y_eat += 1
            else:
                thc_n += 1
                if sh_:  thc_n_sh += 1
                if eat_: thc_n_eat += 1
    
    # --- Compute probabilities ---
    p_sh_given_thc_y  = thc_y_sh  / thc_y if thc_y > 0 else 0
    p_eat_given_thc_y = thc_y_eat / thc_y if thc_y > 0 else 0
    
    p_sh_given_thc_n  = thc_n_sh  / thc_n if thc_n > 0 else 0
    p_eat_given_thc_n = thc_n_eat / thc_n if thc_n > 0 else 0

    # --- STATS (Console) ---
    print(f"\n--- Stats: Outcome SH vs Eat (Shifted by THC) ---")
    print(f"Outcome Ratio (Eat:SH):")
    print(f"  THC (+):  {thc_y_eat}:{thc_y_sh} ({p_eat_given_thc_y:.1%} vs {p_sh_given_thc_y:.1%})")
    print(f"  THC (-):  {thc_n_eat}:{thc_n_sh} ({p_eat_given_thc_n:.1%} vs {p_sh_given_thc_n:.1%})")
        
    from scipy.stats import fisher_exact
    # Fisher Test: (Eat vs SH) x (THC+ vs THC-)
    # This tests if the "Choice" (Eat/SH) is dependent on the "Detection" (THC)
    table_choice = [[thc_y_eat, thc_y_sh], 
                    [thc_n_eat, thc_n_sh]]
    
    if (thc_y_eat + thc_y_sh) > 0 and (thc_n_eat + thc_n_sh) > 0:
        odds_c, p_c = fisher_exact(table_choice)
        print(f"Fisher Exact (Behavior Choice dependent on THC): p={p_c:.4g}, odds={odds_c:.2f}")
        if p_c < 0.05:
            print("Result: THC significantly shifts the balance between Eating and Scatter-hoarding.")
        else:
            print("Result: No significant shift in Eating/SH ratio due to THC.")

    # Put into dict for plotting (showing the THC+ probabilities as previously)
    pvals = {
        "Eating": p_eat_given_thc_y,
        "Scatter-hoarding": p_sh_given_thc_y
    }
    
    labels = list(pvals.keys())
    values = list(pvals.values())
    
    # === PLOT ===================================================================
    plt.figure(figsize=(6, 6))
    
    ax = sns.barplot(
        x=labels,
        y=values,
        palette={
            "Scatter-hoarding": "#FFB722",
            "Eating": "#1A897B"
        },
        width=0.6,
        edgecolor="white",
        linewidth=2
    )
    
    # --- Axis labels ---
    ax.set_ylabel("P(Behavior | Teeth-hole contact)", fontsize=20, labelpad=10)
    ax.set_xlabel("", fontsize=20)
    
    # --- Style ---
    sns.set_style("white")
    ax.grid(False)
    sns.despine(ax=ax, top=True, right=True)
    
    # Add n tags on top
    # Show n_outcome / total_THC
    ax.text(0, values[0] + 0.02, f"n={thc_y_eat}/{thc_y}", ha='center', va='bottom', fontsize=15, fontweight='bold')
    ax.text(1, values[1] + 0.02, f"n={thc_y_sh}/{thc_y}", ha='center', va='bottom', fontsize=15, fontweight='bold')

    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color("black")
    
    ax.tick_params(axis='x', labelsize=18, width=2, length=6, color='black', direction='out')
    ax.tick_params(axis='y', labelsize=18, width=2, length=6, color='black', direction='out')
    plt.gca().tick_params(which='both', bottom=True, left=True)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    plt.show()
    plt.close()



def show_peanuts_mass(masses_df, save_path=None):
    sns.reset_defaults()
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    
    # --- Data Prep ---
    masses_df["recovering_time_num"] = pd.to_numeric(masses_df["recovering_time"], errors="coerce")
    masses_df = masses_df.dropna(subset=["recovering_time_num", "prop"])
    
    palette = {"intact": "#D4B298", "hole": "#B07C65", "crushed": "#88756B"}
    treatment_order = ["intact", "hole", "crushed"]
    shifts = {"intact": -0.1, "hole": 0.0, "crushed": 0.1}
    
    # --- Plot Scatter with Shifts ---
    for treatment in treatment_order:
        subset = masses_df[masses_df["treatment"] == treatment]
        if subset.empty: continue
        ax.scatter(
            subset["recovering_time_num"] + shifts[treatment],
            subset["prop"],
            color=palette[treatment],
            s=40,
            label=treatment,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )
        
    # --- Add Regression Lines ---
    results = []
    print("\nLinear regression of weight loss vs. recovery time:\n")
    slopes = {}
    for treatment in treatment_order:
        subset = masses_df[masses_df["treatment"] == treatment]
        if len(subset) < 2: continue
        
        sns.regplot(
            data=subset,
            x="recovering_time_num",
            y="prop",
            scatter=False,
            ax=ax,
            color=palette[treatment],
            truncate=True,
            line_kws={"linewidth": 3, "alpha": 0.8}
        )
        
        # Stats
        slope, intercept, r_value, p_value, std_err = linregress(subset["recovering_time_num"], subset["prop"])
        slopes[treatment] = slope
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{treatment:>8} → slope = {slope:.4f}, p = {p_value:.4g} ({sig}), R² = {r_value**2:.3f}")
        results.append({"treatment": treatment, "slope": slope, "p_value": p_value, "r2": r_value**2})

    # --- Aesthetics ---
    ax.set_xlabel("Recovery time (days)", fontsize=18, labelpad=10)
    ax.set_xticks(range(int(masses_df["recovering_time_num"].min()), int(masses_df["recovering_time_num"].max()) + 1))
    ax.set_ylabel("Proportion weight loss", fontsize=18, labelpad=10)
    ax.set_title("Weight loss across recovery time and treatment", fontsize=20, pad=15)
    
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(False)
    
    # Axis styling (equivalent to style_axes)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(3)
        ax.spines[spine].set_color("black")
    
    ax.tick_params(axis="both", labelsize=15, width=3, length=6, color="black", direction="out", bottom=True, left=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight', transparent=True)
    
    plt.show()
    return slopes


def show_peanut_mold(df_merged, save_path=None):
    sns.reset_defaults()
    df = df_merged.copy()
    
    # --- Data Prep ---
    df["delay_num"] = pd.to_numeric(df["delay"], errors="coerce")
    df = df.dropna(subset=["delay_num", "proportion of red"])
    
    palette = {"intact": "#D4B298", "hole": "#B07C65", "crushed": "#88756B"}
    condition_order = ["intact", "hole", "crushed"]
    shifts = {"intact": -0.1, "hole": 0.0, "crushed": 0.1}
    
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    
    # --- Plot Scatter with Shifts ---
    for cond in condition_order:
        subset = df[df["condition"] == cond]
        if subset.empty: continue
        ax.scatter(
            subset["delay_num"] + shifts[cond],
            subset["proportion of red"],
            color=palette[cond],
            s=40,
            label=cond,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )

    # --- Add Regression Lines & Stats ---
    print("\n=== Linear regressions (proportion of red ~ delay) ===\n")
    slopes = {}
    for cond in condition_order:
        subset = df[df["condition"] == cond]
        if len(subset) < 2: continue
        
        sns.regplot(
            data=subset,
            x="delay_num",
            y="proportion of red",
            scatter=False,
            ax=ax,
            color=palette[cond],
            truncate=True,
            line_kws={"linewidth": 3, "alpha": 0.8}
        )
        
        # Stats
        slope, intercept, r_value, p_value, std_err = linregress(subset["delay_num"], subset["proportion of red"])
        slopes[cond] = slope
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"Condition: {cond}")
        print(f"  Slope:        {slope:.4f}")
        print(f"  Intercept:    {intercept:.4f}")
        print(f"  R²:           {r_value**2:.4f}")
        print(f"  p-value:      {p_value:.4g} ({sig})")
        print("-" * 50)

    # --- Aesthetics ---
    ax.set_xlabel("Delay (days)", fontsize=20, labelpad=10)
    ax.set_xticks(range(int(df["delay_num"].min()), int(df["delay_num"].max()) + 1))
    ax.set_ylabel("Proportion red pixels", fontsize=20, labelpad=10)
    ax.set_title("Red pixel proportion across delay and condition", fontsize=20, pad=20)
    
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(False)
    
    # Axis styling
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(3)
        ax.spines[spine].set_color("black")
    
    ax.tick_params(axis='both', labelsize=18, width=3, length=6, color='black', direction='out', bottom=True, left=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        if save_path.endswith('.png'):
            plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight', transparent=True)
    
    plt.show()
    return slopes


def plot_slopes(slopes_dict, ylabel="Slope", save_path=None, save_path_svg=None):
    """
    Plots a bar chart of slopes for each condition.
    """
    sns.reset_defaults()
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # Data prep
    conditions = ['intact', 'hole', 'crushed']
    
    plot_data = []
    for cond in conditions:
        val = slopes_dict.get(cond) or slopes_dict.get(cond.capitalize())
        if val is not None:
            plot_data.append({'Condition': cond, 'Slope': val})
    
    df = pd.DataFrame(plot_data)
    palette = {"intact": "#D4B298", "hole": "#B07C65", "crushed": "#88756B"}

    # Plot
    sns.barplot(data=df, x='Condition', y='Slope', palette=palette, ax=ax, order=conditions)
    
    # Aesthetics
    ax.set_xlabel("Condition", fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)
    
    sns.despine(ax=ax, top=True, right=True)
    
    # Axis styling
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(3)
        ax.spines[spine].set_color("black")
    
    ax.tick_params(axis="both", labelsize=15, width=3, length=6, color="black", direction="out", bottom=True, left=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        if save_path.endswith('.png') and not save_path_svg:
            plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight', transparent=True)
    
    if save_path_svg:
        plt.savefig(save_path_svg, bbox_inches='tight', transparent=True)
    
    plt.show()




# ===========================================================================
# Ethogram outcome plotting and resistance analysis (from from_antigravity_peanuts.py)
# ===========================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.formula.api import ols
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from matplotlib.transforms import blended_transform_factory
import statsmodels.formula.api as smf
from scipy.stats import linregress, norm, ttest_1samp
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import quad
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
import json
import pickle
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

def plot_eating_psth(ethogram_path, eating_scoring_path, window=20.0, save_path=None):
    """
    Plots a PSTH of 'Clamp' and 'Teeth Hole Contact' events relative to the nearest
    'Eat' event onset, to avoid misattributing events when multiple peanuts are processed.
    """
    import pickle
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    with open(ethogram_path, 'rb') as f:
        data1 = pickle.load(f)
    with open(eating_scoring_path, 'rb') as f:
        data2 = pickle.load(f)

    # Get eat times
    eat_times = data2[data2['Behavior'].str.lower().str.strip() == 'eat']['Start (s)'].values

    # Get clamp and teeth times from data1
    clamp_times = []
    teeth_times = []
    for obs, obs_data in data1['observations'].items():
        if 'events' in obs_data:
            for ev in obs_data['events']:
                if len(ev) > 2:
                    t = float(ev[0])
                    beh = str(ev[2]).lower().strip()
                    if beh == 'clamp':
                        clamp_times.append(t)
                    elif 'teeth' in beh:
                        teeth_times.append(t)

    clamp_times = np.array(clamp_times)
    teeth_times = np.array(teeth_times)

    bins = np.linspace(-window, window, int(window * 2) + 1) # 1 second bins

    def get_nearest_diffs(event_times, target_times, window_size):
        diffs = []
        if len(target_times) == 0: return diffs
        for t in event_times:
            nearest_target = target_times[np.argmin(np.abs(target_times - t))]
            diff = t - nearest_target
            if -window_size <= diff <= window_size:
                diffs.append(diff)
        return diffs

    rel_clamps = get_nearest_diffs(clamp_times, eat_times, window)
    rel_teeth = get_nearest_diffs(teeth_times, eat_times, window)

    print("=== PSTH Event Counts ===")
    print(f"Eat events found in eating scoring: {len(eat_times)}")
    print(f"Clamp events found in ethogram: {len(clamp_times)}")
    print(f"Teeth Hole Contact events found in ethogram: {len(teeth_times)}")
    print(f"Clamps mapped within ±{window}s window: {len(rel_clamps)}")
    print(f"Teeth contacts mapped within ±{window}s window: {len(rel_teeth)}")
    print("=========================")


    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].hist(rel_clamps, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_ylabel('Clamp Count')
    axes[0].set_title('PSTH of Clamps Relative to Eat Onset')

    axes[1].hist(rel_teeth, bins=bins, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Teeth Contact Count')
    axes[1].set_xlabel('Time relative to Eat (s)')
    axes[1].set_title('PSTH of Teeth Hole Contacts Relative to Eat Onset')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(axis='both', width=2, length=6, labelsize=12)

    plt.tight_layout()
    if save_path:
        import os
        if not save_path.endswith('.svg'):
            save_path = save_path + '.svg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    return fig, axes

def plot_last_three_movements_probability(ethogram_path, sites_directory, output_file=None, teeth_mode='bottom'):
    """
    Plots the probability of the last three clamp movements right before Teeth Hole Contact.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    def peanut_radius(z, a=1.0, b=1.1):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    all_sequences = load_aligned_clamp_sequences(ethogram_path, sites_directory, teeth_mode)
        
    patterns = {'LLL': 0, 'LLT': 0, 'LTL': 0, 'LTT': 0, 'TLL': 0, 'TLT': 0, 'TTL': 0, 'TTT': 0}
    total_valid = 0
    
    for seq in all_sequences:
        before_thc = []
        for pt in seq:
            before_thc.append(pt)
            if pt['t_rel'] >= 0:
                break
                
        # We need at least 4 clamps to have 3 movements
        if len(before_thc) < 4:
            continue
            
        moves = []
        for i in range(len(before_thc) - 1):
            p1 = before_thc[i]['p']
            p2 = before_thc[i+1]['p']
            dz = p2[2] - p1[2]
            avg_z = (p1[2] + p2[2]) / 2
            r = peanut_radius(avg_z)
            ang1 = np.arctan2(p1[1], p1[0])
            ang2 = np.arctan2(p2[1], p2[0])
            darc = r * ((ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi)
            
            if abs(dz) >= abs(darc): 
                moves.append('L')
            else: 
                moves.append('T')
                
        last_three = "".join(moves[-3:])
        if last_three in patterns:
            patterns[last_three] += 1
            total_valid += 1
            
    if total_valid == 0:
        print("No peanuts found with at least 3 movements before contact.")
        return
        
    labels = list(patterns.keys())
    counts = list(patterns.values())
    probs = [(c / total_valid) * 100 for c in counts]
    
    plt.figure(figsize=(10, 6))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.size'] = 14
    
    colors = ['red' if l == 'LTT' else '#1f77b4' for l in labels]
    bars = plt.bar(labels, probs, color=colors, edgecolor='black', alpha=0.8)
    
    plt.title(f'Probability of Last Three Movements Before Contact (n={total_valid})')
    plt.xlabel('Movement Pattern (L=Longitudinal, T=Transverse)')
    plt.ylabel('Probability (%)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12)
        
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()

def plot_longitudinal_barycenter_ratios(sites_directory, teeth_mode='bottom', output_file=None, plot_mode='ratio'):
    """
    Computes and plots the ratio of 'Away from Middle' over 'Towards Middle' 
    for the three longitudinal thirds, relative to the barycenter.
    plot_mode can be 'ratio' or 'stacked' (stacked percentage bar plot).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    all_files = [os.path.join(sites_directory, el) for el in os.listdir(sites_directory)]
    valid_files = [f for f in all_files if 'front' not in os.path.basename(f).lower()]

    master_dict = {
        'sites': [],
        'arrows': []
    }

    for sites_path in valid_files:
        try:
            data = generate_folded_half_chamber_map_with_sequence(
                sites_path=sites_path, 
                output_file=None, 
                teeth_mode=teeth_mode, 
                show_quadrants=True, 
                show_axes=False, 
                plot=False, 
                show_full_context=False
            )
            if data:
                master_dict['sites'].extend(data['sites'])
                master_dict['arrows'].extend(data['arrows'])
        except Exception:
            pass

    a, b = 1.2602, 1.3749
    z_max = np.sqrt(a**2 + b**2)
    z_edges = [0, -z_max/3, -2*z_max/3, -z_max]
    z_center = -z_max / 2

    longitudinal_arrows = []
    for arrow in master_dict['arrows']:
        start_y = arrow['start_y']
        dy = arrow['dy']
        dx = arrow['dx']
        
        if abs(dy) < abs(dx):
            continue
        
        if start_y >= z_edges[1]:
            region = 'Equatorial'
        elif start_y >= z_edges[2]:
            region = 'Intermediate'
        else:
            region = 'Tip'
            
        # Waist is at z=0, so dy > 0 means towards waist.
        direction = 'Towards Waist' if dy > 0 else 'Away from Waist'
            
        arrow['region'] = region
        arrow['direction'] = direction
        longitudinal_arrows.append(arrow)

    regions = ['Equatorial', 'Intermediate', 'Tip']
    ratios = []
    towards_pcts = []
    away_pcts = []
    counts = {}
    
    for r in regions:
        r_arrows = [a for a in longitudinal_arrows if a['region'] == r]
        towards = sum(1 for a in r_arrows if a['direction'] == 'Towards Waist')
        away = sum(1 for a in r_arrows if a['direction'] == 'Away from Waist')
        
        total = towards + away
        ratio = away / towards if towards > 0 else np.nan
        ratios.append(ratio)
        
        t_pct = (towards / total * 100) if total > 0 else 0.0
        a_pct = (away / total * 100) if total > 0 else 0.0
        towards_pcts.append(t_pct)
        away_pcts.append(a_pct)
        counts[r] = {'away': away, 'towards': towards, 'ratio': ratio, 'total': total}

    fig, ax = plt.subplots(figsize=(8, 6))

    if plot_mode == 'ratio':
        bars = ax.bar(regions, ratios, color=['#2ca02c', '#ff7f0e', '#9467bd'], edgecolor='black', linewidth=2, alpha=0.85)
        
        # Add ratio line at 1.0
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Equal Ratio (1:1)')
        
        # Add count text on top of bars
        for bar, r in zip(bars, regions):
            yval = bar.get_height()
            if np.isnan(yval):
                continue
            text_str = f"{counts[r]['away']} / {counts[r]['towards']}"
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, text_str, ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Ratio (Away / Towards)')
        ax.set_title('Ratio of Longitudinal Vectors (Away vs Towards Barycenter)')
        ax.legend(loc='upper left')
        
        # Expand ylim a bit to fit text
        max_ratio = np.nanmax(ratios)
        ax.set_ylim(0, max_ratio * 1.25)
    else:
        # Stacked bar plot for percentages
        bar_width = 0.6
        p1 = ax.bar(regions, towards_pcts, bar_width, color='#1f77b4', edgecolor='black', label='Towards Waist')
        p2 = ax.bar(regions, away_pcts, bar_width, bottom=towards_pcts, color='#d62728', edgecolor='black', label='Away from Waist')
        
        # Add text in the middle of each stacked bar section
        for i, r in enumerate(regions):
            t_pct, a_pct = towards_pcts[i], away_pcts[i]
            t_count, a_count = counts[r]['towards'], counts[r]['away']
            
            if t_pct > 5:
                ax.text(i, t_pct/2, f"{t_pct:.1f}%\n({t_count})", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            if a_pct > 5:
                ax.text(i, t_pct + a_pct/2, f"{a_pct:.1f}%\n({a_count})", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        
        ax.axhline(50.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Percentage of Vector Directions by Region')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 100)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.tick_params(axis='both', width=2.5, length=6, labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.title.set_size(16)
    
    plt.tight_layout()
    if output_file:
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        
    plt.show()
    return counts

def plot_longitudinal_barycenter_vectors(sites_directory, teeth_mode='bottom', show_full_context=True, output_file=None, show_marginal=False):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    all_files = [os.path.join(sites_directory, el) for el in os.listdir(sites_directory)]
    valid_files = [f for f in all_files if 'front' not in os.path.basename(f).lower()]

    master_dict = {
        'sites': [],
        'arrows': []
    }

    for sites_path in valid_files:
        try:
            data = generate_folded_half_chamber_map_with_sequence(
                sites_path=sites_path, 
                output_file=None, 
                teeth_mode=teeth_mode, 
                show_quadrants=True, 
                show_axes=False, 
                plot=False, 
                show_full_context=show_full_context
            )
            if data:
                master_dict['sites'].extend(data['sites'])
                master_dict['arrows'].extend(data['arrows'])
        except Exception as e:
            pass

    a, b = 1.2602, 1.3749
    z_max = np.sqrt(a**2 + b**2)
    z_edges = [0, -z_max/3, -2*z_max/3, -z_max]

    z_center = -z_max / 2

    longitudinal_arrows = []
    for arrow in master_dict['arrows']:
        start_y = arrow['start_y']
        dy = arrow['dy']
        dx = arrow['dx']
        
        if abs(dy) < abs(dx):
            continue
        
        if start_y >= z_edges[1]:
            region = 'Equatorial'
        elif start_y >= z_edges[2]:
            region = 'Intermediate'
        else:
            region = 'Tip'
            
        # Waist is at z = 0. Since domain is 0 to -z_max, pointing towards waist means dy > 0
        if dy > 0:
            direction = 'Towards Waist'
            color = '#1f77b4'
        else:
            direction = 'Away from Waist'
            color = '#d62728'
            
        arrow['region'] = region
        arrow['direction'] = direction
        arrow['custom_color'] = color
        longitudinal_arrows.append(arrow)

    if show_marginal:
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1.2], wspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

    def peanut_radius_local(z, a=a, b=b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    z_refs = np.linspace(0, -z_max, 200)
    r_refs = peanut_radius_local(z_refs)
    width_vals = r_refs * (np.pi / 2)

    outline_x_pos = width_vals
    outline_x_neg = -width_vals
    outline_y = z_refs

    ax.plot(outline_x_pos, outline_y, 'k-', linewidth=2)
    ax.plot(outline_x_neg, outline_y, 'k-', linewidth=2)
    ax.plot([outline_x_pos[0], outline_x_neg[0]], [outline_y[0], outline_y[0]], 'k-', linewidth=2)
    ax.plot([outline_x_pos[-1], outline_x_neg[-1]], [outline_y[-1], outline_y[-1]], 'k-', linewidth=2)

    max_width = max(width_vals)
    for z in z_edges[1:3]:
        ax.hlines(y=z, xmin=-max_width*1.1, xmax=max_width*1.1, color='green', linestyle='--', linewidth=2, alpha=0.6)

    ax.hlines(y=0, xmin=-max_width*0.8, xmax=max_width*0.8, color='purple', linestyle=':', linewidth=3, alpha=0.8)
    ax.text(max_width*0.85, 0, 'Waist (Middle of Peanut)', color='purple', fontsize=12, verticalalignment='center')
        
    ax.text(-max_width*1.1, z_edges[0]/2 + z_edges[1]/2, 'Equatorial Third', color='green', fontsize=12, verticalalignment='center')
    ax.text(-max_width*1.1, z_edges[1]/2 + z_edges[2]/2, 'Intermediate Third', color='green', fontsize=12, verticalalignment='center')
    ax.text(-max_width*1.1, z_edges[2]/2 + z_edges[3]/2, 'Tip Third', color='green', fontsize=12, verticalalignment='center')

    for arrow in longitudinal_arrows:
        ax.arrow(arrow['start_x'], arrow['start_y'], arrow['dx'], arrow['dy'],
                 head_width=0.08, head_length=0.1,
                 fc=arrow['custom_color'], ec=arrow['custom_color'],
                 length_includes_head=True, linewidth=2, zorder=15, alpha=0.85)

    custom_lines = [Line2D([0], [0], color='#1f77b4', lw=4),
                    Line2D([0], [0], color='#d62728', lw=4)]
    ax.legend(custom_lines, ['Towards Waist', 'Away from Waist'], loc='upper right')

    ax.set_aspect('equal')
    ax.set_title(f"Filtered Vectors Relative to Waist (n={len(longitudinal_arrows)})")
    ax.axis('off')

    if show_marginal:
        from scipy.stats import gaussian_kde
        towards_y = [a['start_y'] for a in longitudinal_arrows if a['direction'] == 'Towards Waist']
        away_y = [a['start_y'] for a in longitudinal_arrows if a['direction'] == 'Away from Waist']

        # Blue KDE
        if len(towards_y) > 1:
            try:
                kde_t = gaussian_kde(towards_y)
                y_grid_t = np.linspace(np.min(towards_y), np.max(towards_y), 200)
                ax_hist.plot(kde_t(y_grid_t), y_grid_t, color='#1f77b4', linewidth=2.5, zorder=4)
                ax_hist.fill_betweenx(y_grid_t, 0, kde_t(y_grid_t), color='#1f77b4', alpha=0.3)
            except Exception:
                pass

        # Red KDE
        if len(away_y) > 1:
            try:
                kde_a = gaussian_kde(away_y)
                y_grid_a = np.linspace(np.min(away_y), np.max(away_y), 200)
                ax_hist.plot(kde_a(y_grid_a), y_grid_a, color='#d62728', linewidth=2.5, zorder=4)
                ax_hist.fill_betweenx(y_grid_a, 0, kde_a(y_grid_a), color='#d62728', alpha=0.3)
            except Exception:
                pass

        for z in z_edges[1:3]:
            ax_hist.hlines(y=z, xmin=0, xmax=ax_hist.get_xlim()[1], color='green', linestyle='--', linewidth=2, alpha=0.6)
            
        ax_hist.hlines(y=0, xmin=0, xmax=ax_hist.get_xlim()[1], color='purple', linestyle=':', linewidth=3, alpha=0.8)
        
        ax_hist.set_ylim(ax.get_ylim())
        ax_hist.set_xlabel("Density")
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
        
    plt.show()
    
    return longitudinal_arrows

def plot_survival_rates(survival_df):
    dict_orders = {}
    for tp in ['control', 'crushed']:
        type_df = survival_df[survival_df['Type'] == tp]
        probs = type_df['Order retreived'].value_counts(normalize=True)
        dict_orders[tp] = probs.get(1.0, 0)

    survival_df["Time_ret"] = survival_df["Time retreived"].apply(parse_time)

    labels = ["control", "crushed"]
    values = [dict_orders[l] for l in labels]  # percent
    
    colors = {
        "control": "#D4B298",
        "crushed": "#88756B"
    }
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    bars = ax.bar(
        labels,
        values,
        color=[colors[l] for l in labels],
        edgecolor="black",
        linewidth=1.2
    )
    
    # Axis label
    ax.set_ylabel("Proportion of\npeanuts retreived first", fontsize=20)
    
    # Y limits and ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1])
    ax.set_yticklabels([0, 0.25, 0.50, 0.75, 1], fontsize=20)
    
    # X ticks
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=20)
    
    # Chance line
    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )
    
    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.03,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=20
        )
    
    # ---------------------------
    # AESTHETIC FIXES
    # ---------------------------
    ax.grid(False)
    
    # Spine styling
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.4)
        ax.spines[spine].set_color("black")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # FORCE TICKS TO APPEAR
    ax.tick_params(
        axis="both",
        which="major",
        left=True,
        bottom=True,
        width=1.4,
        length=6,
        labelsize=20
    )
    
    plt.tight_layout()

    save_path = r'figures/survival_rate.svg'
    plt.savefig(
        save_path, 
        dpi=300, 
        bbox_inches='tight', 
        transparent=True)
    
    plt.show()

def plot_transition_evolution(
    ethogram_path, 
    sites_directory, 
    output_file=None, 
    teeth_mode='bottom', 
    bin_size=1.0, 
    time_range=(-15, 15), 
    x_axis_mode='time', 
    min_sample_size=3, 
    probability_mode='conditional', 
    show_only_TT=False
):
    """
    Plots the transition probabilities between Longitudinal (L) and Transversal (T) 
    clamp movements relative to the first Teeth Hole Contact (t=0).
    """
    import pickle
    import os
    import glob
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    
    def peanut_radius(z, a=1.0, b=1.1):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    with open(ethogram_path, 'rb') as f:
        data = pickle.load(f)
    
    transitions_data = []
    
    for obs_name, obs_data in data['observations'].items():
        if 'file' not in obs_data or '1' not in obs_data['file']:
            continue
            
        media_files = obs_data['file']['1']
        media_info = obs_data.get('media_info', {})
        events = obs_data.get('events', [])
        
        cum_time = 0.0
        for media_path in media_files:
            length = media_info['length'].get(media_path, 0)
            fps = media_info['fps'].get(media_path, 29.97)
            video_name = os.path.basename(media_path).replace('.MP4', '').replace('.mp4', '')
            
            v_start = cum_time
            v_end = cum_time + length
            cum_time += length
            
            # Find first Teeth Hole Contact in this window
            contact_abs_time = None
            for ev in events:
                if len(ev) < 3: continue
                ev_time = float(ev[0])
                ev_type = str(ev[2]).lower().strip()
                if 'teeth' in ev_type and v_start <= ev_time <= v_end:
                    contact_abs_time = ev_time
                    break
                    
            if contact_abs_time is None:
                continue
                
            contact_local_time = contact_abs_time - v_start
            
            # Load sites.json
            if teeth_mode == 'top':
                site_file_pattern = f"*_{video_name}_front_sites.json"
            else:
                site_file_pattern = f"*_{video_name}_sites.json"
                
            potential_files = glob.glob(os.path.join(sites_directory, site_file_pattern))
            
            if not potential_files:
                if teeth_mode == 'top':
                    site_file_pattern = f"{video_name}_front_sites.json"
                else:
                    site_file_pattern = f"{video_name}_sites.json"
                potential_files = glob.glob(os.path.join(sites_directory, site_file_pattern))
                
            if not potential_files:
                continue
                
            if teeth_mode == 'top':
                valid_files = [f for f in potential_files if 'front' in os.path.basename(f).lower()]
            else:
                valid_files = [f for f in potential_files if 'front' not in os.path.basename(f).lower()]
                
            if not valid_files:
                continue
                
            site_file = valid_files[0]
                
            with open(site_file, 'r') as f:
                sites_dict = json.load(f)
                
            if isinstance(sites_dict, dict) and 'sites' in sites_dict:
                site_list = sites_dict['sites']
            elif isinstance(sites_dict, dict):
                site_list = []
                for k, v in sites_dict.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                        site_list.extend(v)
            else:
                site_list = sites_dict
                
            # Filter valid points and sort by frame
            valid_points = [s for s in site_list if len(s) >= 4]
            valid_points.sort(key=lambda x: int(x[3]))
            
            # Determine L or T for each step
            seq = []
            for i in range(len(valid_points) - 1):
                p1 = [float(valid_points[i][0]), float(valid_points[i][1]), float(valid_points[i][2])]
                p2 = [float(valid_points[i+1][0]), float(valid_points[i+1][1]), float(valid_points[i+1][2])]
                dz = p2[2] - p1[2]
                avg_z = (p1[2] + p2[2]) / 2
                r = peanut_radius(avg_z)
                ang1 = np.arctan2(p1[1], p1[0])
                ang2 = np.arctan2(p2[1], p2[0])
                darc = r * ((ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi)
                
                if abs(dz) >= abs(darc):
                    seq.append('L')
                else:
                    seq.append('T')
                    
            # Identify transitions
            video_transitions = []
            for i in range(len(seq) - 1):
                trans_type = seq[i] + seq[i+1]
                mid_frame = float(valid_points[i+1][3])
                clamp_local_time = mid_frame / fps
                t_rel = clamp_local_time - contact_local_time
                video_transitions.append({'type': trans_type, 't_rel': t_rel})
                
            if not video_transitions:
                continue
                
            if x_axis_mode == 'steps':
                closest_idx = min(range(len(video_transitions)), key=lambda idx: abs(video_transitions[idx]['t_rel']))
                for i, t in enumerate(video_transitions):
                    t['step'] = i - closest_idx
                    
            transitions_data.extend(video_transitions)
            
    if not transitions_data:
        print("No transitions found to plot.")
        return
        
    # Binning
    if x_axis_mode == 'steps':
        bins = np.arange(time_range[0], time_range[1] + 2) - 0.5
    else:
        bins = np.arange(time_range[0], time_range[1] + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    prob_LL = []
    prob_TT = []
    prob_LT = []
    prob_TL = []
    
    for i in range(len(bins) - 1):
        b_start = bins[i]
        b_end = bins[i+1]
        
        if x_axis_mode == 'steps':
            in_bin = [t for t in transitions_data if b_start <= t.get('step', 0) < b_end]
        else:
            in_bin = [t for t in transitions_data if b_start <= t['t_rel'] < b_end]
        
        count_LL = sum(1 for t in in_bin if t['type'] == 'LL')
        count_LT = sum(1 for t in in_bin if t['type'] == 'LT')
        count_TL = sum(1 for t in in_bin if t['type'] == 'TL')
        count_TT = sum(1 for t in in_bin if t['type'] == 'TT')
        
        total_L = count_LL + count_LT
        total_T = count_TT + count_TL
        total_all = total_L + total_T
        
        denom_L = total_all if probability_mode == 'marginal' else total_L
        denom_T = total_all if probability_mode == 'marginal' else total_T
        
        prob_LL.append(count_LL / denom_L if (total_L > 0 and total_L >= min_sample_size) else np.nan)
        prob_LT.append(count_LT / denom_L if (total_L > 0 and total_L >= min_sample_size) else np.nan)
        prob_TT.append(count_TT / denom_T if (total_T > 0 and total_T >= min_sample_size) else np.nan)
        prob_TL.append(count_TL / denom_T if (total_T > 0 and total_T >= min_sample_size) else np.nan)
        
    print("=== Transition Evolution Summary ===")
    print(f"Total videos parsed: {len(data['observations'])}")
    print(f"Total transition segments collected: {len(transitions_data)}")
    print("====================================")

    plt.figure(figsize=(12, 6))
    
    # Global settings for fonts and illustrator compatibility
    plt.rcParams['svg.fonttype'] = 'none' # Ensure SVG uses true text instead of paths
    plt.rcParams['font.size'] = 14        # Uniform font size
    
    if show_only_TT:
        plt.plot(bin_centers, prob_TT, '-o', label='P(T -> T)', color='#d62728', linewidth=2)
    else:
        plt.plot(bin_centers, prob_LL, '-o', label='P(L -> L)', color='#1f77b4', linewidth=2)
        plt.plot(bin_centers, prob_LT, '--o', label='P(L -> T)', color='#1f77b4', linewidth=2, alpha=0.5)
        plt.plot(bin_centers, prob_TT, '-o', label='P(T -> T)', color='#d62728', linewidth=2)
        plt.plot(bin_centers, prob_TL, '--o', label='P(T -> L)', color='#d62728', linewidth=2, alpha=0.5)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Teeth Hole Contact')
    
    plt.xlim(time_range[0], time_range[1])
    
    if x_axis_mode == 'steps':
        plt.xlabel('Sampling Steps relative to Teeth Hole Contact (0 = closest step)')
    else:
        plt.xlabel(f'Time relative to Teeth Hole Contact (s) [Bin size={bin_size}s]')
    plt.ylabel('Transition Probability')
    plt.title('Evolution of Transition Probabilities')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove grid and frames (top/right spines)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
        print(f"Transition evolution plot saved to {output_file}")
    plt.show()

def plot_turning_angles_bias(turning_angles, output_file=None):
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import binomtest
    except ImportError:
        # Fallback for older scipy versions
        from scipy.stats import binom_test
        def binomtest(k, n, p, alternative):
            class Res:
                pass
            res = Res()
            res.pvalue = binom_test(k, n, p, alternative=alternative)
            return res

    right_turns = sum(1 for angle in turning_angles if 0 < angle <= 180)
    left_turns = sum(1 for angle in turning_angles if -180 <= angle < 0)
    zero_turns = sum(1 for angle in turning_angles if angle == 0)

    print(f"Right Turns (0 to 180\u00b0): {right_turns}")
    print(f"Left Turns (0 to -180\u00b0): {left_turns}")
    print(f"Straight/No Turn (0\u00b0): {zero_turns}")

    total_turns = right_turns + left_turns
    if total_turns > 0:
        stat_result = binomtest(right_turns, n=total_turns, p=0.5, alternative='two-sided')
        p_value = stat_result.pvalue
        print(f"\nBinomial Test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Conclusion: There is a statistically significant bias towards one side.")
        else:
            print("Conclusion: No significant difference between the number of Left and Right turns.")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Left Turns (< 0\u00b0)', 'Right Turns (> 0\u00b0)'], [left_turns, right_turns], 
                  color=['#d62728', '#1f77b4'], edgecolor='black', linewidth=1.5, alpha=0.85)

    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Comparison of Left vs. Right Turns', fontsize=14)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(left_turns, right_turns)*0.02), 
                str(int(yval)), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='both', width=2, length=6, labelsize=12)

    plt.tight_layout()
    if output_file:
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def plot_vector_magnitude_evolution(ethogram_path, sites_directory, output_file=None, teeth_mode='bottom', time_range=(-15, 15), movement_type='both'):
    import pickle
    import os
    import glob
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    with open(ethogram_path, 'rb') as f:
        data = pickle.load(f)
        
    all_magnitudes_by_step = {}
    
    for obs_name, obs_data in data['observations'].items():
        if 'file' not in obs_data or '1' not in obs_data['file']: continue
        media_files = obs_data['file']['1']
        media_info = obs_data.get('media_info', {})
        events = obs_data.get('events', [])
        
        cum_time = 0.0
        for media_path in media_files:
            length = media_info['length'].get(media_path, 0)
            fps = media_info['fps'].get(media_path, 29.97)
            video_name = os.path.basename(media_path).replace('.MP4', '').replace('.mp4', '')
            
            v_start = cum_time
            v_end = cum_time + length
            cum_time += length
            
            contact_abs_time = None
            for ev in events:
                if len(ev) < 3: continue
                ev_time = float(ev[0])
                if ev[2] == 'Teeth Hole Contact' and v_start <= ev_time <= v_end:
                    contact_abs_time = ev_time
                    break
                    
            if contact_abs_time is None: continue
            contact_local_time = contact_abs_time - v_start
            
            if teeth_mode == 'top':
                site_file_pattern = f"*_{video_name}_front_sites.json"
            else:
                site_file_pattern = f"*_{video_name}_sites.json"
                
            potential_files = glob.glob(os.path.join(sites_directory, site_file_pattern))
            if not potential_files:
                site_file_pattern = f"{video_name}_front_sites.json" if teeth_mode == 'top' else f"{video_name}_sites.json"
                potential_files = glob.glob(os.path.join(sites_directory, site_file_pattern))
                
            if not potential_files: continue
            valid_files = [f for f in potential_files if 'front' in os.path.basename(f).lower()] if teeth_mode == 'top' else [f for f in potential_files if 'front' not in os.path.basename(f).lower()]
            if not valid_files: continue
            site_file = valid_files[0]
            
            with open(site_file, 'r') as f:
                sites_dict = json.load(f)
                
            if isinstance(sites_dict, dict) and 'sites' in sites_dict: site_list = sites_dict['sites']
            elif isinstance(sites_dict, dict):
                site_list = []
                for k, v in sites_dict.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list): site_list.extend(v)
            else: site_list = sites_dict
                
            valid_points = [s for s in site_list if len(s) >= 4]
            valid_points.sort(key=lambda x: int(x[3]))
            seq = []
            for i in range(len(valid_points) - 1):
                p1 = [float(valid_points[i][0]), float(valid_points[i][1]), float(valid_points[i][2])]
                p2 = [float(valid_points[i+1][0]), float(valid_points[i+1][1]), float(valid_points[i+1][2])]
                dz = p2[2] - p1[2]
                avg_z = (p1[2] + p2[2]) / 2
                term1 = np.sqrt(1.1**4 + 4 * 1.0**2 * avg_z**2)
                r2 = term1 - avg_z**2 - 1.0**2
                r = np.sqrt(np.maximum(0, r2))
                ang1 = np.arctan2(p1[1], p1[0]); ang2 = np.arctan2(p2[1], p2[0])
                darc = r * ((ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi)
                if abs(dz) >= abs(darc): seq.append('L')
                else: seq.append('T')

            step_times = []
            for i in range(len(valid_points) - 1):
                mid_frame = float(valid_points[i+1][3])
                clamp_local_time = mid_frame / fps
                t_rel = clamp_local_time - contact_local_time
                step_times.append(t_rel)
                
            if not step_times: continue
            
            closest_idx = min(range(len(step_times)), key=lambda idx: abs(step_times[idx]))
            
            for i in range(len(valid_points) - 1):
                if movement_type != 'both' and seq[i] != movement_type:
                    continue
                    
                p1 = np.array([float(valid_points[i][0]), float(valid_points[i][1]), float(valid_points[i][2])])
                p2 = np.array([float(valid_points[i+1][0]), float(valid_points[i+1][1]), float(valid_points[i+1][2])])
                magnitude = np.linalg.norm(p2 - p1)
                
                s_idx = i - closest_idx
                if time_range[0] <= s_idx <= time_range[1]:
                    if s_idx not in all_magnitudes_by_step:
                        all_magnitudes_by_step[s_idx] = []
                    all_magnitudes_by_step[s_idx].append(magnitude)
                    
    if not all_magnitudes_by_step:
        print("No magnitudes found to plot.")
        return
        
    steps = sorted(list(all_magnitudes_by_step.keys()))
    avg_mags = []
    std_mags = []
    
    for s in steps:
        vals = all_magnitudes_by_step[s]
        avg_mags.append(np.mean(vals))
        std_mags.append(np.std(vals) if len(vals) > 1 else 0)
        
    avg_mags = np.array(avg_mags)
    std_mags = np.array(std_mags)
    

    # --- SPEARMAN CORRELATION FOR RAMP-UP (STEPS <= 0) ---
    x_raw = []
    y_raw = []
    for s in steps:
        if s <= 0:
            vals = all_magnitudes_by_step[s]
            x_raw.extend([s] * len(vals))
            y_raw.extend(vals)
            
    if len(x_raw) > 2:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(x_raw, y_raw)
        print(f"\n--- Spearman Rank Correlation (Pre-Contact Ramp-Up) ---")
        print(f"Testing steps from min to 0 (N={len(x_raw)} individual movements)")
        print(f"Spearman rho: {corr:.3f}")
        print(f"p-value:      {pval:.4g}")
        if pval < 0.05:
            if corr > 0:
                print("Result: SIGNIFICANT INCREASE in magnitude as animals approach the hole!")
            else:
                print("Result: SIGNIFICANT DECREASE in magnitude as animals approach the hole.")
        else:
            print("Result: No significant monotonic trend leading up to the hole.")
        print("-------------------------------------------------------\n")
    
    plt.figure(figsize=(10, 6))

    plt.rcParams.update({'font.size': 14})
    
    title_str = 'Evolution of Movement Magnitudes'
    if movement_type == 'L':
        plt.plot(steps, avg_mags, '-o', color='#1f77b4', linewidth=3, label='Average L-Vector Magnitude')
        plt.fill_between(steps, avg_mags - std_mags, avg_mags + std_mags, color='#1f77b4', alpha=0.2)
        title_str = 'Evolution of Longitudinal (L) Movement Magnitudes'
    elif movement_type == 'T':
        plt.plot(steps, avg_mags, '-o', color='#d62728', linewidth=3, label='Average T-Vector Magnitude')
        plt.fill_between(steps, avg_mags - std_mags, avg_mags + std_mags, color='#d62728', alpha=0.2)
        title_str = 'Evolution of Transverse (T) Movement Magnitudes'
    else:
        plt.plot(steps, avg_mags, '-o', color='#2ca02c', linewidth=3, label='Average Vector Magnitude (L+T)')
        plt.fill_between(steps, avg_mags - std_mags, avg_mags + std_mags, color='#2ca02c', alpha=0.2)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Teeth Hole Contact')
    
    plt.xlabel('Sampling Steps relative to Teeth Hole Contact (0 = closest step)')
    plt.ylabel('Vector Magnitude (Step Distance)')
    plt.title(title_str)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
        print(f"Magnitude evolution plot saved to {output_file}")
    plt.show()

def process_all_peanuts(sites_directory, teeth_mode, show_full_context, output_file=None):
    
    # 1. Gather all relevant JSON files (excluding 'front')
    all_files = [sites_directory + '/' + el for el in os.listdir(sites_directory)]
    
    valid_files = [
        f for f in all_files 
        if 'front' not in os.path.basename(f).lower()
    ]
    
    
    # 2. Initialize the master dictionary
    master_dict = {
        'sites': [],
        'arrows': []
    }
    
    # 3. Loop through files and accumulate data
    for i, sites_path in enumerate(valid_files):
        
        try:
            # We call the function with output_file=None so it just calculates and returns data
            data = generate_folded_half_chamber_map_with_sequence(sites_path=sites_path, output_file=None, teeth_mode=teeth_mode, show_quadrants=True, show_axes=False, plot=False, show_full_context=show_full_context)
            
            if data:
                master_dict['sites'].extend(data['sites'])
                master_dict['arrows'].extend(data['arrows'])
                
        except Exception as e:
            print(f"  Error processing {os.path.basename(sites_path)}: {e}")
        
    plot_folded_half_chamber_map_from_data(
        data=master_dict,
        output_file=output_file,
        show_quadrants=True,
        show_axes=False,
        show_full_context=show_full_context
    )

def process_all_peanuts_density(sites_directory, teeth_mode, show_full_context, output_file=None, show_marginal=False):
    
    # 1. Gather all relevant JSON files (excluding 'front')
    all_files = [sites_directory + '/' + el for el in os.listdir(sites_directory)]
    
    valid_files = [
        f for f in all_files 
        if 'front' not in os.path.basename(f).lower()
    ]
    
    # 2. Initialize the master dictionary
    master_dict = {
        'sites': [],
        'arrows': []
    }
    
    # 3. Loop through files and accumulate data
    for i, sites_path in enumerate(valid_files):
        
        try:
            # We call the function with output_file=None so it just calculates and returns data
            data = generate_folded_half_chamber_map_with_sequence(sites_path=sites_path, output_file=None, teeth_mode=teeth_mode, show_quadrants=True, show_axes=False, plot=False, show_full_context=show_full_context)
            
            if data:
                master_dict['sites'].extend(data['sites'])
                # We don't really need arrows for density, but we keep it for compatibility
                master_dict['arrows'].extend(data['arrows'])
                
        except Exception as e:
            print(f"  Error processing {os.path.basename(sites_path)}: {e}")
        
    plot_density_folded_half_chamber_map_from_data(
        data=master_dict,
        output_file=output_file,
        show_quadrants=True,
        show_axes=False,
        show_full_context=show_full_context,
        show_marginal=show_marginal
    )

def compare_binary_teeth_contacts(
    sites_dir,
    ethogram_path,
    output_file=None,
    use_rectangle=True,
    clamp_dims=(0.7, 0.3),
    filter_no_contact=False
):
    """
    Counts and compares the proportion of sessions where bottom teeth vs top teeth
    touched the hole at least once.
    
    For each session, it checks if any coded 'teeth-hole contact' event was assigned
    to the bottom teeth or top teeth (using the distance-based assignment).
    
    Performs McNemar's exact test (paired test for binary proportions) and prints the
    contingency table. Generates a bar plot comparing the proportion of sessions.
    """
    import os
    import glob
    import json
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import binom

    # Find all bottom sites files
    bottom_files = sorted([
        f for f in glob.glob(os.path.join(sites_dir, "*_sites.json")) 
        if not f.endswith("_front_sites.json")
    ])

    if not bottom_files:
        print(f"No *_sites.json files found in directory: {sites_dir}")
        return None

    # Load Ethogram Events
    if not os.path.exists(ethogram_path):
        print(f"Error: Ethogram file not found at {ethogram_path}")
        return None

    try:
        with open(ethogram_path, 'rb') as f:
            etho_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading ethogram file: {e}")
        return None

    # Physical parameters
    a_true, b_true = 1.2602, 1.3749
    z_max_true = np.sqrt(a_true**2 + b_true**2)

    def peanut_radius_sim(z, a, b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    def parse_sites(raw_sites):
        valid = []
        for i, s in enumerate(raw_sites):
            if len(s) >= 4:
                try:
                    frame = int(s[3])
                    p = [float(s[0]), float(s[1]), float(s[2])]
                    valid.append({'p': p, 'frame': frame, 'orig_idx': i})
                except: pass
        valid.sort(key=lambda x: x['frame'])
        return valid

    rows_data = []

    for b_file in bottom_files:
        name = os.path.basename(b_file).replace("_sites.json", "").replace("hole_", "")
        
        # Load Bottom
        with open(b_file, 'r') as f:
            b_data = json.load(f)
            sites_bottom = b_data if isinstance(b_data, list) else b_data.get("sites", [])
            json_version = b_data.get("version", "v2") if isinstance(b_data, dict) else "v2"
            
        # Load Front (Top)
        front_file = b_file.replace("_sites.json", "_front_sites.json")
        sites_front = []
        if os.path.exists(front_file):
            with open(front_file, 'r') as f:
                fd = json.load(f)
                sites_front = fd if isinstance(fd, list) else fd.get("sites", [])
                
        valid_bottom = parse_sites(sites_bottom)
        valid_front = parse_sites(sites_front)
        
        # Find matching events in ethogram
        video_events = []
        for obs_name, obs_data in etho_data.get('observations', {}).items():
            if 'file' not in obs_data or '1' not in obs_data['file']:
                continue
            media_files = obs_data['file']['1']
            media_info = obs_data.get('media_info', {})
            events = obs_data.get('events', [])
            
            cum_time = 0.0
            for media_path in media_files:
                length = media_info.get('length', {}).get(media_path, 0)
                fps = media_info.get('fps', {}).get(media_path, 29.97)
                video_name = os.path.basename(media_path).replace('.MP4', '').replace('.mp4', '')
                
                v_start = cum_time
                v_end = cum_time + length
                cum_time += length
                
                if name.lower() == video_name.lower():
                    # Extract contact events
                    for ev in events:
                        if len(ev) < 6:
                            continue
                        ev_time = float(ev[0])
                        if v_start <= ev_time <= v_end:
                            video_events.append(ev)

        # Extract contact frames
        contact_frames = []
        for e in video_events:
            behavior_name = str(e[2]).lower()
            if 'teeth' in behavior_name and 'hole' in behavior_name and 'contact' in behavior_name:
                try:
                    contact_frames.append(int(e[5]))
                except:
                    pass

        if filter_no_contact and len(contact_frames) == 0:
            continue

        z_max_orig = np.sqrt(1.0**2 + 1.1**2)
        scale_factor = z_max_true / z_max_orig
        
        if json_version == "v3":
            p_hole = np.array([0.0, 0.0, z_max_true])
        else:
            term = np.sqrt(1.1**4 + 4 * 1.0**2 * 0.75**2)
            r_hole_orig = np.sqrt(np.maximum(0.0, term - 0.75**2 - 1.0**2))
            p_hole = np.array([0.0, r_hole_orig, 0.75]) * scale_factor

        bottom_touched = False
        top_touched = False
        
        for f_idx in contact_frames:
            pt_b = None
            closest_b_dist = float('inf')
            for s in valid_bottom:
                if abs(s['frame'] - f_idx) <= 5:
                    p_scaled = np.array([s['p'][0]*scale_factor, s['p'][1]*scale_factor, s['p'][2]*scale_factor])
                    if abs(s['frame'] - f_idx) < closest_b_dist:
                        closest_b_dist = abs(s['frame'] - f_idx)
                        pt_b = p_scaled
            
            pt_f = None
            closest_f_dist = float('inf')
            for s in valid_front:
                if abs(s['frame'] - f_idx) <= 5:
                    p_scaled = np.array([s['p'][0]*scale_factor, s['p'][1]*scale_factor, s['p'][2]*scale_factor])
                    if abs(s['frame'] - f_idx) < closest_f_dist:
                        closest_f_dist = abs(s['frame'] - f_idx)
                        pt_f = p_scaled
            
            if pt_b is not None and pt_f is not None:
                dist_b = np.linalg.norm(pt_b - p_hole)
                dist_f = np.linalg.norm(pt_f - p_hole)
                if dist_b < dist_f:
                    bottom_touched = True
                else:
                    top_touched = True
            elif pt_b is not None:
                bottom_touched = True
            elif pt_f is not None:
                top_touched = True

        rows_data.append({
            'Session': name,
            'Bottom Touched': bottom_touched,
            'Top Touched': top_touched
        })

    df = pd.DataFrame(rows_data)
    if len(df) == 0:
        print("No valid sessions found after filtering.")
        return df

    # McNemar's exact test calculation
    A = len(df[(df['Bottom Touched'] == True) & (df['Top Touched'] == True)])
    B = len(df[(df['Bottom Touched'] == True) & (df['Top Touched'] == False)])
    C = len(df[(df['Bottom Touched'] == False) & (df['Top Touched'] == True)])
    D = len(df[(df['Bottom Touched'] == False) & (df['Top Touched'] == False)])

    print(f"\nContingency Table (Paired Bottom vs Top Teeth Touch):")
    print(f"               Top Teeth Touch  No Top Teeth Touch")
    print(f"Bottom Teeth    {A:<16} {B:<18}")
    print(f"No Bottom Teeth {C:<16} {D:<18}")

    n_discordant = B + C
    if n_discordant > 0:
        p_mcnemar = min(1.0, 2.0 * binom.cdf(min(B, C), n_discordant, 0.5))
    else:
        p_mcnemar = 1.0
        
    print(f"\nMcNemar's exact test p-value: {p_mcnemar:.4f}")

    # Percentages
    pct_bottom = (df['Bottom Touched'].sum() / len(df)) * 100.0
    pct_top    = (df['Top Touched'].sum() / len(df)) * 100.0

    print(f"Proportion of sessions where Bottom Teeth touched: {pct_bottom:.1f}%")
    print(f"Proportion of sessions where Top Teeth touched:    {pct_top:.1f}%")

    # Plotting
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=(10, 8))

    # Bar plot of proportions
    bars = ax.bar(['Bottom Teeth', 'Top Teeth'], [pct_bottom, pct_top], 
                  color=['#EF4444', '#3B82F6'], edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_ylabel("Sessions with Contact (%)")
    ax.set_title(f"Proportion of Sessions with Teeth-Hole Contact\n(McNemar's test p = {p_mcnemar:.4f})", pad=20, fontweight='bold')
    
    style_plot(ax)
    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
        print(f"Saved binary comparison plot to {output_file}")

    plt.show()
    plt.close(fig)

    return df

def analyze_peanut_resistance(csv_path, output_dir=None, show_regression=False, select_days=None, metric = 'turgitdity'):
    """
    Analyzes normalized peanut resistance data from a CSV file.
    Generates a linear regression scatter plot and a violin plot.
    """

    # -----------------------------
    # Load data
    # -----------------------------
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    target_col = "Resistance normalized"

    if target_col not in df.columns:
        print(f"Error: Column '{target_col}' not found in CSV.")
        return

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    def parse_day(val):
        s = str(val)
        if s.startswith("Day "):
            return int(s.split(" ")[1])
        try:
            return int(val)
        except Exception:
            return None

    df["Day_Num"] = df["Day"].apply(parse_day)
    df = df.dropna(subset=["Day_Num"])
    df["Day_Num"] = df["Day_Num"].astype(int)

    if select_days:
        days = ["Day " + str(el) for el in select_days]
        df = df[df["Day"].isin(days)]

    df["Condition"] = df["Condition"].replace("control", "intact")

    condition_order = ["intact", "hole", "crushed"]
    palette = {"intact": "#D4B298", "hole": "#B07C65", "crushed": "#88756B"}

    max_day = int(df["Day_Num"].max())
    days_order = list(range(1, max_day + 1))

    # =============================
    # Plot 1: Scatter + regression
    # =============================
    sns.reset_defaults()
    fig1, ax1 = plt.subplots(figsize=(7, 6))

    shifts = {"intact": -0.1, "hole": 0.0, "crushed": 0.1}

    for cond in condition_order:
        subset = df[df["Condition"] == cond]
        if subset.empty:
            continue
        ax1.scatter(
            subset["Day_Num"] + shifts[cond],
            subset[target_col],
            color=palette[cond],
            s=20,
            label=cond
        )

    for cond in condition_order:
        subset = df[df["Condition"] == cond]
        if subset.empty:
            continue
        sns.regplot(
            data=subset,
            x="Day_Num",
            y=target_col,
            scatter=False,
            ax=ax1,
            color=palette[cond],
            truncate=True
        )

    ax1.set_title("Linear Regression of Normalized Resistance vs Day by Condition", fontsize=20)
    ax1.set_xlabel("Retrieval Day", fontsize=20)
    ax1.set_ylabel("Normalized Resistance", fontsize=20)

    # Force all days to appear on x-axis
    ax1.set_xticks(range(int(df["Day_Num"].min()), int(df["Day_Num"].max()) + 1))

    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim(xmin, xmax + 0.5)
    ax1.set_ylim(-0.5, 25)

    ax1.legend(title="Condition", fontsize=20, title_fontsize=20)
    ax1.legend_.remove()


    style_axes(
        ax1,
        spine_width=3,
        show_ticks=True,
        axis_color="black",
        remove_grid=True,
        force_ticks=True,
        tick_label_size=20
    )

    fig1.tight_layout()

    save_path = r'figures/putrefaction_' + metric + '.svg'
    plt.savefig(
            save_path, 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)
    
    plt.show()

    # =============================
    # Plot 2: Violin + overlays
    # =============================
    sns.set_theme(style="whitegrid")
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    sns.violinplot(
        data=df,
        x="Day_Num",
        y=target_col,
        hue="Condition",
        hue_order=condition_order,
        order=days_order,
        palette=palette,
        cut=0,
        ax=ax2,
        inner=None
    )

    sns.stripplot(
        data=df,
        x="Day_Num",
        y=target_col,
        hue="Condition",
        hue_order=condition_order,
        order=days_order,
        dodge=True,
        jitter=True,
        color="black",
        alpha=0.5,
        size=3,
        ax=ax2,
        legend=False
    )

    day_to_x = {d: i for i, d in enumerate(days_order)}
    offsets = {"intact": -0.27, "hole": 0.0, "crushed": 0.27}

    for cond in condition_order:
        subset = df[df["Condition"] == cond]
        if subset.empty:
            continue

        slope, intercept, *_ = stats.linregress(subset["Day_Num"], subset[target_col])
        x_vals = np.array(days_order)
        y_vals = intercept + slope * x_vals
        x_plot = np.array([day_to_x[d] + offsets[cond] for d in x_vals])

        y_hat = intercept + slope * subset["Day_Num"].values
        resid_sd = np.std(subset[target_col] - y_hat, ddof=1)

        ax2.fill_between(x_plot, y_vals - resid_sd, y_vals + resid_sd,
                         color=palette[cond], alpha=0.15)

        ax2.plot(x_plot, y_vals, color="black", linewidth=7)
        ax2.plot(x_plot, y_vals, color=palette[cond], linewidth=4)

    ax2.set_title("Evolution of Normalized Resistance Across Days per Condition", fontsize=20)
    ax2.set_xlabel("Retrieval Day", fontsize=20)
    ax2.set_ylabel("Normalized Resistance", fontsize=20)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:3], labels[:3], title="Condition", fontsize=20, title_fontsize=20)
    ax2.legend_.remove()

    style_axes(
        ax2,
        spine_width=3,
        tick_width=2.5,
        tick_length=7,
        show_ticks=True,
        axis_color="black",
        remove_grid=True,
        force_ticks=True,
        tick_label_size=20
    )

    ax2.set_xticks(range(len(days_order)))
    ax2.set_xticklabels(days_order, fontsize=20)

    fig2.tight_layout()
    plt.show()

    # =============================
    # Print stats
    # =============================
    print("Linear Regression Results:")
    slopes = {}
    for cond in condition_order:
        subset = df[df["Condition"] == cond]
        if subset.empty:
            continue
        slope, intercept, r_value, p_value, _ = stats.linregress(subset["Day_Num"], subset[target_col])
        slopes[cond.capitalize()] = slope
        print(f"{cond.capitalize()}: Slope = {slope:.4f}, P = {p_value:.4e}, R² = {r_value**2:.4f}")
    return slopes

def load_aligned_clamp_sequences(ethogram_path, sites_directory, teeth_mode='bottom', specific_peanut=None):
    """
    Robustly loads clamp sequences and aligns them to the first Teeth Hole Contact (THC).
    Returns a list of sequences, where each sequence is a list of point dictionaries:
    [{'p': [x, y, z], 'frame': frame, 't_rel': relative_time_in_seconds, 'video': name}, ...]
    """
    import pickle
    import os
    import glob
    import json
    
    with open(ethogram_path, 'rb') as f:
        data = pickle.load(f)
        
    all_sequences = []
    
    for obs_name, obs_data in data.get('observations', {}).items():
        if 'file' not in obs_data or '1' not in obs_data['file']:
            continue
            
        media_files = obs_data['file']['1']
        media_info = obs_data.get('media_info', {})
        events = obs_data.get('events', [])
        
        cum_time = 0.0
        for media_path in media_files:
            length = media_info.get('length', {}).get(media_path, 0)
            fps = media_info.get('fps', {}).get(media_path, 29.97)
            video_name = os.path.basename(media_path).replace('.MP4', '').replace('.mp4', '')
            
            v_start = cum_time
            v_end = cum_time + length
            cum_time += length
            
            if specific_peanut and specific_peanut not in video_name:
                continue
                
            contact_abs_time = None
            for ev in events:
                if len(ev) < 3: continue
                ev_time = float(ev[0])
                ev_type = str(ev[2]).lower().strip()
                if 'teeth' in ev_type and v_start <= ev_time <= v_end:
                    contact_abs_time = ev_time
                    break
                    
            if contact_abs_time is None:
                continue
                
            contact_local_time = contact_abs_time - v_start
            
            def _load_sites(t_mode):
                if t_mode == 'top':
                    site_file_pattern = f"*_{video_name}_front_sites.json"
                else:
                    site_file_pattern = f"*_{video_name}_sites.json"
                    
                potential_files = glob.glob(os.path.join(sites_directory, site_file_pattern))
                
                if not potential_files:
                    if t_mode == 'top':
                        site_file_pattern = f"{video_name}_front_sites.json"
                    else:
                        site_file_pattern = f"{video_name}_sites.json"
                    potential_files = glob.glob(os.path.join(sites_directory, site_file_pattern))
                    
                if not potential_files:
                    return None
                    
                if t_mode == 'top':
                    valid = [f for f in potential_files if 'front' in os.path.basename(f).lower()]
                else:
                    valid = [f for f in potential_files if 'front' not in os.path.basename(f).lower()]
                    
                if not valid:
                    return None
                    
                with open(valid[0], 'r') as f:
                    sites_dict = json.load(f)
                    
                if isinstance(sites_dict, dict) and 'sites' in sites_dict:
                    raw = sites_dict['sites']
                elif isinstance(sites_dict, dict):
                    raw = []
                    for k, v in sites_dict.items():
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                            raw.extend(v)
                else:
                    raw = sites_dict
                    
                pts = []
                for s in raw:
                    if len(s) >= 4:
                        try:
                            frame = int(s[3])
                            clamp_local_time = frame / fps
                            t_rel = clamp_local_time - contact_local_time
                            pts.append({
                                'p': [float(s[0]), float(s[1]), float(s[2])],
                                'frame': frame,
                                't_rel': t_rel,
                                'video': video_name
                            })
                        except: pass
                pts.sort(key=lambda x: x['frame'])
                return pts
                
            if teeth_mode in ['both', 'bottom']:
                cb = _load_sites('bottom')
                if cb and len(cb) > 0: all_sequences.append(cb)
            if teeth_mode in ['both', 'top', 'front']:
                cf = _load_sites('top')
                if cf and len(cf) > 0: all_sequences.append(cf)
                
    return all_sequences

