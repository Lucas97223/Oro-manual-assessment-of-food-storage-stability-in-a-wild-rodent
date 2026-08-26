import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- ENFORCE GLOBAL CLEAN STYLE ---
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_style("ticks", {"axes.edgecolor": "black"})
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


# =============================================================================
# SHARED GEODESIC UTILS
# =============================================================================
_GEODESIC_GLOBALS = {
    'mesh_tree': None,
    'adj_matrix': None,
    'L_MAX': None,
    'C_MAX': None
}

def get_geodesic_profile_distance(z1, z2, a=1.0, b=1.1):
    """Calculates the surface distance along the profile from z1 to z2."""
    from scipy.integrate import quad
    def integrand(z):
        u = z**2
        term1 = np.sqrt(b**4 + 4 * a**2 * u)
        r2 = term1 - u - a**2
        if r2 <= 1e-12: return 1.0
        r = np.sqrt(r2)
        f_prime = (2 * a**2 / term1) - 1
        dr_dz = (z / r) * f_prime
        return np.sqrt(1 + dr_dz**2)
    val, err = quad(integrand, min(z1, z2), max(z1, z2))
    return val * np.sign(z2 - z1)

def get_geodesic_total_length(a=1.0, b=1.1):
    """Calculates the full tip-to-tip surface distance."""
    z_max = np.sqrt(a**2 + b**2)
    # Using small eps to avoid singularity at tips
    return abs(get_geodesic_profile_distance(-z_max + 1e-6, z_max - 1e-6, a, b))

def style_plot(ax):
    """
    Applies custom style to the plot:
    - Only X and Y axes (remove top/right spines).
    - No grid.
    - Thicker axes (linewidth=2.5).
    - All fonts size 20.
    """
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicker axes
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2.5)
        
    # Thicker ticks and FORCE them to be visible
    ax.tick_params(width=2.5, length=8, bottom=True, left=True)
    
    # Remove Grid
    ax.grid(False)
    
    # Font Sizes
    item_size = 20
    # Bold removed as per user request
    ax.set_title(ax.get_title(), fontsize=item_size, pad=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=item_size, labelpad=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=item_size, labelpad=10)
    ax.tick_params(axis='both', labelsize=item_size)
    
    # Update Legend if exists
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_texts(), fontsize=item_size)
        plt.setp(legend.get_title(), fontsize=item_size)
        # Also make legend frame thicker? Optional.
        # frame = legend.get_frame()
        # frame.set_linewidth(2.0)



def style_axes(
    ax,
    spine_width=2.5,
    tick_width=2,
    tick_length=6,
    show_ticks=True,
    axis_color="black",
    remove_grid=False,
    force_ticks=False,
    tick_label_size=20
):
    """
    Styles axes to keep only left/bottom spines, optionally force ticks on,
    remove grid, and set axes/ticks to a specific color.
    """

    # Remove grid if requested
    if remove_grid:
        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

    # Keep only left & bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Thicker black axes
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(spine_width)
        ax.spines[side].set_color(axis_color)

    # Force ticks to exist
    if force_ticks:
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(bottom=True, left=True)

    # Style ticks
    if show_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            left=True,
            top=False,
            right=False,
            width=tick_width,
            length=tick_length,
            direction="out",
            colors=axis_color,
            labelcolor=axis_color,
            labelsize=tick_label_size
        )
    else:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            top=False,
            right=False,
            length=0,
            labelsize=tick_label_size
        )


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


# ==============================================================================
# PEANUT ROTATION ANALYSIS AND VISUALIZATION
# ==============================================================================

import json
import pickle
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree

def peanut_radius(z, a=1.0, b=1.1):
    term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
    r2 = term1 - z**2 - a**2
    return np.sqrt(np.maximum(0, r2))

def generate_peanut_mesh(a=1.0, b=1.1, num_z=100, num_theta=100):
    z_max = np.sqrt(a**2 + b**2)
    z = np.linspace(-z_max * 0.99, z_max * 0.99, num_z)
    theta = np.linspace(0, 2*np.pi, num_theta)
    Z, Theta = np.meshgrid(z, theta)
    R = peanut_radius(Z, a, b)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, Z

def build_mesh_graph(X, Y, Z):
    rows, cols = X.shape
    num_nodes = rows * cols
    adj = lil_matrix((num_nodes, num_nodes))
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    get_idx = lambda r, c: r * cols + c
    
    for r in range(rows):
        for c in range(cols):
            u = get_idx(r, c)
            neighbors = []
            if c < cols - 1:
                nc = c + 1
                neighbors.append((r, nc))
            nr = (r + 1) % rows
            neighbors.append((nr, c))
            for nr_idx, nc_idx in neighbors:
                v = get_idx(nr_idx, nc_idx)
                d = np.linalg.norm(points[u] - points[v])
                adj[u, v] = d
                adj[v, u] = d
    return adj.tocsr(), points




















































def plot_sequence_for_peanut_two_faces(sites_path, ethogram_path, show_bottom=True, show_front=True, marker_total_area=0.5, output_file = None, show_lines=True, simple_lines=False, bottom_color='red', front_color='blue'):
    """
    Plots sequential clamping sites on an unfolded peanut map using TWO FACES (Front/Back)
    instead of four quadrants.
    
    Face 0: Azimuth 0 (Front)
    Face 1: Azimuth 180 (Back)
    """
    if not os.path.exists(sites_path):
        print(f"Sites file not found: {sites_path}")
        return
    
    # Load Bottom
    sites_bottom = []
    json_version = "v2"  # default
    if show_bottom:
        with open(sites_path, 'r') as f:
            data = json.load(f)
            sites_bottom = data if isinstance(data, list) else data.get("sites", [])
            if isinstance(data, dict):
                json_version = data.get("version", "v2")

    # Load Front
    sites_front = []
    if show_front:
        path_base = sites_path.replace("_sites.json", "")
        front_path = path_base + "_front_sites.json"
        if os.path.exists(front_path):
            with open(front_path, 'r') as f:
                fd = json.load(f)
                sites_front = fd if isinstance(fd, list) else fd.get("sites", [])
        else:
            print(f"Front sites file not found: {front_path}")

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

    valid_bottom = parse_sites(sites_bottom)
    valid_front = parse_sites(sites_front)
    
    if not valid_bottom and not valid_front:
        print("No valid sites found to plot.")
        return

    # Load Ethogram Events
    events = []
    if os.path.exists(ethogram_path):
        video_base = os.path.basename(sites_path).replace("_sites.json", "").replace("_front", "").replace("hole_", "")
        try:
            with open(ethogram_path, 'rb') as f:
                etho_data = pickle.load(f)
            
            # 1. Try standard observation matching
            found_obs = None
            for video_key, obs in etho_data['observations'].items():
                if video_base in video_key:
                    found_obs = obs
                    events = obs.get('events', [])
                    break
            
            # 2. If not found directly, check if it's the continuous playlist "First_batch"
            if not found_obs and 'First_batch' in etho_data['observations']:
                obs = etho_data['observations']['First_batch']
                media_list = obs.get('file', {}).get('1', [])
                media_lengths = obs.get('media_info', {}).get('length', {})
                
                # Calculate start and end time for each video in the playlist
                current_time = 0.0
                video_info = {}
                for m in media_list:
                    length = media_lengths.get(m, 0.0)
                    m_base = os.path.basename(m).replace(".MP4", "").replace(".mp4", "")
                    video_info[m_base] = {
                        'start': current_time,
                        'end': current_time + length
                    }
                    current_time += length
                
                # Match video_base with m_base
                matched_video = None
                for m_base in video_info.keys():
                    if m_base in video_base or video_base in m_base:
                        matched_video = m_base
                        break
                
                if matched_video:
                    info = video_info[matched_video]
                    start_t, end_t = info['start'], info['end']
                    raw_events = obs.get('events', [])
                    for e in raw_events:
                        t = float(e[0])
                        if start_t <= t <= end_t:
                            # Shift time to be relative to this video's start
                            local_e = list(e)
                            local_e[0] = t - start_t
                            events.append(local_e)
        except: pass

    # Plot Setup
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # True Physical Dimensions
    a_true, b_true = 1.2602, 1.3749
    
    # Original Collection Dimensions
    is_hole = 'hole' in os.path.basename(sites_path).lower() or 'hole' in os.path.dirname(sites_path).lower()
    if is_hole:
        a_orig, b_orig = 1.0, 1.1
    else:
        a_orig, b_orig = 1.0, 1.07
    z_max_orig = np.sqrt(a_orig**2 + b_orig**2)
    
    # Local peanut_radius definition for true physical dimensions
    def peanut_radius_local(z, a=a_true, b=b_true):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    z_max_true = np.sqrt(a_true**2 + b_true**2)
    z_vals = np.linspace(-z_max_true, z_max_true, 200)
    r_vals = peanut_radius_local(z_vals, a_true, b_true)
    
    # Width = r * (pi/2) - Covers +/- 90 degrees horizontally
    width_vals = r_vals * (np.pi / 2) 
    
    S = 1.85  # Spacing between faces (reduced to make them almost touch)
    faces = {0: {'az': 0, 'cx': -S/1.5, 'cy': 0},
             1: {'az': 180, 'cx': S/1.5, 'cy': 0}}

    def map_to_canvas(p):
        x, y, z = p
        
        # --- SCALE COORDINATES TO TRUE PHYSICAL MODEL ---
        # Scale Z proportionally
        z_scaled = z * (z_max_true / z_max_orig)
        # Recalculate radius at this new Z, maintain original ratio
        r_orig = np.sqrt(x**2 + y**2)
        max_r_orig = np.sqrt((b_orig**4) ** 0.5) # simplify
        
        # Simple uniform scaling might distort if it's not a sphere, 
        # let's just use simple uniform scale factor based on z_max difference:
        scale_factor = z_max_true / z_max_orig
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        z_scaled = z * scale_factor

        if is_hole:
            # Rotate -90 degrees around Z axis: (x, y) -> (y, -x)
            x_scaled, y_scaled = y_scaled, -x_scaled

        angle = np.degrees(np.arctan2(y_scaled, x_scaled)) % 360
        
        # Find closest face
        best_diff = 360; best_fid = -1
        for fid, f in faces.items():
            diff = abs(angle - f['az'])
            if diff > 180: diff = 360 - diff
            if diff < best_diff: best_diff = diff; best_fid = fid
            
        f = faces[best_fid]
        raw_diff = angle - f['az']
        if raw_diff > 180: raw_diff -= 360
        elif raw_diff < -180: raw_diff += 360
        
        delta_rad = np.radians(raw_diff)
        curr_r = np.sqrt(x_scaled**2 + y_scaled**2)
        arc_dist = curr_r * delta_rad
        
        # Vertical Map: X = cx + arc_dist, Y = cy + z_scaled
        return f['cx'] + arc_dist, f['cy'] + z_scaled

    # Draw Backgrounds (Vertical Orientation) using TRUE dimensions
    for fid, f in faces.items():
        cx, cy = f['cx'], f['cy']
        # Vertical: Z is Y-axis, Arc is X-axis
        ax.plot(cx + width_vals, cy + z_vals, 'k-')
        ax.plot(cx - width_vals, cy + z_vals, 'k-')
        ax.fill_betweenx(cy + z_vals, cx - width_vals, cx + width_vals, color='#a6806d', alpha=0.5)
        
        if is_hole:
            if json_version == "v3":
                # Hole is at the tip (positive Z). In the app, it's drawn from z_max - 0.20 to z_max.
                # We scale it to the true model dimensions.
                z_hole_min = z_max_true - 0.20 * (z_max_true / z_max_orig)
                z_hole_max = z_max_true
                z_hole_vals = np.linspace(z_hole_min, z_hole_max, 50)
                r_hole_vals = peanut_radius_local(z_hole_vals, a_true, b_true)
                width_hole_vals = r_hole_vals * (np.pi / 2)
                ax.fill_betweenx(cy + z_hole_vals, cx - width_hole_vals, cx + width_hole_vals, color='black', alpha=0.85, zorder=8)
            else:
                # Hole is at z_orig = 0.75, theta = np.pi/2 (on the right of Front face and left of Back face)
                z_hole_orig = 0.75
                # Recalculate r at z_hole_orig in original system (a=1.0, b=1.1 for hole sessions)
                def r_orig_func(z):
                    term1 = np.sqrt(1.21 + 4.0 * z**2)
                    r2 = term1 - z**2 - 1.0
                    return np.sqrt(np.maximum(0, r2))
                
                r_hole_orig = r_orig_func(z_hole_orig)
                p_hole = [0.0, r_hole_orig, z_hole_orig]
                
                # Map this point using the standard map_to_canvas (which uses faces and z_max_orig)
                x_h, y_h = map_to_canvas(p_hole)
                
                # Draw the hole circle if it maps to the current face
                if abs(x_h - cx) < S:
                    from matplotlib.patches import Circle
                    c_hole = Circle((x_h, y_h), radius=0.15, facecolor='black', edgecolor='black', zorder=18)
                    ax.add_patch(c_hole)
            
        label = f"Front ({f['az']}°)" if fid == 0 else f"Back ({f['az']}°)"
        ax.text(cx, cy + z_max_true + 0.5, label, ha='center', fontweight='bold')

    rotation_colors = {'Vertical Rotation': 'blue', 'Horizontal Rotation': 'orange', 'Roll': 'green', 'Translation': 'purple'}
    
    mapped_bottom = [map_to_canvas(s['p']) for s in valid_bottom]
    mapped_front = [map_to_canvas(s['p']) for s in valid_front]
    
    def draw_sequence_lines(sites, mapped_pts, line_color='black'):
        for i in range(len(sites) - 1):
            s1, s2 = sites[i], sites[i+1]
            p1, p2 = mapped_pts[i], mapped_pts[i+1]
            f1, f2 = s1['frame'], s2['frame']
            found_type = None
            for e in events:
                try: e_frame = int(e[-1])
                except: e_frame = -1
                if e_frame > f1 and e_frame < f2 and e[2] in rotation_colors:
                    if found_type is None: found_type = e[2]
            
            if found_type and not simple_lines:
                color = rotation_colors.get(found_type, 'black'); ls, alpha, w = '-', 0.8, 3
            else:
                color = line_color; ls = '--'; alpha = 0.5; w = 2
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=ls, linewidth=w, alpha=alpha, zorder=15)

    if show_lines:
        if show_bottom: draw_sequence_lines(valid_bottom, mapped_bottom, line_color=bottom_color)
        if show_front: draw_sequence_lines(valid_front, mapped_front, line_color=front_color)

    from matplotlib.patches import Circle

    # Extract Teeth-Hole Contact frames from events
    contact_frames = []
    for e in events:
        behavior_name = str(e[2]).lower()
        if 'teeth' in behavior_name and 'hole' in behavior_name and 'contact' in behavior_name:
            try:
                contact_frames.append(int(e[5]))
            except: pass

    # Red (Bottom)
    if show_bottom and mapped_bottom:
        n_b = len(mapped_bottom)
        r_b = np.sqrt((marker_total_area / n_b) / np.pi) if n_b > 0 else 0.1
        for i, (X, Y) in enumerate(mapped_bottom):
            site_frame = valid_bottom[i]['frame']
            is_contact = any(abs(site_frame - cf) <= 5 for cf in contact_frames)
            if is_contact and is_hole and json_version != "v3":
                if X > 0:  # Back face, opposite to the hole
                    is_contact = False
            color = 'green' if is_contact else bottom_color
            c = Circle((X, Y), radius=r_b, facecolor=color, edgecolor='black', zorder=20, alpha=0.7)
            ax.add_patch(c)
            # Reduce font size heavily if crowded, or rely on zoom
            ax.text(X, Y, str(valid_bottom[i]['orig_idx']), fontsize=14, color='white', fontweight='bold', ha='center', va='center', zorder=25)

    # Blue (Front)
    if show_front and mapped_front:
        n_f = len(mapped_front)
        r_f = np.sqrt((marker_total_area / n_f) / np.pi) if n_f > 0 else 0.1
        for i, (X, Y) in enumerate(mapped_front):
            site_frame = valid_front[i]['frame']
            is_contact = any(abs(site_frame - cf) <= 5 for cf in contact_frames)
            if is_contact and is_hole and json_version != "v3":
                if X > 0:  # Back face, opposite to the hole
                    is_contact = False
            color = 'green' if is_contact else front_color
            c = Circle((X, Y), radius=r_f, facecolor=color, edgecolor='black', zorder=20, alpha=0.7)
            ax.add_patch(c)
            ax.text(X, Y, str(valid_front[i]['orig_idx']), fontsize=14, color='white', fontweight='bold', ha='center', va='center', zorder=25)

    ax.set_aspect('equal')
    ax.set_xlim(-S*1.5, S*1.5)
    ax.set_ylim(-3.5, 3.5) # Increased Y limit for vertical peanut length
    ax.axis('off')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
        print(f"Saved two-face sequence map to {output_file} (and SVG)")
    plt.show()
    plt.close(fig)


















def visualize_vector_analysis(sites_dir, mode='population', teeth_mode='bottom', specific_peanut=None, output_file=None, return_stats=False, absolute_val=True):
    """
    Analyzes vectors using separated chains.
    Generates Rose Plot.
    teeth_mode: 'both', 'bottom', 'front'.
    """
    if not os.path.exists(sites_dir):
        print(f"Directory not found: {sites_dir}")
        return

    # Handle 'mode' as specific peanut if not 'population'
    if mode != 'population' and specific_peanut is None:
        specific_peanut = mode

    def peanut_radius(z, a=1.0, b=1.1):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    files = [f for f in os.listdir(sites_dir) if f.endswith("_sites.json") and "_front" not in f]
    
    # Filter for Specific Peanut if requested
    if specific_peanut:
        files = [f for f in files if str(specific_peanut) in f]
        if not files:
            print(f"No files found matching specific peanut: {specific_peanut}")
            return
    
    all_dz = []
    all_darc = []
    
    for f_name in files:
        chains_to_process = []
        
        # Load Bottom
        if teeth_mode in ['both', 'bottom']:
            try:
                path = os.path.join(sites_dir, f_name)
                with open(path, 'r') as f:
                    d = json.load(f)
                    sites_b = [{'p': [float(s[0]), float(s[1]), float(s[2])]} for s in d.get('sites', []) if len(s) >= 3]
                chains_to_process.append(sites_b)
            except: pass
        
        # Load Front
        if teeth_mode in ['both', 'front']:
            try:
                front_path = os.path.join(sites_dir, f_name.replace("_sites.json", "_front_sites.json"))
                if os.path.exists(front_path):
                    with open(front_path, 'r') as f:
                        d = json.load(f)
                        sites_f = [{'p': [float(s[0]), float(s[1]), float(s[2])]} for s in d.get('sites', []) if len(s) >= 3]
                    chains_to_process.append(sites_f)
            except: pass
            
        # Process Chains
        for chain in chains_to_process:
            for i in range(len(chain) - 1):
                p1 = chain[i]['p']; p2 = chain[i+1]['p']
                dz = get_geodesic_profile_distance(p1[2], p2[2], a=1.0, b=1.1)
                avg_z = (p1[2] + p2[2]) / 2
                r = peanut_radius(avg_z)
                ang1 = np.arctan2(p1[1], p1[0]); ang2 = np.arctan2(p2[1], p2[0])
                delta_ang = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
                darc = r * delta_ang
                all_dz.append(dz)
                all_darc.append(darc)

    if not all_dz: return

    # Option: Absolute Values (Fold into First Quadrant)
    if absolute_val:
        all_dz = np.abs(all_dz)
        all_darc = np.abs(all_darc)

    # Rose Plot
    angles = np.arctan2(all_dz, all_darc) # y=Z, x=Arc
    
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    if absolute_val == False:
        bins = 20
    else:
        bins = 10
        
    counts, bin_edges, patches = ax.hist(angles, bins=bins, color='gray', alpha=1, edgecolor='black', linewidth=1, zorder = 2)
    
    # ------------------
    # Statistical Significance (Uniformity)
    # ------------------
    total_vectors = len(all_dz)
    expected_per_bin = total_vectors / bins
    
    # Global Chi-square (Uniformity)
    chi2_stat, p_val = stats.chisquare(counts, f_exp=expected_per_bin)
    print(f"\n--- Rose Plot Uniformity Stats ---")
    print(f"Total Vectors: {total_vectors}")
    print(f"Chi-square Stat: {chi2_stat:.3f}, p-value: {p_val:.4g}")
    
    if p_val < 0.05:
        print("Result: Distribution is significantly different from uniform (biased).")
        
        # --- Post-hoc: Bin-wise Residual Analysis ---
        # Adjusted Standardized Residuals: Z = (O-E)/sqrt(E*(1-1/bins))
        std_resid = (counts - expected_per_bin) / np.sqrt(expected_per_bin * (1 - 1/bins))
        p_vals_resid = norm.sf(np.abs(std_resid)) * 2
        reject_resid, padj_resid, _, _ = multipletests(p_vals_resid, method='holm')
        
        print(f"\nPost-hoc Bin Analysis (vs Uniform):")
        print(f"  {'Range':<15} {'Count':<8} {'p_uncorr':<10} {'p_adj':<10} {'Sig'}")
        print(f"  {'-'*55}")
        for i, (rej, p_adj, p_raw) in enumerate(zip(reject_resid, padj_resid, p_vals_resid)):
            deg1 = np.degrees(bin_edges[i]); deg2 = np.degrees(bin_edges[i+1])
            range_str = f"[{deg1:.0f}-{deg2:.0f}°]"
            sig_str = "*" if rej else ("+" if p_raw < 0.05 else "n.s.")
            print(f"  {range_str:<15} {counts[i]:<8.0f} {p_raw:<10.4g} {p_adj:<10.4g} {sig_str}")
            
            if rej:
                # Add star to the plot for truly significant (corrected)
                center = (bin_edges[i] + bin_edges[i+1]) / 2
                ax.text(center, counts[i] + max(counts)*0.05, "*", ha='center', va='center', fontsize=20, color='red', fontweight='bold')
            elif p_raw < 0.05:
                # Add a small '+' for nominally significant (uncorrected)
                center = (bin_edges[i] + bin_edges[i+1]) / 2
                ax.text(center, counts[i] + max(counts)*0.05, "+", ha='center', va='center', fontsize=15, color='orange')
    else:
        print("Result: Distribution is not significantly different from uniform.")

    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['+Arc', '+Z', '-Arc', '-Z'])
    # ax.set_yticklabels([]) # Replaced to show labels as requested
    ax.tick_params(axis='y', labelsize=15, colors='gray') # Style the grid labels
    ax.grid(True, linestyle=':', alpha=1)

    # --- Add Peanut Shape (Unfolded Map Boundary) for Context ---
    # Define boundary in (Z, Arc) space and map to (Theta, R)
    z_max_shape = np.sqrt(1.0**2 + 0**2)
    z_vals = np.linspace(-z_max_shape - 1, z_max_shape, 200)
    r_vals = peanut_radius(z_vals)
    
    # Boundary: Right (+pi) and Left (-pi) edges
    z_right = z_vals; arc_right = r_vals * np.pi
    z_left = z_vals[::-1]; arc_left = -r_vals[::-1] * np.pi
    
    z_shape = np.concatenate([z_right, z_left, [z_right[0]]])
    arc_shape = np.concatenate([arc_right, arc_left, [arc_right[0]]])
    
    # Convert to Plot Coordinates (Theta, R_physical)
    # Theta = arctan2(Z, Arc)
    theta_shape = np.arctan2(z_shape, arc_shape)
    r_shape_physical = np.sqrt(z_shape**2 + arc_shape**2)
    
    # Scale to fit plot (Align Z-extent with Axis Radial Limit)
    radial_limit = ax.get_ylim()[1]
    scale_factor = radial_limit / z_max_shape 

    if absolute_val == True:
        for angle in [0, np.pi/2]:
            ax.axvline(angle, color='black', linewidth=2, zorder=5)
    
    #ax.plot(theta_shape, r_shape_physical * scale_factor, 'r-', linewidth=1.5, alpha=0.7, label='Peanut Shape')
    #ax.fill(theta_shape, r_shape_physical * scale_factor, 'red', alpha=0.05)
    
    avg_dz = np.mean(np.abs(all_dz))
    avg_darc = np.mean(np.abs(all_darc))
    ratio = avg_dz / avg_darc if avg_darc > 0 else 0
    
    title_suffix = ""
    if teeth_mode == 'bottom': title_suffix = "\n(Bottom Teeth Only)"
    if teeth_mode == 'front': title_suffix = "\n(Front Teeth Only)"
    
    
    stats_text = f"Longitudinal Bias: {ratio:.2f}x\n(Avg dZ / Avg dArc)"
    #plt.figtext(0.02, 0.95, stats_text, fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    title_text = f"Sampling Direction"
    if specific_peanut: title_text += f"\n({specific_peanut})"
    title_text += f"{title_suffix}"
    
    #plt.title(title_text, y=1.08, fontsize=16)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    
    plt.show()
    plt.close(fig)
        
    if return_stats:
        df_stats = pd.DataFrame({
            'Bin_Start_Rad': bin_edges[:-1],
            'Bin_End_Rad': bin_edges[1:],
            'Bin_Center_Rad': (bin_edges[:-1] + bin_edges[1:]) / 2,
            'Count': counts,
            'Expected': expected_per_bin
        })
        return df_stats


# =============================================================================
# SHARED GEODESIC UTILS
# =============================================================================
def init_geodesic_resources():
    if _GEODESIC_GLOBALS['mesh_tree'] is not None:
        return
        
    print("Building Mesh Graph for Geodesic Distance (Shared)...")
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import shortest_path
    
    # generate_peanut_mesh and build_mesh_graph must be defined above
    X_m, Y_m, Z_m = generate_peanut_mesh(num_z=100, num_theta=100)
    adj_matrix, mesh_points = build_mesh_graph(X_m, Y_m, Z_m)
    
    _GEODESIC_GLOBALS['adj_matrix'] = adj_matrix
    _GEODESIC_GLOBALS['mesh_tree'] = cKDTree(mesh_points)
    
    # Calculate Constants (Geodesic)
    # Model parameters used in generate_peanut_mesh above
    a_mod, b_mod = 1.0, 1.1
    L_MAX = get_geodesic_total_length(a_mod, b_mod)
    
    # C_MAX is circumference at the peak radius
    r_peak = b_mod**2 / (2 * a_mod)
    C_MAX = 2 * np.pi * r_peak
    
    _GEODESIC_GLOBALS['L_MAX'] = L_MAX
    _GEODESIC_GLOBALS['C_MAX'] = C_MAX
    # print(f"Stats (Geodesic): L_max={L_MAX:.4f}, C_max={C_MAX:.4f}")

def get_vectors_geodesic(sites, normalize=False):
    """
    Computes vectors with Geodesic Dimensions.
    Uses shared globals.
    """
    init_geodesic_resources()
    
    mesh_tree = _GEODESIC_GLOBALS['mesh_tree']
    adj_matrix = _GEODESIC_GLOBALS['adj_matrix']
    L_MAX = _GEODESIC_GLOBALS['L_MAX']
    C_MAX = _GEODESIC_GLOBALS['C_MAX']
    
    from scipy.sparse.csgraph import shortest_path
    vecs = []
    
    for i in range(len(sites) - 1):
        p1_raw = sites[i]['p']; p2_raw = sites[i+1]['p']
        
        # Geodesic
        d, idx1 = mesh_tree.query(p1_raw)
        _, idx2 = mesh_tree.query(p2_raw)
        
        dist_arr = shortest_path(adj_matrix, indices=idx1, directed=False)
        mag = dist_arr[idx2]
        
        # Components
        z1, z2 = p1_raw[2], p2_raw[2]
        dz = get_geodesic_profile_distance(z1, z2, a=1.0, b=1.1)
        avg_z = (z1 + z2) / 2
        r = peanut_radius(avg_z)
        ang1 = np.arctan2(p1_raw[1], p1_raw[0])
        ang2 = np.arctan2(p2_raw[1], p2_raw[0])
        delta_ang = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
        darc = r * delta_ang
        
        is_long = abs(dz) >= abs(darc)
        color = '#1f77b4' if is_long else '#d62728' 
        
        if normalize:
            factor = L_MAX if is_long else C_MAX
            mag = mag / factor
            dz = dz / factor
            darc = darc / factor
            
        vecs.append({'dz': dz, 'darc': darc, 'mag': mag, 'color': color, 'is_long': is_long})
        
    return vecs


def visualize_arrow_sequence(sites_path, show_bottom=True, show_front=True, output_file=None, normalize_amplitude=False):
    """
    Visualizes sequence of movements as parallel tracks.
    Colors arrows by Movement Type: 
    - Blue = Longitudinal (|dZ| >= |dArc|)
    - Red = Transverse (|dZ| < |dArc|)
    """
    if not os.path.exists(sites_path):
        print(f"File not found: {sites_path}")
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import json
    import matplotlib.lines as mlines


    vectors_b = []
    if show_bottom:
        try:
            with open(sites_path, 'r') as f:
                d = json.load(f)
                sites_b = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3]), 'type': 'Bottom'} for s in d.get('sites', []) if len(s) >= 4]
            sites_b.sort(key=lambda x: x['frame'])
            vectors_b = get_vectors_geodesic(sites_b, normalize=normalize_amplitude)
        except: pass
        
    vectors_f = []
    if show_front:
        front_path = sites_path.replace("_sites.json", "_front_sites.json")
        if os.path.exists(front_path):
            try:
                with open(front_path, 'r') as f:
                    d = json.load(f)
                    sites_f = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3]), 'type': 'Front'} for s in d.get('sites', []) if len(s) >= 4]
                sites_f.sort(key=lambda x: x['frame'])
                vectors_f = get_vectors_geodesic(sites_f, normalize=normalize_amplitude)
            except: pass

    if not vectors_b and not vectors_f: return

    # Plot
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})
    max_steps = max(len(vectors_b), len(vectors_f))
    
    n_tracks = (1 if show_bottom else 0) + (1 if show_front else 0)
    
    # Use subplots to give each track a real Y-axis
    fig, axes = plt.subplots(nrows=n_tracks, ncols=1, sharex=True, squeeze=False, 
                             figsize=(max(10, max_steps * 0.5), 4 * n_tracks))
    axes = axes.flatten()
    
    track_idx = 0
    
    if show_bottom:
        ax = axes[track_idx]
        track_idx += 1
        
        ax.set_ylabel("Bottom", fontsize=16, fontweight='bold')
        ax.axhline(0, color='lightgray', linewidth=2, zorder=0)

        for i, v in enumerate(vectors_b):
            c = v['color']
            mag = v['mag']
            
            hw = 0.03 if normalize_amplitude else 0.15
            hl = 0.04 if normalize_amplitude else 0.15
            
            if mag < hl:
                 hl = mag * 0.5
                 hw = hl * 0.6
                 
            ax.arrow(i, 0, v['darc'], v['dz'], head_width=hw, head_length=hl, fc=c, ec=c, length_includes_head=True, linewidth=4)
            
        ax.set_aspect('equal')
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if show_front:
        ax = axes[track_idx]
        track_idx += 1
        
        ax.set_ylabel("Front", fontsize=16, fontweight='bold')
        ax.axhline(0, color='lightgray', linewidth=2, zorder=0)

        for i, v in enumerate(vectors_f):
            c = v['color']
            mag = v['mag']
            
            hw = 0.03 if normalize_amplitude else 0.15
            hl = 0.04 if normalize_amplitude else 0.15
            
            if mag < hl:
                 hl = mag * 0.5
                 hw = hl * 0.6
            
            ax.arrow(i, 0, v['darc'], v['dz'], head_width=hw, head_length=hl, fc=c, ec=c, length_includes_head=True, linewidth=2)

        ax.set_aspect('equal')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.xlabel("Step Index")
    from matplotlib.ticker import MultipleLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1))

    plt.ylim(-1, 1)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    max_steps = max(len(vectors_b), len(vectors_f))
    







def visualize_transition_graph(sites_dir, mode='population', teeth_mode='bottom', specific_peanut=None, output_file=None):
    """
    Visualizes transition probabilities between Longitudinal (L) and Transverse (T) movements.
    Generates a directed graph with probabilities.
    teeth_mode: 'both', 'bottom', 'front'.
    """
    if not os.path.exists(sites_dir):
        print(f"Directory not found: {sites_dir}")
        return

    import matplotlib.patches as patches
    
    # Handle 'mode' as specific peanut if not 'population'
    if mode != 'population' and specific_peanut is None:
        specific_peanut = mode
    
    def peanut_radius(z, a=1.0, b=1.1):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    files = [f for f in os.listdir(sites_dir) if f.endswith("_sites.json") and "_front" not in f]
    
    # Filter for Specific Peanut
    if specific_peanut:
        files = [f for f in files if str(specific_peanut) in f]
        if not files:
            print(f"No files found matching specific peanut: {specific_peanut}")
            return
    
    transitions = {'LL': 0, 'LT': 0, 'TL': 0, 'TT': 0}
    counts = {'L': 0, 'T': 0}

    for f_name in files:
        chains_to_process = []
        
        # Load Bottom
        if teeth_mode in ['both', 'bottom']:
            try:
                path = os.path.join(sites_dir, f_name)
                with open(path, 'r') as f:
                    d = json.load(f)
                    sites_b = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3])} for s in d.get('sites', []) if len(s) >= 4]
                sites_b.sort(key=lambda x: x['frame'])
                chains_to_process.append(sites_b)
            except: pass
        
        # Load Front
        if teeth_mode in ['both', 'front']:
            try:
                front_path = os.path.join(sites_dir, f_name.replace("_sites.json", "_front_sites.json"))
                if os.path.exists(front_path):
                    with open(front_path, 'r') as f:
                        d = json.load(f)
                        sites_f = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3])} for s in d.get('sites', []) if len(s) >= 4]
                    sites_f.sort(key=lambda x: x['frame'])
                    chains_to_process.append(sites_f)
            except: pass

        for chain in chains_to_process:
            seq = []
            for i in range(len(chain) - 1):
                p1 = chain[i]['p']; p2 = chain[i+1]['p']
                dz = p2[2] - p1[2]
                avg_z = (p1[2] + p2[2]) / 2
                r = peanut_radius(avg_z)
                ang1 = np.arctan2(p1[1], p1[0]); ang2 = np.arctan2(p2[1], p2[0])
                darc = r * ((ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi)
                
                if abs(dz) >= abs(darc): seq.append('L')
                else: seq.append('T')
            
            for i in range(len(seq) - 1):
                pair = seq[i] + seq[i+1]
                if pair in transitions:
                    transitions[pair] += 1
                    counts[seq[i]] += 1

    if counts['L'] == 0 and counts['T'] == 0: return

    # Calculate Probabilities
    prob = {}
    prob['LL'] = transitions['LL'] / counts['L'] if counts['L'] > 0 else 0
    prob['LT'] = transitions['LT'] / counts['L'] if counts['L'] > 0 else 0
    prob['TL'] = transitions['TL'] / counts['T'] if counts['T'] > 0 else 0
    prob['TT'] = transitions['TT'] / counts['T'] if counts['T'] > 0 else 0

    # Plot Graph
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # Nodes
    rad = 0.1
    circle_L = patches.Circle((0.3, 0.5), rad, facecolor='#1f77b4', edgecolor='black', linewidth=2, alpha=0.8)
    circle_T = patches.Circle((0.7, 0.5), rad, facecolor='#d62728', edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(circle_L)
    ax.add_patch(circle_T)

    ax.text(0.3, 0.5, "Longitudinal\n(L)", ha='center', va='center', color='white', fontweight='bold', fontsize=14)
    ax.text(0.7, 0.5, "Transverse\n(T)", ha='center', va='center', color='white', fontweight='bold', fontsize=14)

    # Edges
    style = "Simple, tail_width=2, head_width=10, head_length=10"
    
    # L -> T
    ax.annotate("", xy=(0.7, 0.5), xycoords='data', xytext=(0.3, 0.5), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", lw=2))
    ax.text(0.5, 0.6, f"{prob['LT']:.2f}", ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # T -> L
    ax.annotate("", xy=(0.3, 0.5), xycoords='data', xytext=(0.7, 0.5), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", lw=2))
    ax.text(0.5, 0.4, f"{prob['TL']:.2f}", ha='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Self Loops (Annotations)
    ax.text(0.15, 0.5, f"{prob['LL']:.2f}\n(Stay)", ha='center', va='center', color='#1f77b4', fontweight='bold', fontsize=12)
    ax.text(0.85, 0.5, f"{prob['TT']:.2f}\n(Stay)", ha='center', va='center', color='#d62728', fontweight='bold', fontsize=12)
    
    title_suffix = ""
    if teeth_mode == 'bottom': title_suffix = "\\n(Bottom Teeth Only)"
    if teeth_mode == 'front': title_suffix = "\\n(Front Teeth Only)"
    if specific_peanut: title_suffix += f" - {specific_peanut}"
    
    plt.title(f"Transition Probabilities{title_suffix}", fontsize=20)
    
    if output_file: 
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def visualize_step_size_analysis(sites_dir, mode='population', teeth_mode='bottom', specific_peanut=None, return_stats=False, output_file=None, normalize_amplitude=False):
    """
    Analyzes step sizes across the population, classifying them as Longitudinal (> Transverse) or Transverse.
    Generates a Box Plot with Jitter and optional Statistics.
    teeth_mode: 'both', 'bottom', 'front'.
    """
    if not os.path.exists(sites_dir):
        print(f"Directory not found: {sites_dir}")
        return

    # Handle 'mode' as specific peanut if not 'population'
    if mode != 'population' and specific_peanut is None:
        specific_peanut = mode

    files = [f for f in os.listdir(sites_dir) if f.endswith("_sites.json") and "_front" not in f]
    
    # Filter for Specific Peanut
    if specific_peanut:
        files = [f for f in files if str(specific_peanut) in f]
        if not files:
            print(f"No files found matching specific peanut: {specific_peanut}")
            return

    chains_to_process = []
    for f_name in files:
        # Load Bottom
        if teeth_mode in ['both', 'bottom']:
            try:
                path = os.path.join(sites_dir, f_name)
                with open(path, 'r') as f:
                    d = json.load(f)
                    sites_b = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3]), 'type': 'Bottom'} for s in d.get('sites', []) if len(s) >= 4]
                sites_b.sort(key=lambda x: x['frame'])
                chains_to_process.append(sites_b)
            except: pass
        
        # Load Front
        if teeth_mode in ['both', 'front']:
            try:
                front_path = os.path.join(sites_dir, f_name.replace("_sites.json", "_front_sites.json"))
                if os.path.exists(front_path):
                    with open(front_path, 'r') as f:
                        d = json.load(f)
                        sites_f = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3]), 'type': 'Front'} for s in d.get('sites', []) if len(s) >= 4]
                    sites_f.sort(key=lambda x: x['frame'])
                    chains_to_process.append(sites_f)
            except: pass

    # --- Plotting Helper ---
    def draw_standard_plot(ax, d_long, d_trans, is_norm):
        u, p = stats.mannwhitneyu(d_long, d_trans)
        data = [d_long, d_trans]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
        
        colors = ['#1f77b4', '#d62728'] # Original Blue/Red
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(1) # Original solid
            
        # Jitter (Original style)
        np.random.seed(42)
        for i, d in enumerate(data):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=1, s=15, color='black', zorder=3)
            
        # Significance (Original style)
        max_val = max(max(d_long), max(d_trans))
        y_bracket = max_val * 1.05
        y_bar = max_val * 1.05 + (max_val * 0.03) 
        h = max_val * 0.02
        ax.plot([1, 1, 2, 2], [y_bracket, y_bar, y_bar, y_bracket], lw=1.5, c='black')
        
        sig_text = "n.s."
        if p < 0.001: sig_text = "***"
        elif p < 0.01: sig_text = "**"
        elif p < 0.05: sig_text = "*"
        ax.text(1.5, y_bar + h/2, sig_text, ha='center', va='bottom', color='black', fontsize=20, fontweight='bold')
        
        ax.set_xticklabels(['Longitudinal\n(Along Z)', 'Transverse\n(Along Arc)'], fontsize=20)
        ax.set_ylabel("Normalized Magnitude" if is_norm else "Absolute Magnitude (cm)", fontsize=20)
        if is_norm: ax.set_ylim(0, 1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # --- Reference Lines (Updated to Geodesic) ---
        init_geodesic_resources()
        L_MOD = _GEODESIC_GLOBALS['L_MAX']
        C_MOD = _GEODESIC_GLOBALS['C_MAX']
        
        # Physical Parameters
        A_PHYS, B_PHYS = 1.2602, 1.3749
        L_PHYS = get_geodesic_total_length(A_PHYS, B_PHYS)
        C_PHYS = 2 * np.pi * (B_PHYS**2 / (2 * A_PHYS))
        
        # Lobe-to-Lobe (Physical)
        u_peak = A_PHYS**2 - (B_PHYS**4) / (4 * A_PHYS**2)
        z_peak = np.sqrt(max(0, u_peak))
        dist_lobe_phys = get_geodesic_profile_distance(-z_peak, z_peak, A_PHYS, B_PHYS)
        
        # Lobe-to-Lobe (Model)
        a_mod, b_mod = 1.0, 1.1
        u_peak_mod = a_mod**2 - (b_mod**4) / (4 * a_mod**2)
        z_peak_mod = np.sqrt(max(0, u_peak_mod))
        dist_lobe_mod = get_geodesic_profile_distance(-z_peak_mod, z_peak_mod, a_mod, b_mod)
        
        PHYS_SCALE = L_PHYS / L_MOD # Factor to scale model geodesic to physical geodesic
        
        refs = [
            {'val': L_PHYS if not is_norm else 1.0, 'label': 'Tip-to-Tip', 'type': 'long', 'color': 'blue'},
            {'val': dist_lobe_phys if not is_norm else (dist_lobe_mod / L_MOD), 'label': 'Lobe-to-Lobe', 'type': 'long', 'color': 'cyan'},
            {'val': (C_PHYS / 2) if not is_norm else (C_MOD / (2 * C_MOD)), 'label': 'Face-to-Face', 'type': 'trans', 'color': 'red'}
        ]
        
        # Add scale factor to data if Absolute
        if not is_norm:
            d_long = [d * PHYS_SCALE for d in d_long]
            d_trans = [d * PHYS_SCALE for d in d_trans]
            # Redefine data for plotting
            data = [d_long, d_trans]
            
        for ref in refs:
            val = ref['val']
            x_pos = 1 if ref['type'] == 'long' else 2
            ax.axhline(val, color=ref['color'], linestyle='--', alpha=0.6, linewidth=2)
            # Adjust label position slightly to avoid overlap
            y_off = 0.01 if ref['type'] == 'long' else 0.01
            ax.text(x_pos - 0.45, val + y_off, ref['label'], color=ref['color'], fontsize=14, fontweight='bold', va='bottom')
        
        return p

    # --- Execution Logic ---
    plt.rcParams.update({'font.size': 20})

    if normalize_amplitude == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Calculate raw data
        long_raw = []; trans_raw = []
        for chain in chains_to_process:
            v_list = get_vectors_geodesic(chain, normalize=False)
            for v in v_list:
                if v['is_long']: long_raw.append(v['mag'])
                else: trans_raw.append(v['mag'])
        
        # Calculate norm data
        long_norm = []; trans_norm = []
        for chain in chains_to_process:
            v_list = get_vectors_geodesic(chain, normalize=True)
            for v in v_list:
                if v['is_long']: long_norm.append(v['mag'])
                else: trans_norm.append(v['mag'])
        
        p_raw = draw_standard_plot(ax1, long_raw, trans_raw, False)
        p_norm = draw_standard_plot(ax2, long_norm, trans_norm, True)
        ax1.set_title(f"Absolute (p={p_raw:.1e})", fontsize=20)
        ax2.set_title(f"Normalized (p={p_norm:.1e})", fontsize=20)
        p_val = p_norm
        l_final, t_final = long_norm, trans_norm
        
    else:
        fig, ax = plt.subplots(figsize=(6, 8))
        is_norm = (normalize_amplitude == True)
        
        l_dat = []; t_dat = []
        for chain in chains_to_process:
            v_list = get_vectors_geodesic(chain, normalize=is_norm)
            for v in v_list:
                if v['is_long']: l_dat.append(v['mag'])
                else: t_dat.append(v['mag'])
        
        p_val = draw_standard_plot(ax, l_dat, t_dat, is_norm)
        t_type = "Normalized" if is_norm else "Absolute"
        ax.set_title(f"{t_type} Step Size", fontsize=20)
        l_final, t_final = l_dat, t_dat

    # Calculate and Print Statistics
    if len(l_final) > 0 and len(t_final) > 0:
        print("\n--- Step Size Statistics ---")
        print(f"{'Category':<15} | {'N':<5} | {'Mean':<8} | {'Median':<8} | {'Std':<8}")
        print("-" * 55)
        for name, d in [("Longitudinal", l_final), ("Transverse", t_final)]:
            m, med, s = np.mean(d), np.median(d), np.std(d)
            print(f"{name:<15} | {len(d):<5} | {m:<8.3f} | {med:<8.3f} | {s:<8.3f}")
        print("-" * 55)

    plt.tight_layout()
    if output_file: plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)
    print("Final p-value: ", p_val)

    if return_stats:
        return pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Median', 'Std'],
            'Longitudinal': [len(l_final), np.mean(l_final), np.median(l_final), np.std(l_final)],
            'Transverse': [len(t_final), np.mean(t_final), np.median(t_final), np.std(t_final)],
        })













def generate_folded_half_chamber_map_with_sequence(sites_path, output_file=None, teeth_mode='bottom', show_quadrants=True, show_axes=True, plot=True, show_full_context=True, center_on_first=False):
    """
    Generates a map where all surfaces are folded onto the Bottom-Left Half-Chamber.
    Includes ARROWS cprocess_all_peanutsonnecting sequential sites to visualize movement.
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    
    # --- Geometric Constants (True Physical Dimensions) ---
    a_true, b_true = 1.2602, 1.3749
    z_max_true = np.sqrt(a_true**2 + b_true**2)
    
    # Original Collection Dimensions for scaling
    a_orig, b_orig = 1.0, 1.07
    z_max_orig = np.sqrt(a_orig**2 + b_orig**2)
    scale_factor = z_max_true / z_max_orig
    
    a, b = a_true, b_true
    z_max = z_max_true
    
    def peanut_radius_local(z):
        # r(z) for the true model
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    # --- Load and Sort Sites ---
    all_sites = []
    
    # Load Bottom
    if teeth_mode in ['both', 'bottom']:
        try:
            with open(sites_path, 'r') as f:
                d = json.load(f)
                sites = [s for s in d.get('sites', []) if len(s) >= 4] # Need frame info
                for s in sites:
                    all_sites.append({
                        'p': np.array(s[:3], dtype=float),
                        'frame': int(s[3]),
                        'type': 'bottom'
                    })
        except: pass

    # Load Front
    if teeth_mode in ['both', 'front']:
        try:
            front_path = os.path.join(sites_path.replace("_sites.json", "_front_sites.json"))
            if os.path.exists(front_path):
                with open(front_path, 'r') as f:
                    d = json.load(f)
                    sites = [s for s in d.get('sites', []) if len(s) >= 4]
                    for s in sites:
                        all_sites.append({
                            'p': np.array(s[:3], dtype=float),
                            'frame': int(s[3]),
                            'type': 'top'
                        })
        except: pass
        
    # Sort by time (frame)
    all_sites.sort(key=lambda x: x['frame'])
    
    if not all_sites:
        print("No sites found.")
        return

    # --- Pre-Process Points for Map ---
    points_data = []
    for i, s in enumerate(all_sites):
        p = s['p'] * scale_factor  # Scale to true dimensions
        
        z_raw = p[2]
        x_raw = p[0]
        y_raw = p[1]
        
        # Folding Logic (Standard V16)
        z_fold = -abs(z_raw) 
        x_fold = -abs(x_raw)
        y_fold = y_raw 
        theta_fold = np.arctan2(y_fold, x_fold)
        
        # Map Coordinates
        z = z_fold
        y_map = z # Linear Z mapping
        
        theta_ref_val = np.pi
        if center_on_first:
            if i == 0: theta_ref_seq = theta_fold
            theta_ref_val = theta_ref_seq
        
        diff = theta_fold - theta_ref_val
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        # Mirroring Logic
        if center_on_first and abs(diff) > np.pi / 2:
            diff = np.sign(diff) * (np.pi - abs(diff))

        r = peanut_radius_local(z)
        x_map = r * diff
        
        s['map_x'] = x_map
        s['map_y'] = y_map
        s['index'] = i + 1

    # --- Compute Vectors ---
    vectors = []
    for i in range(len(all_sites) - 1):
        s1 = all_sites[i]
        s2 = all_sites[i+1]
        
        p1 = s1['p']
        p2 = s2['p']
        
        z1, z2 = p1[2], p2[2]
        
        # dZ
        dz = z2 - z1
        
        # dArc
        avg_z = (z1 + z2) / 2
        r_avg = peanut_radius_local(avg_z)
        
        ang1 = np.arctan2(p1[1], p1[0])
        ang2 = np.arctan2(p2[1], p2[0])
        
        delta_ang = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
        darc = r_avg * delta_ang
        
        # Axis-Based Component Mirroring
        # Mirror Z-component based on movement towards/away from Z=0 (equator)
        z1_abs = abs(z1)
        z2_abs = abs(z2)
        moving_towards_equator = z2_abs < z1_abs  # Moving towards Z=0
        
        # On map, Y represents Z. Check if dz currently points towards Y=0
        y_start = s1['map_y']
        y_end = y_start + dz
        dz_points_towards_y0 = abs(y_end) < abs(y_start)
        
        # If mismatch, flip dZ
        if moving_towards_equator != dz_points_towards_y0:
            dz = -dz
        
        
        # Quadrant-Based Arc Mirroring
        # If the starting point was on the Right hemisphere (x_raw > 0),
        # the folding (x_fold = -abs(x_raw)) flips it to Left.
        # The arc direction must be negated to preserve correct orientation.
        x1_raw = p1[0]
        
        if x1_raw > 0:
            # Point was on Right hemisphere, now folded to Left
            # Flip the arc direction
            darc = -darc
        
        # Classification
        is_long = abs(dz) >= abs(darc)
        color = '#1f77b4' if is_long else '#d62728' # Blue=Long, Red=Transverse
        
        vectors.append({
            'start_x': s1['map_x'],
            'start_y': s1['map_y'],
            'dx': darc,  # Physical dArc (axis-mirrored)
            'dy': dz,    # Physical dZ (axis-mirrored)
            'color': color
        })

    # --- Plotting ---
    plt.rcParams.update({'font.size': 18})
    
    fig_w, fig_h = 10, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Reference Outline
    z_refs = np.linspace(0, -z_max, 200) 
    r_refs = peanut_radius_local(z_refs)
    width_vals = r_refs * (np.pi / 2)
    
    outline_x_pos = width_vals
    outline_x_neg = -width_vals
    outline_y = z_refs
    
    # --- Add Shaded Dashed Lobes (Back/Side Extensions) ---
    # These represent the 'back' of the chamber (angles pi/2 to pi)
    outer_width_vals = r_refs * np.pi
    
    if show_full_context:
        # Left Lobe (dashed)
        ax.fill_betweenx(outline_y, -outer_width_vals, outline_x_neg, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(-outer_width_vals, outline_y, color='k', linestyle='--', linewidth=1.5)

        # Right Lobe (dashed)
        ax.fill_betweenx(outline_y, outline_x_pos, outer_width_vals, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(outer_width_vals, outline_y, color='k', linestyle='--', linewidth=1.5)

    # --- Add Top Half Shaded Area (z > 0) for Context ---
    if show_full_context:
        z_top = np.linspace(0, z_max, 200)
        r_top = peanut_radius_local(z_top)
        width_top = r_top * (np.pi / 2)
        outer_width_top = r_top * np.pi
        
        # Top Bowl (center) - Context
        ax.fill_betweenx(z_top, width_top, -width_top, color='#a6806d', alpha=0.3)
        ax.plot(width_top, z_top, 'k-', linewidth=2)
        ax.plot(-width_top, z_top, 'k-', linewidth=2)
        # Close top edge
        ax.plot([width_top[-1], -width_top[-1]], [z_top[-1], z_top[-1]], 'k-', linewidth=2)
        
        # Top Side Lobes
        ax.fill_betweenx(z_top, -outer_width_top, -width_top, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(-outer_width_top, z_top, color='k', linestyle='--', linewidth=1.5)
        
        ax.fill_betweenx(z_top, width_top, outer_width_top, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(outer_width_top, z_top, color='k', linestyle='--', linewidth=1.5)

    # Clean Orientation: No Inversion (Solid Bowl)
    ax.fill_betweenx(outline_y, outline_x_pos, outline_x_neg, color='#a6806d', alpha=0.5, label='Peanut Surface')
    ax.plot(outline_x_pos, outline_y, 'k-', linewidth=2)
    ax.plot(outline_x_neg, outline_y, 'k-', linewidth=2)
    ax.plot([outline_x_pos[0], outline_x_neg[0]], [outline_y[0], outline_y[0]], 'k-', linewidth=2)
    ax.plot([outline_x_pos[-1], outline_x_neg[-1]], [outline_y[-1], outline_y[-1]], 'k-', linewidth=2)

    if show_axes:
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Draw Vectors
    for v in vectors:
        ax.arrow(v['start_x'], v['start_y'], v['dx'], v['dy'], 
                 head_width=0.08, head_length=0.1, 
                 fc=v['color'], ec=v['color'], 
                 length_includes_head=True, linewidth=2, zorder=15)

    # Scatter Sites
    x_vals = [s['map_x'] for s in all_sites]
    y_vals = [s['map_y'] for s in all_sites]
    colors = ['#1f77b4' if s['type']=='bottom' else '#d62728' for s in all_sites]
    indices = [s['index'] for s in all_sites]
    
    marker_size = 300
    ax.scatter(x_vals, y_vals, c=colors, alpha=0.9, s=marker_size, edgecolors='w', linewidth=0.5, zorder=20)

    for x, y, idx in zip(x_vals, y_vals, indices):
        ax.text(x, y, str(idx), fontsize=10, ha='center', va='center', color='white', fontweight='bold', zorder=25)
    
    if show_axes:
        ax.set_xlabel("Arc Length (Model Units)")
        ax.set_ylabel("Linear Z (Model Units)")
    else:
        ax.axis('off')
    
    
    # Calculate arrow extents to ensure all arrows are visible
    arrow_end_x = [v['start_x'] + v['dx'] for v in vectors]
    arrow_end_y = [v['start_y'] + v['dy'] for v in vectors]
    
    # Include peanut outline and outer lobes (both top and bottom) in bounds calculation
    all_x = x_vals + [v['start_x'] for v in vectors] + arrow_end_x + list(outline_x_pos) + list(outline_x_neg)
    all_y = y_vals + [v['start_y'] for v in vectors] + arrow_end_y + list(outline_y)
    
    if show_full_context:
        # Add Side Lobes and Top Half
        all_x += list(outer_width_vals) + list(-outer_width_vals)
        all_x += list(outer_width_top) + list(-outer_width_top)
        all_y += list(z_top)
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Set equal aspect ratio to match plot_sequence_for_peanut_two_faces
    ax.set_aspect('equal')
    
    if show_axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
        print(f"Saved Folded Sequence Map to {output_file} (and SVG)")
    elif plot == True:
        plt.show()
    plt.close(fig)
    
    # Return dictionary with sites and arrow coordinates
    return {
        'sites': [
            {
                'map_x': s['map_x'],
                'map_y': s['map_y'],
                'p': s['p'].tolist(),  # Original 3D coordinates
                'index': s['index'],
                'type': s['type']
            }
            for s in all_sites
        ],
        'arrows': [
            {
                'start_x': v['start_x'],
                'start_y': v['start_y'],
                'dx': v['dx'],
                'dy': v['dy'],
                'color': v['color']
            }
            for v in vectors
        ]
    }








def plot_folded_half_chamber_map_from_data(data, output_file=None, show_quadrants=True, show_axes=True, show_full_context=True):
    """
    Plot folded half-chamber map from data dictionary.
    Takes the dictionary output from generate_folded_half_chamber_map_with_sequence
    and produces the exact same plot.
    
    Args:
        data: Dictionary with 'sites' and 'arrows' keys (from generate_folded_half_chamber_map_with_sequence)
        output_file: Path to save figure
        show_quadrants: Show quadrant labels
        show_axes: Show axis labels
    """
    # Peanut parameters (True physical dimensions)
    a, b = 1.2602, 1.3749
    def peanut_radius_local(z, a=a, b=b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))
    
    z_max = np.sqrt(a**2 + b**2)
    
    # Extract data
    sites = data['sites']
    arrows = data['arrows']
    
    # --- Plotting (EXACT SAME AS ORIGINAL FUNCTION) ---
    plt.rcParams.update({'font.size': 18})
    
    fig_w, fig_h = 10, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Reference Outline
    z_refs = np.linspace(0, -z_max, 200)
    r_refs = peanut_radius_local(z_refs)
    width_vals = r_refs * (np.pi / 2)
    
    outline_x_pos = width_vals
    outline_x_neg = -width_vals
    outline_y = z_refs
    
    # --- Add Shaded Dashed Lobes (Back/Side Extensions) ---
    outer_width_vals = r_refs * np.pi
    
    if show_full_context:
        # Left Lobe (dashed)
        ax.fill_betweenx(outline_y, -outer_width_vals, outline_x_neg, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(-outer_width_vals, outline_y, color='k', linestyle='--', linewidth=1.5)

        # Right Lobe (dashed)
        ax.fill_betweenx(outline_y, outline_x_pos, outer_width_vals, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(outer_width_vals, outline_y, color='k', linestyle='--', linewidth=1.5)
    
    # --- Add Top Half Shaded Area (z > 0) for Context ---
    if show_full_context:
        z_top = np.linspace(0, z_max, 200)
        r_top = peanut_radius_local(z_top)
        width_top = r_top * (np.pi / 2)
        outer_width_top = r_top * np.pi
        
        # Top Bowl (center) - Context
        ax.fill_betweenx(z_top, width_top, -width_top, color='#a6806d', alpha=0.3)
        ax.plot(width_top, z_top, 'k-', linewidth=2)
        ax.plot(-width_top, z_top, 'k-', linewidth=2)
        # Close top edge
        ax.plot([width_top[-1], -width_top[-1]], [z_top[-1], z_top[-1]], 'k-', linewidth=2)
        
        # Top Side Lobes
        ax.fill_betweenx(z_top, -outer_width_top, -width_top, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(-outer_width_top, z_top, color='k', linestyle='--', linewidth=1.5)
        
        ax.fill_betweenx(z_top, width_top, outer_width_top, color='#a6806d', alpha=0.15, edgecolor='none')
        ax.plot(outer_width_top, z_top, color='k', linestyle='--', linewidth=1.5)

    # Solid Bowl
    ax.fill_betweenx(outline_y, outline_x_pos, outline_x_neg, color='#a6806d', alpha=0.5, label='Peanut Surface')
    ax.plot(outline_x_pos, outline_y, 'k-', linewidth=2)
    ax.plot(outline_x_neg, outline_y, 'k-', linewidth=2)
    ax.plot([outline_x_pos[0], outline_x_neg[0]], [outline_y[0], outline_y[0]], 'k-', linewidth=2)
    ax.plot([outline_x_pos[-1], outline_x_neg[-1]], [outline_y[-1], outline_y[-1]], 'k-', linewidth=2)
    
    if show_axes:
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Draw Vectors
    for v in arrows:
        ax.arrow(v['start_x'], v['start_y'], v['dx'], v['dy'],
                 head_width=0.08, head_length=0.1,
                 fc=v['color'], ec=v['color'],
                 length_includes_head=True, linewidth=2, zorder=15, alpha = 0.75)
    
    # Scatter Sites
    x_vals = [s['map_x'] for s in sites]
    y_vals = [s['map_y'] for s in sites]
    colors = ['#1f77b4' if s['type']=='bottom' else '#d62728' for s in sites]
    indices = [s['index'] for s in sites]
    
    marker_size = 70
    ax.scatter(x_vals, y_vals, c='black', alpha=0.9, s=marker_size, edgecolors='w', linewidth=0.5, zorder=10)
    
    if show_axes:
        ax.set_xlabel("Arc Length (Model Units)")
        ax.set_ylabel("Linear Z (Model Units)")
    else:
        ax.axis('off')
    
    # Calculate bounds
    arrow_end_x = [v['start_x'] + v['dx'] for v in arrows]
    arrow_end_y = [v['start_y'] + v['dy'] for v in arrows]
    
    all_x = x_vals + [v['start_x'] for v in arrows] + arrow_end_x + list(outline_x_pos) + list(outline_x_neg)
    all_y = y_vals + [v['start_y'] for v in arrows] + arrow_end_y + list(outline_y)

    if show_full_context:
        all_x.extend(list(outer_width_vals) + list(-outer_width_vals))
        all_x.extend(list(outer_width_top) + list(-outer_width_top))
        all_y.extend(list(z_top))
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    if show_axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
        print(f"Saved Folded Sequence Map to {output_file} (and SVG)")
    plt.show()
    plt.close(fig)


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

def plot_density_folded_half_chamber_map_from_data(data, output_file=None, show_quadrants=True, show_axes=True, show_full_context=True, show_marginal=False):
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    # Peanut parameters (True physical dimensions)
    a, b = 1.2602, 1.3749
    def peanut_radius_local(z, a=a, b=b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))
    
    z_max = np.sqrt(a**2 + b**2)
    
    # Extract data
    sites = data['sites']
    
    # --- Plotting ---
    plt.rcParams.update({'font.size': 18})
    
    if show_marginal:
        import matplotlib.gridspec as gridspec
        from scipy.stats import gaussian_kde
        fig = plt.figure(figsize=(13, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1.2], wspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1], sharey=ax)
    else:
        fig_w, fig_h = 10, 8
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Reference Outline
    z_refs = np.linspace(0, -z_max, 200)
    r_refs = peanut_radius_local(z_refs)
    width_vals = r_refs * (np.pi / 2)
    
    outline_x_pos = width_vals
    outline_x_neg = -width_vals
    outline_y = z_refs
    
    # --- Add Shaded Dashed Lobes (Back/Side Extensions) ---
    outer_width_vals = r_refs * np.pi
    
    if show_full_context:
        # Left Lobe (dashed)
        ax.plot(-outer_width_vals, outline_y, color='k', linestyle='--', linewidth=1.5)
        # Right Lobe (dashed)
        ax.plot(outer_width_vals, outline_y, color='k', linestyle='--', linewidth=1.5)
    
    # --- Add Top Half Shaded Area (z > 0) for Context ---
    if show_full_context:
        z_top = np.linspace(0, z_max, 200)
        r_top = peanut_radius_local(z_top)
        width_top = r_top * (np.pi / 2)
        outer_width_top = r_top * np.pi
        
        # Top Bowl (center) - Context
        ax.plot(width_top, z_top, 'k-', linewidth=2)
        ax.plot(-width_top, z_top, 'k-', linewidth=2)
        ax.plot([width_top[-1], -width_top[-1]], [z_top[-1], z_top[-1]], 'k-', linewidth=2)
        
        # Top Side Lobes
        ax.plot(-outer_width_top, z_top, color='k', linestyle='--', linewidth=1.5)
        ax.plot(outer_width_top, z_top, color='k', linestyle='--', linewidth=1.5)

    # Solid Bowl Outline
    ax.plot(outline_x_pos, outline_y, 'k-', linewidth=2)
    ax.plot(outline_x_neg, outline_y, 'k-', linewidth=2)
    ax.plot([outline_x_pos[0], outline_x_neg[0]], [outline_y[0], outline_y[0]], 'k-', linewidth=2)
    ax.plot([outline_x_pos[-1], outline_x_neg[-1]], [outline_y[-1], outline_y[-1]], 'k-', linewidth=2)
    
    if show_axes:
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # --- DENSITY PLOT ---
    x_vals = [s['map_x'] for s in sites]
    y_vals = [s['map_y'] for s in sites]
    
    # Draw heatmap
    # We use seaborn kdeplot which can fill contours. 
    # clip=( (xmin, xmax), (ymin, ymax) ) to keep it mostly inside the peanut
    # We will just let it bleed slightly and standard outline will be on top.
    
    sns.kdeplot(
        x=x_vals, y=y_vals,
        ax=ax,
        fill=True,
        cmap="magma_r",  # Or 'viridis', 'hot_r', 'mako_r'
        levels=20,
        alpha=0.8,
        thresh=0.05,
        zorder=5 # Below outline but above grid
    )
    
    # Re-draw outline over the heatmap to clip it visually
    ax.plot(outline_x_pos, outline_y, 'k-', linewidth=2, zorder=10)
    ax.plot(outline_x_neg, outline_y, 'k-', linewidth=2, zorder=10)
    ax.plot([outline_x_pos[0], outline_x_neg[0]], [outline_y[0], outline_y[0]], 'k-', linewidth=2, zorder=10)
    ax.plot([outline_x_pos[-1], outline_x_neg[-1]], [outline_y[-1], outline_y[-1]], 'k-', linewidth=2, zorder=10)
    
    if show_axes:
        ax.set_xlabel("Arc Length (Model Units)")
        ax.set_ylabel("Linear Z (Model Units)")
    else:
        ax.axis('off')
    
    # Calculate bounds
    all_x = list(outline_x_pos) + list(outline_x_neg)
    all_y = list(outline_y)

    if show_full_context:
        all_x.extend(list(outer_width_vals) + list(-outer_width_vals))
        all_x.extend(list(outer_width_top) + list(-outer_width_top))
        all_y.extend(list(z_top))
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    if show_axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
    
    if show_marginal:
        ax_hist.hist(y_vals, bins=15, color='white', alpha=0.5, edgecolor='white', 
                     orientation='horizontal', density=True, zorder=3)
        if len(y_vals) > 1:
            try:
                kde = gaussian_kde(y_vals)
                y_grid = np.linspace(np.min(y_vals), np.max(y_vals), 200)
                ax_hist.plot(kde(y_grid), y_grid, color='#112244', linewidth=2.5, zorder=4)
            except Exception:
                pass
        
        ax_hist.set_xlabel("Density", fontsize=18)
        plt.setp(ax_hist.get_yticklabels(), visible=False)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['left'].set_visible(False)
        ax_hist.spines['bottom'].set_linewidth(2)
        ax_hist.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax_hist.tick_params(axis='y', which='both', left=False)

    plt.tight_layout()
    if output_file:
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
        print(f"Saved Density Map to {output_file} (and SVG)")
    plt.show()
    plt.close(fig)

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




    
        






















    













































def visualize_simulated_uncorrelated_vectors(sites_dir, mode='population', teeth_mode='bottom', specific_peanut=None, output_file=None, return_stats=False, absolute_val=True, sims_per_sequence=50):
    """
    Simulates an Uncorrelated Random Walk on the 3D true peanut model,
    matching the exact sequence lengths found in the empirical JSON sites.
    Generates a Rose plot identical in style to the empirical vector analysis.
    """
    import os, json
    import numpy as np
    import matplotlib.pyplot as plt
    import from_antigravity_peanuts as fp

    chains_to_process = []
    files = [f for f in os.listdir(sites_dir) if f.endswith("_sites.json") and "_front" not in f]
    
    if mode != 'population' and specific_peanut is None:
        specific_peanut = mode
        
    if specific_peanut:
        files = [f for f in files if str(specific_peanut) in f]

    for f_name in files:
        if teeth_mode in ['both', 'bottom']:
            path = os.path.join(sites_dir, f_name)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        d = json.load(f)
                    raw = d.get('sites', []) if isinstance(d, dict) else d
                    pts = [{'p': [float(s[0]), float(s[1]), float(s[2])]} for s in raw if len(s) >= 3]
                    if len(pts) >= 2: chains_to_process.append(pts)
                except: pass
        if teeth_mode in ['both', 'front', 'top']:
            front_path = os.path.join(sites_dir, f_name.replace("_sites.json", "_front_sites.json"))
            if os.path.exists(front_path):
                try:
                    with open(front_path, 'r') as f:
                        d = json.load(f)
                    raw = d.get('sites', []) if isinstance(d, dict) else d
                    pts = [{'p': [float(s[0]), float(s[1]), float(s[2])]} for s in raw if len(s) >= 3]
                    if len(pts) >= 2: chains_to_process.append(pts)
                except: pass

    chain_lengths = [len(chain) for chain in chains_to_process]
    
    _a_p, _b_p = 1.2602, 1.3749
    _z_max_p = np.sqrt(_a_p**2 + _b_p**2)

    def _apply_z_move(z, th, dz, z_max):
        limit = z_max * 0.999
        next_z = z + dz
        if next_z > limit:
            return limit - (next_z - limit), (th + np.pi) % (2 * np.pi)
        elif next_z < -limit:
            return -limit + (-limit - next_z), (th + np.pi) % (2 * np.pi)
        return next_z, th

    def peanut_radius(z, a, b):
        t1 = np.sqrt(b**4 + 4*a**2*z**2)
        r2 = t1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    def _peanut_surface_point(z, th, a, b):
        r = peanut_radius(z, a, b)
        return np.array([r*np.cos(th), r*np.sin(th), z])

    z_lin = np.linspace(-_z_max_p * 0.999, _z_max_p * 0.999, 100)
    th_lin = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    Zm, Thm = np.meshgrid(z_lin, th_lin, indexing='ij')
    Rm = peanut_radius(Zm, _a_p, _b_p)
    Xm = Rm * np.cos(Thm)
    Ym = Rm * np.sin(Thm)

    faces = []
    face_areas = []
    rows, cols = Xm.shape
    for r in range(rows - 1):
        for c in range(cols):
            p1 = np.array([Xm[r, c], Ym[r, c], Zm[r, c]])
            p2 = np.array([Xm[r+1, c], Ym[r+1, c], Zm[r+1, c]])
            nxt_c = (c + 1) % cols
            p3 = np.array([Xm[r+1, nxt_c], Ym[r+1, nxt_c], Zm[r+1, nxt_c]])
            p4 = np.array([Xm[r, nxt_c], Ym[r, nxt_c], Zm[r, nxt_c]])
            
            c1 = np.linalg.norm(np.cross(p2-p1, p3-p1)) / 2.0
            c2 = np.linalg.norm(np.cross(p3-p1, p4-p1)) / 2.0
            faces.append((p1 + p2 + p3 + p4) / 4.0)
            face_areas.append(c1 + c2)
            
    centroids = np.array(faces)
    face_probs = np.array(face_areas) / np.sum(face_areas)

    simulated_chains = []
    for n_steps in chain_lengths:
        for _ in range(sims_per_sequence):
            idx = np.random.choice(len(centroids), p=face_probs)
            start_pt = centroids[idx]
            z0 = start_pt[2]
            th0 = np.arctan2(start_pt[1], start_pt[0])
            
            chain = [{'p': _peanut_surface_point(z0, th0, _a_p, _b_p)}]
            cur_z, cur_th = z0, th0
            
            for _ in range(1, n_steps):
                step_len = np.random.exponential(scale=1.0)
                direction = np.random.uniform(0, 2*np.pi)
                cur_z, cur_th = _apply_z_move(cur_z, cur_th, step_len * np.cos(direction), _z_max_p)
                r_cur = max(1e-6, peanut_radius(cur_z, _a_p, _b_p))
                cur_th = (cur_th + step_len * np.sin(direction)/r_cur) % (2*np.pi)
                
                chain.append({'p': _peanut_surface_point(cur_z, cur_th, _a_p, _b_p)})
            simulated_chains.append(chain)

    all_dz, all_darc = [], []
    for chain in simulated_chains:
        for i in range(len(chain) - 1):
            p1 = chain[i]['p']
            p2 = chain[i+1]['p']
            dz = fp.get_geodesic_profile_distance(p1[2], p2[2], a=_a_p, b=_b_p)
            avg_z = (p1[2] + p2[2]) / 2
            r = peanut_radius(avg_z, a=_a_p, b=_b_p)
            ang1 = np.arctan2(p1[1], p1[0])
            ang2 = np.arctan2(p2[1], p2[0])
            delta_ang = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
            darc = r * delta_ang
            all_dz.append(dz)
            all_darc.append(darc)

    if absolute_val:
        all_dz = np.abs(all_dz)
        all_darc = np.abs(all_darc)
        
    angles = np.arctan2(all_dz, all_darc)

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    bins = 10 if absolute_val else 20
    weights = np.ones_like(angles) / sims_per_sequence
    counts, bin_edges, patches = ax.hist(angles, bins=bins, 
                                         range=(0, np.pi/2) if absolute_val else (-np.pi, np.pi),
                                         weights=weights,
                                         color='gray', alpha=1, edgecolor='black', linewidth=1)

    expected = counts.sum() / bins
    print(f"\n--- Simulated Null Distribution Stats ---")
    print(f"Total simulated vectors: {len(angles)}")
    from scipy import stats
    chi2_stat, p_val = stats.chisquare(counts, f_exp=expected)
    print(f"Chi-square test for uniformity: stat={chi2_stat:.3f}, p={p_val:.4g}")

    if p_val < 0.05:
        print("Result: Simulated distribution deviates significantly from mathematical uniform.")
        for i, count in enumerate(counts):
            std_resid = (count - expected) / np.sqrt(expected)
            if std_resid > 1.96:
                center = (bin_edges[i] + bin_edges[i+1]) / 2
                ax.text(center, count + max(counts)*0.05, "*", ha='center', va='center', fontsize=20, color='red', fontweight='bold')
            elif std_resid < -1.96:
                pass
    else:
        print("Result: Simulated distribution is not significantly different from mathematical uniform.")

    if absolute_val:
        for angle in [0, np.pi/2]:
            ax.axvline(angle, color='black', linewidth=2, zorder=5)

    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['+Arc', '+Z', '-Arc', '-Z'])
    ax.tick_params(axis='y', labelsize=15, colors='gray')
    ax.grid(True, linestyle=':', alpha=1)

    avg_dz = np.mean(all_dz)
    avg_darc = np.mean(all_darc)
    ratio = avg_dz / avg_darc if avg_darc > 0 else 0
    print(f"Mean Null Profile Distance (Z): {avg_dz:.2f} mm")
    print(f"Mean Null Arc Distance: {avg_darc:.2f} mm")
    print(f"Longitudinal / Transversal Ratio: {ratio:.2f}x")

    title_suffix = ""
    if teeth_mode == 'bottom': title_suffix = "\n(Bottom Teeth Only)"
    if teeth_mode == 'front': title_suffix = "\n(Front Teeth Only)"

    title_text = f"Simulated Null Vector Directions"
    if specific_peanut: title_text += f"\n({specific_peanut})"
    title_text += f"{title_suffix}"
    plt.title(title_text, y=1.08, fontsize=16)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    plt.show()

    if return_stats:
        return {'chi2': chi2_stat, 'p_val': p_val, 'ratio': ratio, 'counts': counts, 'expected': expected}
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




def plot_multi_strategy_1cm_hole_batched_ovoid(output_file, sims_per_batch=1000, num_batches=100, target_std=0.3, n_steps=5, plot_type='violin'):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import time
    
    print(f"Running ovoid baseline simulations with effective_radius = 0.5 + 0.3...")
    a_p, b_p = 1.2602, 1.3749
    c = np.sqrt(a_p**2 + b_p**2)
    z_bulk_p = np.sqrt((4*a_p**4 - b_p**4)/(4*a_p**2))
    a = np.sqrt(np.sqrt(b_p**4 + 4*a_p**2*z_bulk_p**2) - z_bulk_p**2 - a_p**2)
    _z_max_p = c

    _pLL, _pTT = 0.65, 0.26

    mag_L = np.random.normal(loc=0.967, scale=target_std, size=100000)
    mag_L = np.clip(mag_L, 0.01, None)

    mag_T = np.random.normal(loc=0.864, scale=target_std, size=100000)
    mag_T = np.clip(mag_T, 0.01, None)

    mag_pooled = np.random.normal(loc=0.9155, scale=target_std, size=100000)
    mag_pooled = np.clip(mag_pooled, 0.01, None)

    def _apply_z_move(z, th, dz, z_max):
        limit = z_max * 0.999
        next_z = z + dz
        if next_z > limit:
            return limit - (next_z - limit), (th + np.pi) % (2 * np.pi)
        elif next_z < -limit:
            return -limit + (-limit - next_z), (th + np.pi) % (2 * np.pi)
        return next_z, th

    def _ovoid_surface_point(z, th):
        r = a * np.sqrt(max(0.0, 1 - (z**2 / c**2)))
        return np.array([r*np.cos(th), r*np.sin(th), z])

    def _dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def peanut_radius_ovoid(z):
        return a * np.sqrt(max(0.0, 1 - (z**2 / c**2)))

    z_lin = np.linspace(-_z_max_p * 0.999, _z_max_p * 0.999, 150)
    th_lin = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    Zm, Thm = np.meshgrid(z_lin, th_lin, indexing='ij')

    Rm = a * np.sqrt(np.maximum(0, 1 - (Zm**2 / c**2)))
    Xm = Rm * np.cos(Thm)
    Ym = Rm * np.sin(Thm)

    faces = []
    face_areas = []
    rows, cols = Xm.shape
    for r_idx in range(rows - 1):
        for c_idx in range(cols):
            p1 = np.array([Xm[r_idx, c_idx], Ym[r_idx, c_idx], Zm[r_idx, c_idx]])
            p2 = np.array([Xm[r_idx+1, c_idx], Ym[r_idx+1, c_idx], Zm[r_idx+1, c_idx]])
            nxt_c = (c_idx + 1) % cols
            p3 = np.array([Xm[r_idx+1, nxt_c], Ym[r_idx+1, nxt_c], Zm[r_idx+1, nxt_c]])
            p4 = np.array([Xm[r_idx, nxt_c], Ym[r_idx, nxt_c], Zm[r_idx, nxt_c]])
            
            c1 = (np.linalg.norm(np.cross(p2-p1, p3-p1)) + np.linalg.norm(np.cross(p3-p1, p4-p1))) / 2.0
            faces.append((p1 + p2 + p3 + p4) / 4.0)
            face_areas.append(c1)

    centroids = np.array(faces)
    face_areas = np.array(face_areas)
    face_probs = face_areas / np.sum(face_areas)

    hole_center = _ovoid_surface_point(z_bulk_p, 0)
    
    strategies = ['Markov Final', 'Markov (50/50 Pooled)', 'Markov Inversed']
    
    D = 1.0 # 1cm hole
    hole_radius = D / 2.0
    effective_radius = hole_radius + 0.3 # HOLE RADIUS + 0.3
    
    batch_results = {k: [] for k in strategies}
    
    t0 = time.time()
    for batch in range(num_batches):
        counts = {k: 0 for k in strategies}
        
        for _ in range(sims_per_batch):
            idx = np.random.choice(len(centroids), p=face_probs)
            start_pt = centroids[idx]
            z0 = start_pt[2]
            th0 = np.arctan2(start_pt[1], start_pt[0])

            p0 = _ovoid_surface_point(z0, th0)
            p0_f = _ovoid_surface_point(z0, (th0 + np.pi)%(2*np.pi))

            # --- Markov Final ---
            touched = False
            if _dist(p0, hole_center) <= effective_radius or _dist(p0_f, hole_center) <= effective_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.68, 0.32])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_L)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < _pLL else 'T'
                    else:
                        darc = np.random.choice(mag_T) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_ovoid(cur_z))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < _pTT else 'L'

                    pt_b = _ovoid_surface_point(cur_z, cur_th)
                    pt_f = _ovoid_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi))
                    if _dist(pt_b, hole_center) <= effective_radius or _dist(pt_f, hole_center) <= effective_radius:
                        touched = True
                        break
            if touched: counts['Markov Final'] += 1

            # --- Markov (50/50 Pooled) ---
            touched = False
            if _dist(p0, hole_center) <= effective_radius or _dist(p0_f, hole_center) <= effective_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.5, 0.5])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_pooled)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < 0.5 else 'T'
                    else:
                        darc = np.random.choice(mag_pooled) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_ovoid(cur_z))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < 0.5 else 'L'

                    pt_b = _ovoid_surface_point(cur_z, cur_th)
                    pt_f = _ovoid_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi))
                    if _dist(pt_b, hole_center) <= effective_radius or _dist(pt_f, hole_center) <= effective_radius:
                        touched = True
                        break
            if touched: counts['Markov (50/50 Pooled)'] += 1

            # --- Markov Inversed ---
            touched = False
            if _dist(p0, hole_center) <= effective_radius or _dist(p0_f, hole_center) <= effective_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.32, 0.68])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_T)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < _pTT else 'T'
                    else:
                        darc = np.random.choice(mag_L) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_ovoid(cur_z))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < _pLL else 'L'

                    pt_b = _ovoid_surface_point(cur_z, cur_th)
                    pt_f = _ovoid_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi))
                    if _dist(pt_b, hole_center) <= effective_radius or _dist(pt_f, hole_center) <= effective_radius:
                        touched = True
                        break
            if touched: counts['Markov Inversed'] += 1

        for k in strategies:
            batch_results[k].append(counts[k] / sims_per_batch)
            
    print(f"Finished {num_batches} batches of {sims_per_batch} sims in {time.time() - t0:.1f} seconds")

    data = []
    for k in strategies:
        for val in batch_results[k]:
            data.append({'Strategy': k, 'Probability': val * 100})
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    if plot_type == 'violin':
        ax = sns.violinplot(x='Strategy', y='Probability', data=df, palette='muted')
    else:
        ax = sns.barplot(x='Strategy', y='Probability', data=df, capsize=.1, palette='muted')
        
    # --- Stats ---
    from scipy import stats
    def get_sig(p):
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return "ns"
        
    def draw_bracket(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='black')

    d1 = df[df['Strategy'] == 'Markov Final']['Probability']
    d2 = df[df['Strategy'] == 'Markov (50/50 Pooled)']['Probability']
    d3 = df[df['Strategy'] == 'Markov Inversed']['Probability']
    
    _, p12 = stats.mannwhitneyu(d1, d2)
    _, p23 = stats.mannwhitneyu(d2, d3)
    _, p13 = stats.mannwhitneyu(d1, d3)
    
    print(f"Stats (Markov Final vs Pooled): p={p12:.4e}")
    print(f"Stats (Pooled vs Inversed): p={p23:.4e}")
    print(f"Stats (Markov Final vs Inversed): p={p13:.4e}")
    
    ymax = df['Probability'].max()
    y_b = max(ymax + 2, 75)
    
    draw_bracket(ax, 0, 1, y_b, 1.5, get_sig(p12))
    draw_bracket(ax, 1, 2, y_b, 1.5, get_sig(p23))
    draw_bracket(ax, 0, 2, y_b + 7, 1.5, get_sig(p13))
    # -------------
    
    plt.ylim(0, max(100, y_b + 12))
    plt.title('Hole Finding Probability on Ovoid (effective_radius = hole_radius + 0.3)', fontsize=14)
    plt.ylabel('Success Probability (%)', fontsize=12)
    plt.xlabel('Strategy', fontsize=12)
    sns.despine()
    
    #os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #plt.savefig(output_file, bbox_inches='tight', dpi=300)
    #plt.savefig(output_file.replace('.svg', '.png'), bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Plot saved to {output_file}")



def plot_2d_transition_sweep_stars(csv_path, output_path):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    df = pd.read_csv(csv_path)
    sub_df = df[(df['magL'] == 0.92) & (df['magT'] == 0.92)]
    pivot_df = sub_df.pivot(index='pTT', columns='pLL', values='prob')
    pivot_df = pivot_df.sort_index(ascending=False)
    pivot_df.sort_index(axis=1, ascending=True, inplace=True)

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
    ax = sns.heatmap(pivot_df, cmap='magma', vmin=sub_df['prob'].min(), vmax=sub_df['prob'].max(), 
                     cbar_kws={'label': 'Probability of finding 1cm hole'})

    pLL_cols = list(pivot_df.columns)
    pTT_rows = list(pivot_df.index)

    try:
        x_emp = pLL_cols.index(0.65)
        y_emp = pTT_rows.index(0.26)
        ax.scatter([x_emp + 0.5], [y_emp + 0.5], color='cyan', marker='*', s=800, edgecolor='black', linewidth=1.5, zorder=10, label='Empirical Agouti')
    except ValueError:
        pass

    try:
        x_opt = pLL_cols.index(0.60)
        y_opt = pTT_rows.index(0.10)
        ax.scatter([x_opt + 0.5], [y_opt + 0.5], color='red', marker='*', s=800, edgecolor='black', linewidth=1.5, zorder=10, label='Global Optimal (4D)')
    except ValueError:
        pass

    ax.set_xlabel("Probability of staying Longitudinal (pLL)", fontsize=14, labelpad=10)
    ax.set_ylabel("Probability of staying Transverse (pTT)", fontsize=14, labelpad=10)
    ax.set_title("Ovoid Hole-Finding Probability (Magnitudes fixed at 0.92cm)", fontsize=16, pad=15)
    ax.set_xticklabels([f"{x:.2f}" for x in pLL_cols], rotation=45)
    ax.set_yticklabels([f"{y:.2f}" for y in pTT_rows], rotation=0)

    plt.legend(loc='upper right')
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #plt.savefig(output_path, bbox_inches='tight', dpi=300)
    #plt.savefig(output_path.replace('.svg', '.png'), bbox_inches='tight', dpi=300)
    plt.show()



def compute_and_plot_turning_angles(sites_directory, teeth_mode='bottom', output_file=None):
    """
    Computes the turning angle between successive vectors in the sequence of clamps
    for each trial and plots the distribution as a histogram.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    all_files = [os.path.join(sites_directory, el) for el in os.listdir(sites_directory)]
    valid_files = [f for f in all_files if 'front' not in os.path.basename(f).lower()]

    all_turning_angles = []

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
            if data and len(data['arrows']) > 1:
                arrows = data['arrows']
                for i in range(len(arrows) - 1):
                    dx1, dy1 = arrows[i]['dx'], arrows[i]['dy']
                    dx2, dy2 = arrows[i+1]['dx'], arrows[i+1]['dy']
                    
                    # Compute angle of each vector
                    theta1 = np.arctan2(dy1, dx1)
                    theta2 = np.arctan2(dy2, dx2)
                    
                    # Difference in radians
                    dtheta = theta2 - theta1
                    
                    # Normalize to [-pi, pi]
                    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                    
                    # Convert to degrees
                    dtheta_deg = np.degrees(dtheta)
                    all_turning_angles.append(dtheta_deg)
        except Exception:
            pass

    # Plot the distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram
    counts, bins, patches = ax.hist(all_turning_angles, bins=36, range=(-180, 180), color='#1f77b4', edgecolor='black', alpha=0.8)
    
    # Add a vertical line at 0 (straight ahead)
    ax.axvline(0, color='red', linestyle='--', linewidth=2.5, label='No turn (0\u00b0)')
    
    # Add vertical lines for +/- 90 (orthogonal turns)
    ax.axvline(90, color='orange', linestyle=':', linewidth=2, label='Orthogonal (\u00b190\u00b0)')
    ax.axvline(-90, color='orange', linestyle=':', linewidth=2)
    
    # Add vertical lines for +/- 180 (U-turns)
    ax.axvline(180, color='purple', linestyle='-.', linewidth=2, label='U-turn (\u00b1180\u00b0)')
    ax.axvline(-180, color='purple', linestyle='-.', linewidth=2)

    ax.set_xlabel('Turning Angle (Degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Turning Angles Between Successive Vectors (n={len(all_turning_angles)})')
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(loc='upper right')

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
        plt.savefig(output_file, dpi=300)
        print(f"Turning angles plot saved to {output_file}")
        
    plt.show()
    
    return all_turning_angles


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


def plot_static_3d_ovoid_hole_bounding_matched(output_svg=r'figures/ovoid_hole_bounding_matched.svg'):
    '''
    Generates a static 3D SVG of the Ovoid matching the True Peanut's bounding box.
    The surface is tan, the background is dark, and the vertices falling within the 
    1cm diameter hole are colored black.
    '''
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D

    # Peanut True Parameters
    a_p, b_p = 1.2602, 1.3749
    c = np.sqrt(a_p**2 + b_p**2)
    z_bulk = np.sqrt((4*a_p**4 - b_p**4)/(4*a_p**2))
    r_max_peanut = np.sqrt(np.sqrt(b_p**4 + 4*a_p**2*z_bulk**2) - z_bulk**2 - a_p**2)

    # Bounding-Box Matched Ovoid
    a = r_max_peanut

    u = np.linspace(0, np.pi, 200)
    v = np.linspace(0, 2 * np.pi, 200)
    U, V = np.meshgrid(u, v)

    X = a * np.sin(U) * np.cos(V)
    Y = a * np.sin(U) * np.sin(V)
    Z = c * np.cos(U)

    fig = plt.figure(figsize=(10, 10), facecolor='#0f0f1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f1a')

    base_color = mcolors.to_rgba('#c8a04a', alpha=0.6)
    black_color = mcolors.to_rgba('black', alpha=0.9)

    hole_radius = 0.5
    hole_center_z = z_bulk
    hole_center_x = a * np.sqrt(max(0, 1 - (hole_center_z**2 / c**2)))

    fc = np.zeros((X.shape[0], X.shape[1], 4))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist = np.sqrt((X[i,j] - hole_center_x)**2 + (Y[i,j] - 0)**2 + (Z[i,j] - hole_center_z)**2)
            if dist <= hole_radius:
                fc[i,j] = black_color
            else:
                fc[i,j] = base_color

    ax.plot_surface(X, Y, Z, facecolors=fc, edgecolor='black', linewidth=0.3, shade=False)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w', alpha=0)

    ax.set_axis_off()
    ax.set_title("Ovoid (Bounding-Box Matched) with 1cm Hole", fontsize=14, fontweight='bold', pad=20, color='white')
    ax.view_init(elev=20, azim=30)

    if output_svg:
        import os
        os.makedirs(os.path.dirname(output_svg), exist_ok=True)
        plt.savefig(output_svg, format='svg', bbox_inches='tight', facecolor='#0f0f1a')
        plt.savefig(output_svg.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight', facecolor='#0f0f1a')
        print(f"Saved SVG to {output_svg}")
    
    plt.show()
    plt.close(fig)


def plot_static_3d_inverted_ovoid_hole_bounding_matched(output_svg=r'figures/inverted_ovoid_hole_bounding_matched.svg'):
    '''
    Generates a static 3D SVG of the Inverted Ovoid matching the True Peanut's bounding box.
    The surface is tan, the background is dark, and the vertices falling within the 
    1cm diameter hole are colored black. Mesh is high-res (300x300).
    '''
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D

    # Peanut True Parameters
    a_p, b_p = 1.2602, 1.3749
    c_peanut = np.sqrt(a_p**2 + b_p**2)
    z_bulk_p = np.sqrt((4*a_p**4 - b_p**4)/(4*a_p**2))
    r_max_peanut = np.sqrt(np.sqrt(b_p**4 + 4*a_p**2*z_bulk_p**2) - z_bulk_p**2 - a_p**2)

    # Bounding-Box Matched Inverted Ovoid
    a = c_peanut
    c = r_max_peanut

    u = np.linspace(0, np.pi, 300)
    v = np.linspace(0, 2 * np.pi, 300)
    U, V = np.meshgrid(u, v)

    X = a * np.sin(U) * np.cos(V)
    Y = a * np.sin(U) * np.sin(V)
    Z = c * np.cos(U)

    fig = plt.figure(figsize=(10, 10), facecolor='#0f0f1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f1a')

    base_color = mcolors.to_rgba('#c8a04a', alpha=0.6)
    black_color = mcolors.to_rgba('black', alpha=0.9)

    ratio = z_bulk_p / c_peanut
    hole_center_z = ratio * c
    hole_center_x = a * np.sqrt(max(0, 1 - (hole_center_z**2 / c**2)))
    hole_radius = 0.5

    fc = np.zeros((X.shape[0], X.shape[1], 4))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist = np.sqrt((X[i,j] - hole_center_x)**2 + (Y[i,j] - 0)**2 + (Z[i,j] - hole_center_z)**2)
            if dist <= hole_radius:
                fc[i,j] = black_color
            else:
                fc[i,j] = base_color

    ax.plot_surface(X, Y, Z, facecolors=fc, edgecolor='black', linewidth=0.2, shade=False)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w', alpha=0)

    ax.set_axis_off()
    ax.set_title("Inverted Ovoid (Bounding-Box Matched)", fontsize=14, fontweight='bold', pad=20, color='white')
    ax.view_init(elev=20, azim=30)

    if output_svg:
        import os
        os.makedirs(os.path.dirname(output_svg), exist_ok=True)
        plt.savefig(output_svg, format='svg', bbox_inches='tight', facecolor='#0f0f1a')
        plt.savefig(output_svg.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight', facecolor='#0f0f1a')
        print(f"Saved SVG to {output_svg}")
    
    plt.show()
    plt.close(fig)


def plot_multi_strategy_1cm_hole_batched_inverted_ovoid(output_file, sims_per_batch=1000, num_batches=100, target_std=0.3, n_steps=5, plot_type='violin'):
    """
    Sweeps search strategies on Inverted Ovoid matching true peanut aspect ratios.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    import time
    
    print(f"Running inverted ovoid baseline simulations with effective_radius = 0.5 + 0.3...")
    a_p, b_p = 1.2602, 1.3749
    c_peanut = np.sqrt(a_p**2 + b_p**2)
    z_bulk_p = np.sqrt((4*a_p**4 - b_p**4)/(4*a_p**2))
    r_max_peanut = np.sqrt(np.sqrt(b_p**4 + 4*a_p**2*z_bulk_p**2) - z_bulk_p**2 - a_p**2)

    # Inverted Ovoid parameters
    a = c_peanut
    c = r_max_peanut
    _z_max_p = c

    _pLL, _pTT = 0.65, 0.26

    mag_L = np.random.normal(loc=0.967, scale=target_std, size=100000)
    mag_L = np.clip(mag_L, 0.01, None)

    mag_T = np.random.normal(loc=0.864, scale=target_std, size=100000)
    mag_T = np.clip(mag_T, 0.01, None)

    mag_pooled = np.random.normal(loc=0.9155, scale=target_std, size=100000)
    mag_pooled = np.clip(mag_pooled, 0.01, None)

    def _apply_z_move(z, th, dz, z_max):
        limit = z_max * 0.999
        next_z = z + dz
        if next_z > limit:
            return limit - (next_z - limit), (th + np.pi) % (2 * np.pi)
        elif next_z < -limit:
            return -limit + (-limit - next_z), (th + np.pi) % (2 * np.pi)
        return next_z, th

    def _inverted_ovoid_surface_point(z, th):
        r = a * np.sqrt(max(0.0, 1 - (z**2 / c**2)))
        return np.array([r*np.cos(th), r*np.sin(th), z])

    def _dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def peanut_radius_inverted_ovoid(z):
        return a * np.sqrt(max(0.0, 1 - (z**2 / c**2)))

    z_lin = np.linspace(-_z_max_p * 0.999, _z_max_p * 0.999, 150)
    th_lin = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    Zm, Thm = np.meshgrid(z_lin, th_lin, indexing='ij')

    Rm = a * np.sqrt(np.maximum(0, 1 - (Zm**2 / c**2)))
    Xm = Rm * np.cos(Thm)
    Ym = Rm * np.sin(Thm)

    faces = []
    face_areas = []
    rows, cols = Xm.shape
    for r_idx in range(rows - 1):
        for c_idx in range(cols):
            p1 = np.array([Xm[r_idx, c_idx], Ym[r_idx, c_idx], Zm[r_idx, c_idx]])
            p2 = np.array([Xm[r_idx+1, c_idx], Ym[r_idx+1, c_idx], Zm[r_idx+1, c_idx]])
            nxt_c = (c_idx + 1) % cols
            p3 = np.array([Xm[r_idx+1, nxt_c], Ym[r_idx+1, nxt_c], Zm[r_idx+1, nxt_c]])
            p4 = np.array([Xm[r_idx, nxt_c], Ym[r_idx, nxt_c], Zm[r_idx, nxt_c]])
            
            c1 = (np.linalg.norm(np.cross(p2-p1, p3-p1)) + np.linalg.norm(np.cross(p3-p1, p4-p1))) / 2.0
            faces.append((p1 + p2 + p3 + p4) / 4.0)
            face_areas.append(c1)

    centroids = np.array(faces)
    face_areas = np.array(face_areas)
    face_probs = face_areas / np.sum(face_areas)

    ratio = z_bulk_p / c_peanut
    hole_center_z = ratio * c
    hole_center_x = a * np.sqrt(max(0.0, 1 - (hole_center_z**2 / c**2)))
    hole_center = np.array([hole_center_x, 0.0, hole_center_z])
    
    strategies = ['Markov Final', 'Markov (50/50 Pooled)', 'Markov Inversed']
    
    D = 1.0 # 1cm hole
    hole_radius = D / 2.0
    effective_radius = hole_radius + 0.3 # HOLE RADIUS + 0.3
    
    batch_results = {k: [] for k in strategies}
    
    t0 = time.time()
    for batch in range(num_batches):
        counts = {k: 0 for k in strategies}
        
        for _ in range(sims_per_batch):
            idx = np.random.choice(len(centroids), p=face_probs)
            start_pt = centroids[idx]
            z0 = start_pt[2]
            th0 = np.arctan2(start_pt[1], start_pt[0])

            p0 = _inverted_ovoid_surface_point(z0, th0)
            p0_f = _inverted_ovoid_surface_point(z0, (th0 + np.pi)%(2*np.pi))

            # --- Markov Final ---
            touched = False
            if _dist(p0, hole_center) <= effective_radius or _dist(p0_f, hole_center) <= effective_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.68, 0.32])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_L)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < _pLL else 'T'
                    else:
                        darc = np.random.choice(mag_T) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_inverted_ovoid(cur_z))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < _pTT else 'L'

                    pt_b = _inverted_ovoid_surface_point(cur_z, cur_th)
                    pt_f = _inverted_ovoid_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi))
                    if _dist(pt_b, hole_center) <= effective_radius or _dist(pt_f, hole_center) <= effective_radius:
                        touched = True
                        break
            if touched: counts['Markov Final'] += 1

            # --- Markov (50/50 Pooled) ---
            touched = False
            if _dist(p0, hole_center) <= effective_radius or _dist(p0_f, hole_center) <= effective_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.5, 0.5])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_pooled)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < 0.5 else 'T'
                    else:
                        darc = np.random.choice(mag_pooled) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_inverted_ovoid(cur_z))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < 0.5 else 'L'

                    pt_b = _inverted_ovoid_surface_point(cur_z, cur_th)
                    pt_f = _inverted_ovoid_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi))
                    if _dist(pt_b, hole_center) <= effective_radius or _dist(pt_f, hole_center) <= effective_radius:
                        touched = True
                        break
            if touched: counts['Markov (50/50 Pooled)'] += 1

            # --- Markov Inversed ---
            touched = False
            if _dist(p0, hole_center) <= effective_radius or _dist(p0_f, hole_center) <= effective_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.32, 0.68])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_T)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < _pTT else 'T'
                    else:
                        darc = np.random.choice(mag_L) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_inverted_ovoid(cur_z))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < _pLL else 'L'

                    pt_b = _inverted_ovoid_surface_point(cur_z, cur_th)
                    pt_f = _inverted_ovoid_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi))
                    if _dist(pt_b, hole_center) <= effective_radius or _dist(pt_f, hole_center) <= effective_radius:
                        touched = True
                        break
            if touched: counts['Markov Inversed'] += 1

        for k in strategies:
            batch_results[k].append(counts[k] / sims_per_batch)
            
    print(f"Finished {num_batches} batches of {sims_per_batch} sims in {time.time() - t0:.1f} seconds")

    data = []
    for k in strategies:
        for val in batch_results[k]:
            data.append({'Strategy': k, 'Probability': val * 100})
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    if plot_type == 'violin':
        ax = sns.violinplot(x='Strategy', y='Probability', data=df, palette='muted')
    else:
        ax = sns.barplot(x='Strategy', y='Probability', data=df, capsize=.1, palette='muted')
        
    # --- Stats ---
    from scipy import stats
    def get_sig(p):
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return "ns"
        
    def draw_bracket(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
        ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='black')

    d1 = df[df['Strategy'] == 'Markov Final']['Probability']
    d2 = df[df['Strategy'] == 'Markov (50/50 Pooled)']['Probability']
    d3 = df[df['Strategy'] == 'Markov Inversed']['Probability']
    
    _, p12 = stats.mannwhitneyu(d1, d2)
    _, p23 = stats.mannwhitneyu(d2, d3)
    _, p13 = stats.mannwhitneyu(d1, d3)
    
    print(f"Stats (Markov Final vs Pooled): p={p12:.4e}")
    print(f"Stats (Pooled vs Inversed): p={p23:.4e}")
    print(f"Stats (Markov Final vs Inversed): p={p13:.4e}")
    
    ymax = df['Probability'].max()
    y_b = max(ymax + 2, 75)
    
    draw_bracket(ax, 0, 1, y_b, 1.5, get_sig(p12))
    draw_bracket(ax, 1, 2, y_b, 1.5, get_sig(p23))
    draw_bracket(ax, 0, 2, y_b + 7, 1.5, get_sig(p13))
    # -------------
    
    plt.ylim(0, max(100, y_b + 12))
    plt.title('Hole Finding Probability on Inverted Ovoid (effective_radius = hole_radius + 0.3)', fontsize=14)
    plt.ylabel('Success Probability (%)', fontsize=12)
    plt.xlabel('Strategy', fontsize=12)
    sns.despine()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    plt.show()










# ==============================================================================
# Recovered Missing Functions
# ==============================================================================

def run_advanced_trajectory_tracking(
    sites_dir,
    ethogram_path=None,
    output_file=None,
    num_simulations=1000,
    overlap_threshold=0.50,
    use_rectangle=False,
    clamp_dims=(0.7, 0.4),
    max_steps=10000,
    filter_teeth_hole_contact=False,
    show_percent_below_star=False,
    use_k_animal_as_budget=False,
    length_params={'exp_scale': 1.0, 'pareto_shape': 2.0},
    record_max_steps=100,
    randomize_starts=False
):
    import os
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    bottom_files = sorted([
        f for f in glob.glob(os.path.join(sites_dir, "*_sites.json")) 
        if not f.endswith("_front_sites.json") and "beyonc" not in f.lower()
    ])

    if not bottom_files:
        print(f"No *_sites.json files found in directory: {sites_dir}")
        return None, None

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

    etho_data = None
    if ethogram_path and os.path.exists(ethogram_path):
        import pickle
        try:
            with open(ethogram_path, 'rb') as f:
                etho_data = pickle.load(f)
        except Exception:
            pass

    a_true, b_true = 1.2602, 1.3749
    z_max_true = np.sqrt(a_true**2 + b_true**2)



    def _apply_z_move(z_in, th_in, dz, z_max):
        limit = z_max * 0.999
        z_out = z_in + dz
        th_out = th_in
        if z_out > limit:
            overshoot = z_out - limit
            z_out = limit - overshoot
            th_out = (th_out + np.pi) % (2*np.pi)
        elif z_out < -limit:
            overshoot = (-limit) - z_out
            z_out = -limit + overshoot
            th_out = (th_out + np.pi) % (2*np.pi)
        return z_out, th_out

    def scale_point(p):
        mag = np.linalg.norm(p)
        z = p[2]
        target_mag = np.sqrt(max(0.0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2) + z**2)
        if mag > 0: return np.array(p) * (target_mag / mag)
        return p

    def adjust_start_point_if_needed(pt_scaled, use_rect, dims, _frame):
        if pt_scaled[2] > 0: new_z = 0.4
        else: new_z = -0.4
        r = np.sqrt(max(0, np.sqrt(b_true**4 + 4*a_true**2*new_z**2) - new_z**2 - a_true**2))
        th = np.arctan2(pt_scaled[1], pt_scaled[0])
        return np.array([r*np.cos(th), r*np.sin(th), new_z])

    # Empirical Parameters
    _pLL = 0.65
    _pTT = 0.26
    sd_T = 0.3
    mag_L = np.random.normal(loc=0.967, scale=0.3, size=50000)
    mag_L = np.clip(mag_L, 0.01, None)
    mag_T = np.random.normal(loc=0.864, scale=sd_T, size=50000)
    mag_T = np.clip(mag_T, 0.01, None)
    _long_amps_abs = mag_L
    sign_T = np.random.choice([-1, 1], size=50000)
    _trans_amps = mag_T * sign_T
    pareto_shape = 2.0

    sim_records = []
    max_steps = record_max_steps

    def get_grid_cell(pt):
        r = np.linalg.norm(pt)
        if r == 0: return (0, 0)
        theta = np.arccos(pt[2] / r)
        phi = np.arctan2(pt[1], pt[0])
        d_ang = np.pi / 18.0 
        t_idx = int(theta / d_ang)
        p_idx = int((phi + np.pi) / d_ang)
        return (t_idx, p_idx)

    animal_records = {}

    for bf in bottom_files:
        name = os.path.basename(bf).replace("_sites.json", "")
        import json
        
        # --- DYNAMIC HOLE LOCATION ---
        p_hole = np.array([0.0, -0.75002, 1.01296])
        with open(bf, 'r') as f:
            raw_text = f.read()
            if "hole_0" in raw_text:
                data = json.loads(raw_text)
                if "p" in data["hole_0"]:
                    p_raw = data["hole_0"]["p"]
                    # Physical scale scaling
                    scale_factor = z_max_true / np.sqrt(1.0**2 + 1.1**2)
                    p_hole = np.array([float(p_raw[0]) * scale_factor, float(p_raw[1]) * scale_factor, float(p_raw[2]) * scale_factor])
            
        global_tracker = {'dists': [], 'coords': [], 'cells': set(), 'cov': [], 'tracking_active': False}
        def get_distance_to_hole(pt):
            dist_to_center = np.linalg.norm(pt - p_hole)
            d = dist_to_center - (max(clamp_dims) / 2.0) if use_rectangle else dist_to_center
            if global_tracker['tracking_active']:
                global_tracker['dists'].append(d)
                global_tracker['coords'].append(pt)
                global_tracker['cells'].add(get_grid_cell(pt))
                global_tracker['cov'].append(len(global_tracker['cells']))
            return d
        # ----------------------------

        with open(bf) as f:
            valid_bottom = parse_sites(json.load(f).get('sites', []))
            
        ff = bf.replace("_sites.json", "_front_sites.json")
        valid_front = []
        if os.path.exists(ff):
            with open(ff) as f:
                valid_front = parse_sites(json.load(f).get('sites', []))
                
        if not valid_bottom and not valid_front: continue
        
        all_sites = sorted(valid_bottom + valid_front, key=lambda x: x['frame'])
        
        # Scaling and Adjusting start point like in other function
        b_first_scaled = scale_point(valid_bottom[0]['p']) if valid_bottom else scale_point(valid_front[0]['p'])
        b_first_frame = valid_bottom[0]['frame'] if valid_bottom else valid_front[0]['frame']
        
        d_b = get_distance_to_hole(b_first_scaled)
        f_math_scaled = np.array([-b_first_scaled[0], -b_first_scaled[1], b_first_scaled[2]])
        d_f = get_distance_to_hole(f_math_scaled)
        
        if d_b <= overlap_threshold or d_f <= overlap_threshold:
            b_first_scaled = adjust_start_point_if_needed(b_first_scaled, use_rectangle, clamp_dims, b_first_frame)
            
        start_z = b_first_scaled[2]
        start_th = np.arctan2(b_first_scaled[1], b_first_scaled[0])
        
        k_animal = None
        if filter_teeth_hole_contact and etho_data:
            contact_frames = []
            video_base = name.split('_')[0].upper()
            
            video_events = []
            found_obs = None
            for obs_name, obs in etho_data['observations'].items():
                if video_base in obs_name.upper():
                    found_obs = obs
                    video_events = obs.get('events', [])
                    break
            
            if not found_obs and 'First_batch' in etho_data['observations']:
                obs = etho_data['observations']['First_batch']
                media_list = obs.get('file', {}).get('1', [])
                media_lengths = obs.get('media_info', {}).get('length', {})
                current_time = 0.0
                video_info = {}
                for m in media_list:
                    length = media_lengths.get(m, 0.0)
                    m_base = os.path.basename(m).replace(".MP4", "").replace(".mp4", "")
                    video_info[m_base] = {'start': current_time, 'end': current_time + length}
                    current_time += length
                
                matched_video = None
                for m_base in video_info.keys():
                    if m_base in video_base or video_base in m_base:
                        matched_video = m_base
                        break
                
                if matched_video:
                    info = video_info[matched_video]
                    start_t, end_t = info['start'], info['end']
                    raw_events = obs.get('events', [])
                    for e in raw_events:
                        t = float(e[0])
                        if start_t <= t <= end_t:
                            video_events.append(e)

            for e in video_events:
                behavior_name = str(e[2]).lower()
                if 'teeth' in behavior_name and 'hole' in behavior_name and 'contact' in behavior_name:
                    try: contact_frames.append(int(e[5]))
                    except: pass
        
            if contact_frames:
                all_actual = []
                for s in valid_bottom: all_actual.append({'frame': s['frame']})
                for s in valid_front: all_actual.append({'frame': s['frame']})
                all_actual.sort(key=lambda x: x['frame'])
                
                for idx_c, x_c in enumerate(all_actual):
                    site_frame = x_c['frame']
                    if any(abs(site_frame - cf) <= 5 for cf in contact_frames):
                        k_animal = idx_c + 1
                        break
                        
        if k_animal is None:
            total_pairs = max(len(valid_bottom), len(valid_front))
            k_animal = total_pairs * 2
            
        animal_records[name] = k_animal / 2.0

        # ---------------------------------------------------------
        # Markov Final Simulation
        # ---------------------------------------------------------
        for sim_idx in range(num_simulations):
            global_tracker['dists'] = []
            global_tracker['coords'] = []
            global_tracker['cells'] = set()
            global_tracker['cov'] = []
            global_tracker['tracking_active'] = True
            hit_step = -1
            if randomize_starts:
                z = np.random.uniform(-z_max_true*0.999, z_max_true*0.999)
                th = np.random.uniform(0, 2*np.pi)
            else:
                z, th = start_z, start_th
            touched = False
            k_sim = 1
            cur_type = np.random.choice(['L', 'T'], p=[0.68, 0.32])
            
            pt_b_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos(th), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin(th), z])
            pt_f_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos((th+np.pi)%(2*np.pi)), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin((th+np.pi)%(2*np.pi)), z])
            d_b0 = get_distance_to_hole(pt_b_first)
            d_f0 = get_distance_to_hole(pt_f_first)
            if (d_b0 <= overlap_threshold or d_f0 <= overlap_threshold) and hit_step == -1: hit_step = 1
            if True:
                for step in range(1, max_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(_long_amps_abs)
                        if abs(z) < z_max_true / 3.0: prob_towards = 0.5614
                        elif abs(z) < 2.0 * z_max_true / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if z > 0: dz_sign = -1 if towards else 1
                        elif z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        dz = dz_mag * dz_sign
                        z, th = _apply_z_move(z, th, dz, z_max_true)
                        cur_type = 'L' if np.random.random() < _pLL else 'T'
                    else:
                        darc = np.random.choice(_trans_amps)
                        t1 = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                        r_cur = max(1e-6, np.sqrt(max(0.0, t1 - z**2 - a_true**2)))
                        th = (th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < _pTT else 'L'
                        
                    t1_fix = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur_fix = max(1e-6, np.sqrt(max(0.0, t1_fix - z**2 - a_true**2)))
                    pt_b = np.array([r_cur_fix*np.cos(th), r_cur_fix*np.sin(th), z])
                    pt_f = np.array([r_cur_fix*np.cos((th + np.pi)%(2*np.pi)), r_cur_fix*np.sin((th + np.pi)%(2*np.pi)), z])
                    k_sim += 2
                    d_b = get_distance_to_hole(pt_b)
                    d_f = get_distance_to_hole(pt_f)
                    if (d_b <= overlap_threshold or d_f <= overlap_threshold) and hit_step == -1: hit_step = k_sim
            sim_records.append({'session': name, 'strategy': 'Markov Final', 'k_sim': hit_step if hit_step != -1 else record_max_steps*2, 'initial_dist_b': global_tracker['dists'][0] if len(global_tracker['dists'])>0 else 0, 'initial_dist_f': global_tracker['dists'][1] if len(global_tracker['dists'])>1 else 0, 'dists': global_tracker['dists'][:record_max_steps*2], 'coords': global_tracker['coords'][:record_max_steps*2], 'coverage': global_tracker['cov'][:record_max_steps*2]})

        # ---------------------------------------------------------
        # CRW Simulation (Theoretical)
        # ---------------------------------------------------------
        for sim_idx in range(num_simulations):
            global_tracker['dists'] = []
            global_tracker['coords'] = []
            global_tracker['cells'] = set()
            global_tracker['cov'] = []
            global_tracker['tracking_active'] = True
            hit_step = -1
            if randomize_starts:
                z = np.random.uniform(-z_max_true*0.999, z_max_true*0.999)
                th = np.random.uniform(0, 2*np.pi)
            else:
                z, th = start_z, start_th
            touched = False
            k_sim = 1
            direction = np.random.uniform(0, 2*np.pi)
            
            pt_b_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos(th), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin(th), z])
            pt_f_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos((th+np.pi)%(2*np.pi)), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin((th+np.pi)%(2*np.pi)), z])
            d_b0 = get_distance_to_hole(pt_b_first)
            d_f0 = get_distance_to_hole(pt_f_first)
            if (d_b0 <= overlap_threshold or d_f0 <= overlap_threshold) and hit_step == -1: hit_step = 1
            if True:
                for step in range(1, max_steps):
                    step_len = np.random.exponential(scale=1.0)
                    direction = (direction + (2 * np.arctan((1 - 0.735) / (1 + 0.735) * np.tan(np.pi * (np.random.uniform(0, 1) - 0.5))))) % (2*np.pi)
                    dz = step_len * np.cos(direction)
                    darc = step_len * np.sin(direction)
                    z, th = _apply_z_move(z, th, dz, z_max_true)
                    t1 = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur = max(1e-6, np.sqrt(max(0.0, t1 - z**2 - a_true**2)))
                    th = (th + darc/r_cur) % (2*np.pi)
                    
                    t1_fix = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur_fix = max(1e-6, np.sqrt(max(0.0, t1_fix - z**2 - a_true**2)))
                    pt_b = np.array([r_cur_fix*np.cos(th), r_cur_fix*np.sin(th), z])
                    pt_f = np.array([r_cur_fix*np.cos((th + np.pi)%(2*np.pi)), r_cur_fix*np.sin((th + np.pi)%(2*np.pi)), z])
                    k_sim += 2
                    d_b = get_distance_to_hole(pt_b)
                    d_f = get_distance_to_hole(pt_f)
                    if (d_b <= overlap_threshold or d_f <= overlap_threshold) and hit_step == -1: hit_step = k_sim
            sim_records.append({'session': name, 'strategy': 'Correlated Random Walk', 'k_sim': hit_step if hit_step != -1 else record_max_steps*2, 'initial_dist_b': global_tracker['dists'][0] if len(global_tracker['dists'])>0 else 0, 'initial_dist_f': global_tracker['dists'][1] if len(global_tracker['dists'])>1 else 0, 'dists': global_tracker['dists'][:record_max_steps*2], 'coords': global_tracker['coords'][:record_max_steps*2], 'coverage': global_tracker['cov'][:record_max_steps*2]})

        # ---------------------------------------------------------
        # URW Simulation
        # ---------------------------------------------------------
        for sim_idx in range(num_simulations):
            global_tracker['dists'] = []
            global_tracker['coords'] = []
            global_tracker['cells'] = set()
            global_tracker['cov'] = []
            global_tracker['tracking_active'] = True
            hit_step = -1
            if randomize_starts:
                z = np.random.uniform(-z_max_true*0.999, z_max_true*0.999)
                th = np.random.uniform(0, 2*np.pi)
            else:
                z, th = start_z, start_th
            touched = False
            k_sim = 1
            
            pt_b_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos(th), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin(th), z])
            pt_f_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos((th+np.pi)%(2*np.pi)), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin((th+np.pi)%(2*np.pi)), z])
            d_b0 = get_distance_to_hole(pt_b_first)
            d_f0 = get_distance_to_hole(pt_f_first)
            if (d_b0 <= overlap_threshold or d_f0 <= overlap_threshold) and hit_step == -1: hit_step = 1
            if True:
                for step in range(1, max_steps):
                    step_len = np.random.exponential(scale=1.0)
                    direction = np.random.uniform(0, 2*np.pi)
                    dz = step_len * np.cos(direction)
                    darc = step_len * np.sin(direction)
                    z, th = _apply_z_move(z, th, dz, z_max_true)
                    t1 = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur = max(1e-6, np.sqrt(max(0.0, t1 - z**2 - a_true**2)))
                    th = (th + darc/r_cur) % (2*np.pi)
                    
                    t1_fix = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur_fix = max(1e-6, np.sqrt(max(0.0, t1_fix - z**2 - a_true**2)))
                    pt_b = np.array([r_cur_fix*np.cos(th), r_cur_fix*np.sin(th), z])
                    pt_f = np.array([r_cur_fix*np.cos((th + np.pi)%(2*np.pi)), r_cur_fix*np.sin((th + np.pi)%(2*np.pi)), z])
                    k_sim += 2
                    d_b = get_distance_to_hole(pt_b)
                    d_f = get_distance_to_hole(pt_f)
                    if (d_b <= overlap_threshold or d_f <= overlap_threshold) and hit_step == -1: hit_step = k_sim
            sim_records.append({'session': name, 'strategy': 'Uncorrelated Walk', 'k_sim': hit_step if hit_step != -1 else record_max_steps*2, 'initial_dist_b': global_tracker['dists'][0] if len(global_tracker['dists'])>0 else 0, 'initial_dist_f': global_tracker['dists'][1] if len(global_tracker['dists'])>1 else 0, 'dists': global_tracker['dists'][:record_max_steps*2], 'coords': global_tracker['coords'][:record_max_steps*2], 'coverage': global_tracker['cov'][:record_max_steps*2]})

        # ---------------------------------------------------------
        # SAW (M=2) Simulation
        # ---------------------------------------------------------
        for sim_idx in range(num_simulations):
            global_tracker['dists'] = []
            global_tracker['coords'] = []
            global_tracker['cells'] = set()
            global_tracker['cov'] = []
            global_tracker['tracking_active'] = True
            hit_step = -1
            if randomize_starts:
                z = np.random.uniform(-z_max_true*0.999, z_max_true*0.999)
                th = np.random.uniform(0, 2*np.pi)
            else:
                z, th = start_z, start_th
            history = [(z, th)]
            touched = False
            k_sim = 1
            
            pt_b_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos(th), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin(th), z])
            pt_f_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos((th+np.pi)%(2*np.pi)), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin((th+np.pi)%(2*np.pi)), z])
            d_b0 = get_distance_to_hole(pt_b_first)
            d_f0 = get_distance_to_hole(pt_f_first)
            if (d_b0 <= overlap_threshold or d_f0 <= overlap_threshold) and hit_step == -1: hit_step = 1
            if True:
                for step in range(1, max_steps):
                    valid_step = False
                    for _tries in range(50):
                        step_len = np.random.exponential(scale=1.0)
                        direction = np.random.uniform(0, 2*np.pi)
                        dz = step_len * np.cos(direction)
                        darc = step_len * np.sin(direction)
                        cand_z, cand_th = _apply_z_move(z, th, dz, z_max_true)
                        t1 = np.sqrt(b_true**4 + 4*a_true**2*cand_z**2)
                        r_cand = max(1e-6, np.sqrt(max(0.0, t1 - cand_z**2 - a_true**2)))
                        cand_th = (cand_th + darc/r_cand) % (2*np.pi)
                        
                        cand_pt = np.array([r_cand*np.cos(cand_th), r_cand*np.sin(cand_th), cand_z])
                        too_close = False
                        for (hz, hth) in history:
                            h_t1 = np.sqrt(b_true**4 + 4*a_true**2*hz**2)
                            h_r = max(1e-6, np.sqrt(max(0.0, h_t1 - hz**2 - a_true**2)))
                            hpt = np.array([h_r*np.cos(hth), h_r*np.sin(hth), hz])
                            if np.linalg.norm(cand_pt - hpt) < 0.5:
                                too_close = True
                                break
                        
                        if not too_close:
                            z, th = cand_z, cand_th
                            valid_step = True
                            break
                    if not valid_step:
                        z, th = cand_z, cand_th
                        
                    history.append((z, th))
                    if len(history) > 2:
                        history.pop(0)
                        
                    t1_fix = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur_fix = max(1e-6, np.sqrt(max(0.0, t1_fix - z**2 - a_true**2)))
                    pt_b = np.array([r_cur_fix*np.cos(th), r_cur_fix*np.sin(th), z])
                    pt_f = np.array([r_cur_fix*np.cos((th + np.pi)%(2*np.pi)), r_cur_fix*np.sin((th + np.pi)%(2*np.pi)), z])
                    k_sim += 2
                    d_b = get_distance_to_hole(pt_b)
                    d_f = get_distance_to_hole(pt_f)
                    if (d_b <= overlap_threshold or d_f <= overlap_threshold) and hit_step == -1: hit_step = k_sim
            sim_records.append({'session': name, 'strategy': 'Self-Avoiding Walk (M=2)', 'k_sim': hit_step if hit_step != -1 else record_max_steps*2, 'initial_dist_b': global_tracker['dists'][0] if len(global_tracker['dists'])>0 else 0, 'initial_dist_f': global_tracker['dists'][1] if len(global_tracker['dists'])>1 else 0, 'dists': global_tracker['dists'][:record_max_steps*2], 'coords': global_tracker['coords'][:record_max_steps*2], 'coverage': global_tracker['cov'][:record_max_steps*2]})

        # SAW (M=1) Simulation
        # ---------------------------------------------------------
        for sim_idx in range(num_simulations):
            global_tracker['dists'] = []
            global_tracker['coords'] = []
            global_tracker['cells'] = set()
            global_tracker['cov'] = []
            global_tracker['tracking_active'] = True
            hit_step = -1
            if randomize_starts:
                z = np.random.uniform(-z_max_true*0.999, z_max_true*0.999)
                th = np.random.uniform(0, 2*np.pi)
            else:
                z, th = start_z, start_th
            history = [(z, th)]
            touched = False
            k_sim = 1
            
            pt_b_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos(th), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin(th), z])
            pt_f_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos((th+np.pi)%(2*np.pi)), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin((th+np.pi)%(2*np.pi)), z])
            d_b0 = get_distance_to_hole(pt_b_first)
            d_f0 = get_distance_to_hole(pt_f_first)
            if (d_b0 <= overlap_threshold or d_f0 <= overlap_threshold) and hit_step == -1: hit_step = 1
            if True:
                for step in range(1, max_steps):
                    valid_step = False
                    for _tries in range(50):
                        step_len = np.random.exponential(scale=1.0)
                        direction = np.random.uniform(0, 2*np.pi)
                        dz = step_len * np.cos(direction)
                        darc = step_len * np.sin(direction)
                        cand_z, cand_th = _apply_z_move(z, th, dz, z_max_true)
                        t1 = np.sqrt(b_true**4 + 4*a_true**2*cand_z**2)
                        r_cand = max(1e-6, np.sqrt(max(0.0, t1 - cand_z**2 - a_true**2)))
                        cand_th = (cand_th + darc/r_cand) % (2*np.pi)
                        
                        cand_pt = np.array([r_cand*np.cos(cand_th), r_cand*np.sin(cand_th), cand_z])
                        too_close = False
                        for (hz, hth) in history:
                            h_t1 = np.sqrt(b_true**4 + 4*a_true**2*hz**2)
                            h_r = max(1e-6, np.sqrt(max(0.0, h_t1 - hz**2 - a_true**2)))
                            hpt = np.array([h_r*np.cos(hth), h_r*np.sin(hth), hz])
                            if np.linalg.norm(cand_pt - hpt) < 0.5:
                                too_close = True
                                break
                        
                        if not too_close:
                            z, th = cand_z, cand_th
                            valid_step = True
                            break
                    if not valid_step:
                        z, th = cand_z, cand_th
                        
                    history.append((z, th))
                    if len(history) > 1:
                        history.pop(0)
                        
                    t1_fix = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur_fix = max(1e-6, np.sqrt(max(0.0, t1_fix - z**2 - a_true**2)))
                    pt_b = np.array([r_cur_fix*np.cos(th), r_cur_fix*np.sin(th), z])
                    pt_f = np.array([r_cur_fix*np.cos((th + np.pi)%(2*np.pi)), r_cur_fix*np.sin((th + np.pi)%(2*np.pi)), z])
                    k_sim += 2
                    d_b = get_distance_to_hole(pt_b)
                    d_f = get_distance_to_hole(pt_f)
                    if (d_b <= overlap_threshold or d_f <= overlap_threshold) and hit_step == -1: hit_step = k_sim
            sim_records.append({'session': name, 'strategy': 'Self-Avoiding Walk (M=1)', 'k_sim': hit_step if hit_step != -1 else record_max_steps*2, 'initial_dist_b': global_tracker['dists'][0] if len(global_tracker['dists'])>0 else 0, 'initial_dist_f': global_tracker['dists'][1] if len(global_tracker['dists'])>1 else 0, 'dists': global_tracker['dists'][:record_max_steps*2], 'coords': global_tracker['coords'][:record_max_steps*2], 'coverage': global_tracker['cov'][:record_max_steps*2]})


        # ---------------------------------------------------------
        # Levy Walk Simulation
        # ---------------------------------------------------------
        for sim_idx in range(num_simulations):
            global_tracker['dists'] = []
            global_tracker['coords'] = []
            global_tracker['cells'] = set()
            global_tracker['cov'] = []
            global_tracker['tracking_active'] = True
            hit_step = -1
            if randomize_starts:
                z = np.random.uniform(-z_max_true*0.999, z_max_true*0.999)
                th = np.random.uniform(0, 2*np.pi)
            else:
                z, th = start_z, start_th
            touched = False
            k_sim = 1
            
            pt_b_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos(th), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin(th), z])
            pt_f_first = np.array([np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.cos((th+np.pi)%(2*np.pi)), 
                                   np.sqrt(np.maximum(0, np.sqrt(b_true**4 + 4*a_true**2*z**2) - z**2 - a_true**2))*np.sin((th+np.pi)%(2*np.pi)), z])
            d_b0 = get_distance_to_hole(pt_b_first)
            d_f0 = get_distance_to_hole(pt_f_first)
            if (d_b0 <= overlap_threshold or d_f0 <= overlap_threshold) and hit_step == -1: hit_step = 1
            if True:
                for step in range(1, max_steps):
                    step_len = np.random.pareto(pareto_shape) * 0.5
                    direction = np.random.uniform(0, 2*np.pi)
                    dz = step_len * np.cos(direction)
                    darc = step_len * np.sin(direction)
                    z, th = _apply_z_move(z, th, dz, z_max_true)
                    t1 = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur = max(1e-6, np.sqrt(max(0.0, t1 - z**2 - a_true**2)))
                    th = (th + darc/r_cur) % (2*np.pi)
                    
                    t1_fix = np.sqrt(b_true**4 + 4*a_true**2*z**2)
                    r_cur_fix = max(1e-6, np.sqrt(max(0.0, t1_fix - z**2 - a_true**2)))
                    pt_b = np.array([r_cur_fix*np.cos(th), r_cur_fix*np.sin(th), z])
                    pt_f = np.array([r_cur_fix*np.cos((th + np.pi)%(2*np.pi)), r_cur_fix*np.sin((th + np.pi)%(2*np.pi)), z])
                    k_sim += 2
                    d_b = get_distance_to_hole(pt_b)
                    d_f = get_distance_to_hole(pt_f)
                    if (d_b <= overlap_threshold or d_f <= overlap_threshold) and hit_step == -1: hit_step = k_sim
            sim_records.append({'session': name, 'strategy': 'Lévy Walk', 'k_sim': hit_step if hit_step != -1 else record_max_steps*2, 'initial_dist_b': global_tracker['dists'][0] if len(global_tracker['dists'])>0 else 0, 'initial_dist_f': global_tracker['dists'][1] if len(global_tracker['dists'])>1 else 0, 'dists': global_tracker['dists'][:record_max_steps*2], 'coords': global_tracker['coords'][:record_max_steps*2], 'coverage': global_tracker['cov'][:record_max_steps*2]})

        print(f"Processed multi-strategy sim for {name} (K_animal = {k_animal})")

    if not sim_records:
        print("No successful simulations recorded.")
        return None, None

    df_sim = pd.DataFrame(sim_records)
    if not df_sim.empty:
        df_sim['k_sim'] = df_sim['k_sim'] / 2.0
    
    # Plotting Average Summary

    # k_sim is halved to match clamps vs sampling steps scale
    if not df_sim.empty:
        df_sim['k_sim'] = df_sim['k_sim'] / 2.0
    return df_sim



def plot_patchiness_real_peanut_two_faces(json_file, output_file=None, max_steps=8):
    '''
    Plots the footprint patchiness of a real empirical peanut sequence 
    on the custom Two-Faces projection map.
    '''
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    import os

    # Handle file path and try to load front sites if the user passed back sites
    load_file = json_file
    if json_file.endswith('_sites.json') and not json_file.endswith('_front_sites.json'):
        front_file = json_file.replace('_sites.json', '_front_sites.json')
        if os.path.exists(front_file):
            load_file = front_file
            print(f"Auto-switched to front sites: {os.path.basename(front_file)}")
            
    with open(load_file, 'r') as f:
        data = json.load(f)
        
    pts_raw = data.get('sites', [])
    if not pts_raw:
        print("No sites found in JSON.")
        return
        
    fronts = np.array([pt[:3] for pt in pts_raw])[:max_steps]
    
    a_true, b_true = 1.2602, 1.3749
    z_max_true = np.sqrt(a_true**2 + b_true**2)
    num_pts = 300
    z_vals = np.linspace(-z_max_true + 1e-4, z_max_true - 1e-4, num_pts)
    th_vals = np.linspace(-np.pi, np.pi, num_pts)
    Z, TH = np.meshgrid(z_vals, th_vals)
    R_sq = b_true**4 + 4*a_true**2*Z**2
    R = np.sqrt(np.maximum(0, np.sqrt(R_sq) - Z**2 - a_true**2))
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    fig, ax_2d = plt.subplots(1, 1, figsize=(7, 8))
    
    S = 1.85
    faces_map = {0: {'az': 0, 'cx': -S/1.5, 'cy': 0},
                 1: {'az': 180, 'cx': S/1.5, 'cy': 0}}
                 
    def peanut_radius_local(z, a=a_true, b=b_true):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    width_vals = peanut_radius_local(z_vals, a_true, b_true) * (np.pi / 2)

    def map_to_canvas(x, y, z):
        angle = np.degrees(np.arctan2(y, x)) % 360
        best_diff = 360; best_fid = -1
        for fid, f in faces_map.items():
            diff = abs(angle - f['az'])
            if diff > 180: diff = 360 - diff
            if diff < best_diff: best_diff = diff; best_fid = fid
        f = faces_map[best_fid]
        raw_diff = angle - f['az']
        if raw_diff > 180: raw_diff -= 360
        elif raw_diff < -180: raw_diff += 360
        delta_rad = np.radians(raw_diff)
        curr_r = np.sqrt(x**2 + y**2)
        arc_dist = curr_r * delta_rad
        return f['cx'] + arc_dist, f['cy'] + z

    mapped_pts = np.array([map_to_canvas(x, y, z) for x, y, z in pts])
    
    title = f'Real Peanut ({len(fronts)} steps)'
    color = '#2ca02c' # distinct green
    
    ax_2d.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax_2d.set_aspect('equal')
    ax_2d.set_xlim(-S*1.5, S*1.5)
    ax_2d.set_ylim(-2.5, 2.5)
    ax_2d.axis('off')
    
    for fid, f in faces_map.items():
        cx, cy = f['cx'], f['cy']
        ax_2d.plot(cx + width_vals, cy + z_vals, 'k-')
        ax_2d.plot(cx - width_vals, cy + z_vals, 'k-')
        ax_2d.fill_betweenx(cy + z_vals, cx - width_vals, cx + width_vals, color='#a6806d', alpha=0.5)
        label = "Front (0 deg)" if fid == 0 else "Back (180 deg)"
        ax_2d.text(cx, cy + z_max_true + 0.3, label, ha='center', fontweight='bold', fontsize=16)

    from from_antigravity_peanuts import get_geodesic_profile_distance
    # Precompute geodesic S profile for the Z values
    s_vals_line = np.array([get_geodesic_profile_distance(0, z, a_true, b_true) for z in z_vals])
    s_vals_line = s_vals_line - s_vals_line[0]
    
    # Flattened parameters for vectorization
    pts_z = Z.ravel()
    pts_th = TH.ravel()
    pts_r = R.ravel()
    # Map z_vals to s_vals for all points
    pts_s = np.interp(pts_z, z_vals, s_vals_line)
    
    covered_mask = np.zeros(len(pts), dtype=bool)
    covered_count = np.zeros(len(pts), dtype=int)
    
    for p_clamp in fronts:
        xc, yc, zc = p_clamp
        c_z_idx = np.argmin(np.abs(z_vals - zc))
        c_S = s_vals_line[c_z_idx]
        c_th = np.arctan2(yc, xc)
        
        # Distance checks (0.7cm x 0.4cm oriented rectangle)
        delta_s = np.abs(pts_s - c_S)
        diff_th = np.abs(pts_th - c_th)
        diff_th[diff_th > np.pi] = 2*np.pi - diff_th[diff_th > np.pi]
        delta_arc = pts_r * diff_th
        
        # Mask
        clamp_mask = (delta_s <= 0.35) & (delta_arc <= 0.20)
        covered_mask[clamp_mask] = True
        covered_count[clamp_mask] += 1
        
    cov_pts = mapped_pts[covered_mask]
    
    # Calculate areas
    with np.errstate(divide='ignore', invalid='ignore'):
        dr_dz = (2*a_true**2*Z/np.sqrt(R_sq) - Z) / R
        dr_dz[np.isnan(dr_dz)] = 0.0
        
    dA = R * np.sqrt(1 + dr_dz**2) * (z_vals[1] - z_vals[0]) * (th_vals[1] - th_vals[0])
    total_area = np.sum(dA)
    
    covered_area = np.sum(dA.ravel()[covered_mask])
    overlap_area = np.sum(dA.ravel()[covered_count >= 2])
    
    # Calculate total geodesic distance traveled
    total_dist = 0.0
    if len(fronts) > 1:
        for i in range(1, len(fronts)):
            x1, y1, z1 = fronts[i-1]
            x2, y2, z2 = fronts[i]
            
            idx1 = np.argmin(np.abs(z_vals - z1))
            idx2 = np.argmin(np.abs(z_vals - z2))
            s1, s2 = s_vals_line[idx1], s_vals_line[idx2]
            ds = s1 - s2
            
            th1 = np.arctan2(y1, x1)
            th2 = np.arctan2(y2, x2)
            dth = np.abs(th1 - th2)
            if dth > np.pi: dth = 2*np.pi - dth
            
            r1 = np.sqrt(x1**2 + y1**2)
            r2 = np.sqrt(x2**2 + y2**2)
            r_avg = (r1 + r2) / 2.0
            darc = dth * r_avg
            
            total_dist += np.sqrt(ds**2 + darc**2)
    
    print(f"\\n--- {title} ---")
    print(f"Total Peanut Surface Area: {total_area:.2f} cm^2")
    print(f"Total Geodesic Distance Traveled: {total_dist:.2f} cm")
    print(f"Area Covered: {covered_area:.2f} cm^2 ({(covered_area/total_area)*100:.1f}%)")
    print(f"Area Overlapped: {overlap_area:.2f} cm^2 ({(overlap_area/total_area)*100:.1f}%)")

    ax_2d.scatter(cov_pts[:,0], cov_pts[:,1], color=color, s=2, alpha=0.8)
    
    if len(fronts) > 0:
        mapped_fronts = np.array([map_to_canvas(x, y, z) for x, y, z in fronts])
        for i in range(len(fronts)-1):
            x1, y1, z1 = fronts[i]
            x2, y2, z2 = fronts[i+1]
            
            idx1 = np.argmin(np.abs(z_vals - z1))
            idx2 = np.argmin(np.abs(z_vals - z2))
            s1, s2 = s_vals_line[idx1], s_vals_line[idx2]
            
            th1 = np.arctan2(y1, x1)
            th2 = np.arctan2(y2, x2)
            
            if th2 - th1 > np.pi: th1 += 2*np.pi
            elif th1 - th2 > np.pi: th2 += 2*np.pi
            
            num_interp = 20
            s_interp = np.linspace(s1, s2, num_interp)
            th_interp = np.linspace(th1, th2, num_interp)
            
            z_interp = np.interp(s_interp, s_vals_line, z_vals)
            R_sq_i = b_true**4 + 4*a_true**2*z_interp**2
            R_interp = np.sqrt(np.maximum(0, np.sqrt(R_sq_i) - z_interp**2 - a_true**2))
            x_interp = R_interp * np.cos(th_interp)
            y_interp = R_interp * np.sin(th_interp)
            
            pts_interp = np.column_stack((x_interp, y_interp, z_interp))
            mapped_interp = np.array([map_to_canvas(x, y, z) for x, y, z in pts_interp])
            
            # Draw curved geodesic segments safely
            for j in range(num_interp - 1):
                pA = mapped_interp[j]
                pB = mapped_interp[j+1]
                if np.linalg.norm(pA - pB) < 1.0: # avoid face-jumping lines
                    ax_2d.plot([pA[0], pB[0]], [pA[1], pB[1]], color='black', linestyle='-', linewidth=2, alpha=0.6)
            
            # Add text label for the true distance
            ds = s1 - s2
            r1 = np.sqrt(x1**2 + y1**2)
            r2 = np.sqrt(x2**2 + y2**2)
            local_d = np.sqrt(ds**2 + (np.abs(th1 - th2) * (r1 + r2)/2.0)**2)
            
            mid_pt = mapped_interp[num_interp//2]
            ax_2d.text(mid_pt[0], mid_pt[1], f"{local_d:.1f}", color='black', fontsize=9, 
                       fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
        ax_2d.scatter(mapped_fronts[:,0], mapped_fronts[:,1], color='black', s=20, zorder=10)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print('Saved patchiness visualization to', output_file)
    plt.show()



def plot_scientific_walks_with_markov(steps=30, saw_radius=0.6, seed=100, output_file=None):
    """
    Generates a publication-quality 2x3 comparison plot of 5 random walks 
    in continuous flat 2D Cartesian space, including the behavioral Markov Final model.
    """
    # Enforce reproducibility
    np.random.seed(seed)
    
    # --- 1. Uncorrelated Random Walk (URW) ---
    l_urw = np.random.exponential(scale=1.0, size=steps)
    theta_urw = np.random.uniform(-np.pi, np.pi, steps)
    x_urw = np.concatenate(([0], np.cumsum(l_urw * np.cos(theta_urw))))
    y_urw = np.concatenate(([0], np.cumsum(l_urw * np.sin(theta_urw))))

    # --- 2. Correlated Random Walk (CRW) ---
    l_crw = np.random.exponential(scale=1.0, size=steps)
    theta_crw = np.zeros(steps)
    theta_crw[0] = np.random.uniform(-np.pi, np.pi)
    r_concentration = 0.75  
    for i in range(1, steps):
        u = np.random.uniform(0, 1)
        delta_theta = 2 * np.arctan((1 - r_concentration) / (1 + r_concentration) * np.tan(np.pi * (u - 0.5)))
        theta_crw[i] = theta_crw[i-1] + delta_theta
    x_crw = np.concatenate(([0], np.cumsum(l_crw * np.cos(theta_crw))))
    y_crw = np.concatenate(([0], np.cumsum(l_crw * np.sin(theta_crw))))

    # --- 3. Self-Avoiding Walk with Memory 1 (SAW-1) ---
    x_saw, y_saw = [0.0], [0.0]
    current_theta = np.random.uniform(-np.pi, np.pi)
    for i in range(steps):
        valid_step = False
        for _ in range(100):
            l_step = np.random.exponential(scale=1.0)
            if i > 0:
                delta_theta = np.random.uniform(-2.2, 2.2)
                candidate_theta = current_theta + delta_theta
            else:
                candidate_theta = current_theta
            cand_x = x_saw[-1] + l_step * np.cos(candidate_theta)
            cand_y = y_saw[-1] + l_step * np.sin(candidate_theta)
            
            too_close = False
            for hx, hy in zip(x_saw, y_saw):
                if np.hypot(cand_x - hx, cand_y - hy) < saw_radius:
                    too_close = True
                    break
            if not too_close:
                x_saw.append(cand_x)
                y_saw.append(cand_y)
                current_theta = candidate_theta
                valid_step = True
                break
        if not valid_step:
            x_saw.append(cand_x)
            y_saw.append(cand_y)
            current_theta = candidate_theta
    x_saw1, y_saw1 = np.array(x_saw), np.array(y_saw)

    # --- 4. Lévy Walk ---
    l_levy = np.random.pareto(1.2, steps) * 0.5
    theta_levy = np.random.uniform(-np.pi, np.pi, steps)
    x_levy = np.concatenate(([0], np.cumsum(l_levy * np.cos(theta_levy))))
    y_levy = np.concatenate(([0], np.cumsum(l_levy * np.sin(theta_levy))))

    # --- 5. Markov Final (Orthogonal Exploration) ---
    x_max_bounds = 5.0
    y_max_bounds = np.pi
    _pLL, _pLT, _pTL, _pTT = 0.65, 0.35, 0.74, 0.26
    
    mag_L = np.clip(np.random.normal(loc=0.967, scale=0.3, size=2000), 0.01, None)
    mag_T = np.clip(np.random.normal(loc=0.864, scale=0.3, size=2000), 0.01, None)
    
    cur_x, cur_y = 0.0, 0.0
    x_m, y_m = [cur_x], [cur_y]
    cur_type = 'L' if np.random.random() < _pLL else 'T'
    sign_T = np.random.choice([-1, 1])
    
    for _ in range(steps):
        if cur_type == 'L':
            dx_mag = np.random.choice(mag_L)
            # Tiered positional spatial bias towards center (x=0)
            if abs(cur_x) < x_max_bounds / 3.0:
                prob_towards = 0.5614
            elif abs(cur_x) < 2.0 * x_max_bounds / 3.0:
                prob_towards = 0.5670
            else:
                prob_towards = 0.8421
                
            towards = (np.random.random() < prob_towards)
            if cur_x > 0:
                dx_sign = -1 if towards else 1
            elif cur_x < 0:
                dx_sign = 1 if towards else -1
            else:
                dx_sign = np.random.choice([-1, 1])
                
            next_x = cur_x + dx_mag * dx_sign
            # Reflective boundary conditions
            if next_x > x_max_bounds:
                cur_x = x_max_bounds - (next_x - x_max_bounds)
                cur_y = -cur_y 
            elif next_x < -x_max_bounds:
                cur_x = -x_max_bounds + (-x_max_bounds - next_x)
                cur_y = -cur_y
            else:
                cur_x = next_x
        else: # 'T' State (Vertical movement)
            dy_mag = np.random.choice(mag_T)
            cur_y = cur_y + dy_mag * sign_T
            # Wrap-around boundary layout for the transversal simulation axis
            if cur_y > y_max_bounds:   cur_y -= 2 * y_max_bounds
            elif cur_y < -y_max_bounds: cur_y += 2 * y_max_bounds
            
        x_m.append(cur_x)
        y_m.append(cur_y)
        
        # Hidden Markov state transitions
        if cur_type == 'L':
            if np.random.random() < _pLT:
                cur_type = 'T'
                sign_T = np.random.choice([-1, 1]) # Choose direction for new T bout
        else:
            if np.random.random() < _pTL:
                cur_type = 'L'
                
    x_markov, y_markov = np.array(x_m), np.array(y_m)

    # --- Plot Layout Rendering ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    
    walks = [
        (x_urw, y_urw, 'Uncorrelated Random Walk (URW)', '#1f77b4', '(A)'),
        (x_crw, y_crw, 'Correlated Random Walk (CRW)', '#2ca02c', '(B)'),
        (x_saw1, y_saw1, f'Self-Avoiding Walk (M=1, r={saw_radius})', '#ff7f0e', '(C)'),
        (x_levy, y_levy, 'Lévy Walk', '#d62728', '(D)'),
        (x_markov, y_markov, 'Markov Final (Orthogonal)', '#9467bd', '(E)')
    ]
    
    # Pre-calculate global maximum extent from barycenter to ensure all plots share the exact same scale
    max_extent = 0
    for x, y, _, _, _ in walks:
        bx, by = np.mean(x), np.mean(y)
        max_extent = max(max_extent, np.max(np.abs(x - bx)))
        max_extent = max(max_extent, np.max(np.abs(y - by)))
        
    for idx, (x, y, title, color, panel) in enumerate(walks):
        row, col = idx // 3, idx % 3
        ax = axs[row, col]
        
        ax.plot(x, y, color=color, linewidth=2, alpha=0.85, zorder=2)
        ax.scatter(x[1:-1], y[1:-1], color=color, s=20, alpha=0.6, zorder=3)
        
        if panel == '(C)' and saw_radius > 0:
            for hx, hy in zip(x[:-1], y[:-1]):
                circle = plt.Circle((hx, hy), saw_radius, color=color, fill=True, alpha=0.05, zorder=1)
                ax.add_patch(circle)
        
        ax.scatter(x[0], y[0], color='#2b2b2b', marker='o', s=70, edgecolors='black', zorder=5)
        ax.scatter(x[-1], y[-1], color='#d62728', marker='X', s=80, edgecolors='black', zorder=5)
        
        y_padding = max_extent * 0.08
        ax.text(x[0], y[0] + y_padding, 'Start', fontsize=9, fontweight='bold', ha='center', zorder=6)
        ax.text(x[-1], y[-1] + y_padding, 'End', fontsize=9, fontweight='bold', ha='center', zorder=6)
        
        ax.set_title(f"{panel} {title}", fontsize=12, fontweight='bold', pad=10, loc='left')
        ax.grid(False) # Removed grid as requested
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
        ax.set_aspect('equal', 'datalim')
        
        # Center trajectory based on its barycenter and apply the unified global scale
        bx, by = np.mean(x), np.mean(y)
        ax.set_xlim(bx - max_extent * 1.1, bx + max_extent * 1.1)
        ax.set_ylim(by - max_extent * 1.1, by + max_extent * 1.1)

    # Clean removal of the extra 6th subpanel space
    axs[1, 2].axis('off')
    plt.tight_layout()
    
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', transparent=True)
        print(f'Saved to {output_file}')
        png_file = output_file.replace('.svg', '.png')
        plt.savefig(png_file, bbox_inches='tight', dpi=300)
    plt.show()






def plot_multi_strategy_hole_size_sweep(output_file=None, num_simulations=10000, target_std=0.3, n_steps = 4):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    hole_diameters = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    _a_p, _b_p = 1.2602, 1.3749
    _z_max_p = np.sqrt(_a_p**2 + _b_p**2)
    
    _pLL, _pTT = 0.65, 0.26
    
    mag_L = np.random.normal(loc=0.967, scale=target_std, size=100000)
    mag_L = np.clip(mag_L, 0.01, None)
    
    mag_T = np.random.normal(loc=0.864, scale=target_std, size=100000)
    mag_T = np.clip(mag_T, 0.01, None)
    
    pareto_shape = 2.0
    
    def _apply_z_move(z, th, dz, z_max):
        limit = z_max * 0.999
        next_z = z + dz
        if next_z > limit:
            return limit - (next_z - limit), (th + np.pi) % (2 * np.pi)
        elif next_z < -limit:
            return -limit + (-limit - next_z), (th + np.pi) % (2 * np.pi)
        return next_z, th

    def _peanut_surface_point(z, th, a, b):
        t1 = np.sqrt(b**4 + 4*a**2*z**2)
        r2 = t1 - z**2 - a**2
        r = np.sqrt(max(0.0, r2))
        return np.array([r*np.cos(th), r*np.sin(th), z])
        
    def _dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    z_lin = np.linspace(-_z_max_p * 0.999, _z_max_p * 0.999, 100)
    th_lin = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    Zm, Thm = np.meshgrid(z_lin, th_lin, indexing='ij')
    
    def peanut_radius_sim(z, a, b):
        t1 = np.sqrt(b**4 + 4*a**2*z**2)
        r2 = t1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))
        
    Rm = peanut_radius_sim(Zm, _a_p, _b_p)
    Xm = Rm * np.cos(Thm)
    Ym = Rm * np.sin(Thm)

    faces = []
    face_areas = []
    rows, cols = Xm.shape
    for r in range(rows - 1):
        for c in range(cols):
            p1 = np.array([Xm[r, c], Ym[r, c], Zm[r, c]])
            p2 = np.array([Xm[r+1, c], Ym[r+1, c], Zm[r+1, c]])
            nxt_c = (c + 1) % cols
            p3 = np.array([Xm[r+1, nxt_c], Ym[r+1, nxt_c], Zm[r+1, nxt_c]])
            p4 = np.array([Xm[r, nxt_c], Ym[r, nxt_c], Zm[r, nxt_c]])
            c1 = (np.linalg.norm(np.cross(p2-p1, p3-p1)) + np.linalg.norm(np.cross(p3-p1, p4-p1))) / 2.0
            faces.append((p1 + p2 + p3 + p4) / 4.0)
            face_areas.append(c1)

    centroids = np.array(faces)
    face_areas = np.array(face_areas)
    face_probs = face_areas / np.sum(face_areas)

    hole_center = _peanut_surface_point(0, 0, _a_p, _b_p)
    
    results = {
        'Markov Final': [],
        'Correlated Random Walk': [],
        'Self-Avoiding Walk (M=2)': [],
        'Uncorrelated Walk': [],
        'Lévy Walk': []
    }
    
    print(f"Sweeping for {n_steps*2} clamps (n_steps={n_steps}). Num sims: {num_simulations}")
    for D in hole_diameters:
        hole_radius = D / 2.0
        counts = {k: 0 for k in results.keys()}
        
        for _ in range(num_simulations):
            idx = np.random.choice(len(centroids), p=face_probs)
            start_pt = centroids[idx]
            z0 = start_pt[2]
            th0 = np.arctan2(start_pt[1], start_pt[0])
            
            p0 = _peanut_surface_point(z0, th0, _a_p, _b_p)
            p0_f = _peanut_surface_point(z0, (th0 + np.pi)%(2*np.pi), _a_p, _b_p)

            # --- Markov Final ---
            touched = False
            if _dist(p0, hole_center) <= hole_radius or _dist(p0_f, hole_center) <= hole_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                cur_type = np.random.choice(['L', 'T'], p=[0.68, 0.32])
                for step in range(1, n_steps):
                    if cur_type == 'L':
                        dz_mag = np.random.choice(mag_L)
                        if abs(cur_z) < _z_max_p / 3.0: prob_towards = 0.5614
                        elif abs(cur_z) < 2.0 * _z_max_p / 3.0: prob_towards = 0.5670
                        else: prob_towards = 0.8421
                        towards = True if np.random.random() < prob_towards else False
                        if cur_z > 0: dz_sign = -1 if towards else 1
                        elif cur_z < 0: dz_sign = 1 if towards else -1
                        else: dz_sign = np.random.choice([-1, 1])
                        cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz_mag * dz_sign, _z_max_p)
                        cur_type = 'L' if np.random.random() < _pLL else 'T'
                    else:
                        darc = np.random.choice(mag_T) * np.random.choice([-1, 1])
                        r_cur = max(1e-6, peanut_radius_sim(cur_z, _a_p, _b_p))
                        cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                        cur_type = 'T' if np.random.random() < _pTT else 'L'

                    pt_b = _peanut_surface_point(cur_z, cur_th, _a_p, _b_p)
                    pt_f = _peanut_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi), _a_p, _b_p)
                    if _dist(pt_b, hole_center) <= hole_radius or _dist(pt_f, hole_center) <= hole_radius:
                        touched = True
                        break
            if touched: counts['Markov Final'] += 1

            # --- CRW ---
            touched = False
            if _dist(p0, hole_center) <= hole_radius or _dist(p0_f, hole_center) <= hole_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                direction = np.random.uniform(0, 2*np.pi)
                for step in range(1, n_steps):
                    step_len = np.random.exponential(scale=1.0)
                    c = 0.735
                    turn_angle = 2 * np.arctan((1 - c) / (1 + c) * np.tan(np.pi * (np.random.uniform(0, 1) - 0.5)))
                    direction = (direction + turn_angle) % (2*np.pi)
                    dz = step_len * np.cos(direction)
                    darc = step_len * np.sin(direction)
                    cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz, _z_max_p)
                    r_cur = max(1e-6, peanut_radius_sim(cur_z, _a_p, _b_p))
                    cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                    
                    pt_b = _peanut_surface_point(cur_z, cur_th, _a_p, _b_p)
                    pt_f = _peanut_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi), _a_p, _b_p)
                    if _dist(pt_b, hole_center) <= hole_radius or _dist(pt_f, hole_center) <= hole_radius:
                        touched = True
                        break
            if touched: counts['Correlated Random Walk'] += 1

            # --- SAW (M=2) ---
            touched = False
            if _dist(p0, hole_center) <= hole_radius or _dist(p0_f, hole_center) <= hole_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                history = [(z0, th0)]
                for step in range(1, n_steps):
                    valid_step = False
                    for _tries in range(50):
                        step_len = np.random.exponential(scale=1.0)
                        direction = np.random.uniform(0, 2*np.pi)
                        dz = step_len * np.cos(direction)
                        darc = step_len * np.sin(direction)
                        cand_z, cand_th = _apply_z_move(cur_z, cur_th, dz, _z_max_p)
                        cand_pt = _peanut_surface_point(cand_z, cand_th, _a_p, _b_p)
                        
                        too_close = False
                        for (hz, hth) in history:
                            hpt = _peanut_surface_point(hz, hth, _a_p, _b_p)
                            if _dist(cand_pt, hpt) < 0.5:
                                too_close = True
                                break
                        
                        if not too_close:
                            cur_z, cur_th = cand_z, cand_th
                            valid_step = True
                            break
                    if not valid_step:
                        cur_z, cur_th = cand_z, cand_th
                        
                    history.append((cur_z, cur_th))
                    if len(history) > 2:
                        history.pop(0)

                    r_cur = max(1e-6, peanut_radius_sim(cur_z, _a_p, _b_p))
                    pt_b = _peanut_surface_point(cur_z, cur_th, _a_p, _b_p)
                    pt_f = _peanut_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi), _a_p, _b_p)
                    if _dist(pt_b, hole_center) <= hole_radius or _dist(pt_f, hole_center) <= hole_radius:
                        touched = True
                        break
            if touched: counts['Self-Avoiding Walk (M=2)'] += 1

            # --- URW ---
            touched = False
            if _dist(p0, hole_center) <= hole_radius or _dist(p0_f, hole_center) <= hole_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                for step in range(1, n_steps):
                    step_len = np.random.exponential(scale=1.0)
                    direction = np.random.uniform(0, 2*np.pi)
                    dz = step_len * np.cos(direction)
                    darc = step_len * np.sin(direction)
                    cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz, _z_max_p)
                    r_cur = max(1e-6, peanut_radius_sim(cur_z, _a_p, _b_p))
                    cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                    
                    pt_b = _peanut_surface_point(cur_z, cur_th, _a_p, _b_p)
                    pt_f = _peanut_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi), _a_p, _b_p)
                    if _dist(pt_b, hole_center) <= hole_radius or _dist(pt_f, hole_center) <= hole_radius:
                        touched = True
                        break
            if touched: counts['Uncorrelated Walk'] += 1

            # --- Levy ---
            touched = False
            if _dist(p0, hole_center) <= hole_radius or _dist(p0_f, hole_center) <= hole_radius:
                touched = True
            else:
                cur_z, cur_th = z0, th0
                for step in range(1, n_steps):
                    step_len = (np.random.pareto(pareto_shape) + 1) * 0.5
                    direction = np.random.uniform(0, 2*np.pi)
                    dz = step_len * np.cos(direction)
                    darc = step_len * np.sin(direction)
                    cur_z, cur_th = _apply_z_move(cur_z, cur_th, dz, _z_max_p)
                    r_cur = max(1e-6, peanut_radius_sim(cur_z, _a_p, _b_p))
                    cur_th = (cur_th + darc/r_cur) % (2*np.pi)
                    
                    pt_b = _peanut_surface_point(cur_z, cur_th, _a_p, _b_p)
                    pt_f = _peanut_surface_point(cur_z, (cur_th + np.pi)%(2*np.pi), _a_p, _b_p)
                    if _dist(pt_b, hole_center) <= hole_radius or _dist(pt_f, hole_center) <= hole_radius:
                        touched = True
                        break
            if touched: counts['Lévy Walk'] += 1
            
        for k in results.keys():
            results[k].append(counts[k] / num_simulations * 100.0)
        print(f"D={D:.1f}cm -> " + ", ".join([f"{k}: {results[k][-1]:.1f}%" for k in results.keys()]))
        
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    empirical_probs = [4.0, 8.0, 22.0, 37.0, 44.0, 82.0]
    
    palette = {
        'Markov Final': '#EF4444',
        'Correlated Random Walk': '#1D4ED8',
        'Self-Avoiding Walk (M=2)': '#10B981',
        'Lévy Walk': '#FDE047',
        'Uncorrelated Walk': '#EC4899'
    }
    
    for k in results.keys():
        ax.plot(hole_diameters, results[k], marker='o', linestyle='--', linewidth=2, color=palette[k], label=k)
    
    ax.plot(hole_diameters, empirical_probs, marker='s', linestyle='-', linewidth=3, color='#475569', label='Real Agouti (Empirical)')
    
    ax.set_xlabel("Hole Diameter (cm)", fontsize=14)
    ax.set_ylabel(f"Probability of Contact after {n_steps*2} clamps (%)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, frameon=False, loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='both', width=2, length=6, labelsize=14)
    ax.set_title(f"Strategy Comparison across Hole Sizes ({n_steps} steps)", fontsize=16, pad=15)
    
    plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        ext = os.path.splitext(output_file)[1].lower()
        alt = output_file[:-len(ext)] + (".png" if ext == ".svg" else ".svg")
        kw  = {"dpi": 300} if ext == ".svg" else {}
        plt.savefig(alt, bbox_inches='tight', **kw)
        print(f"Saved comparison plot to {output_file} and {alt}")
    plt.show()




def plot_patchiness_comparison_two_faces(df_sim, output_file=None, max_steps=20):
    """
    Plots the footprint patchiness of all 5 simulation strategies
    on the custom Two-Faces projection map and prints their coverage stats.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    import os

    a_true, b_true = 1.2602, 1.3749
    z_max_true = np.sqrt(a_true**2 + b_true**2)
    num_pts = 300
    z_vals = np.linspace(-z_max_true + 1e-4, z_max_true - 1e-4, num_pts)
    th_vals = np.linspace(-np.pi, np.pi, num_pts)
    Z, TH = np.meshgrid(z_vals, th_vals)
    R_sq = b_true**4 + 4*a_true**2*Z**2
    R = np.sqrt(np.maximum(0, np.sqrt(R_sq) - Z**2 - a_true**2))
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    tree = cKDTree(pts)

    fig, axes = plt.subplots(1, 5, figsize=(35, 8))
    strategies = [
        ('Self-Avoiding Walk (M=1)', '#00CC96', f'SAW ({max_steps} steps)'), 
        ('Correlated Random Walk', '#EF553B', f'Correlated RW ({max_steps} steps)'),
        ('Uncorrelated Walk', '#FFA15A', f'Uncorrelated Walk ({max_steps} steps)'),
        ('Levy Walk', '#AB63FA', f'Levy Walk ({max_steps} steps)'),
        ('Markov Final', '#636EFA', f'Markov (Agouti) ({max_steps} steps)')
    ]

    S = 1.85
    faces_map = {0: {'az': 0, 'cx': -S/1.5, 'cy': 0},
                 1: {'az': 180, 'cx': S/1.5, 'cy': 0}}
                 
    def peanut_radius_local(z, a=a_true, b=b_true):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    width_vals = peanut_radius_local(z_vals, a_true, b_true) * (np.pi / 2)

    def map_to_canvas(x, y, z):
        angle = np.degrees(np.arctan2(y, x)) % 360
        best_diff = 360; best_fid = -1
        for fid, f in faces_map.items():
            diff = abs(angle - f['az'])
            if diff > 180: diff = 360 - diff
            if diff < best_diff: best_diff = diff; best_fid = fid
        f = faces_map[best_fid]
        raw_diff = angle - f['az']
        if raw_diff > 180: raw_diff -= 360
        elif raw_diff < -180: raw_diff += 360
        delta_rad = np.radians(raw_diff)
        curr_r = np.sqrt(x**2 + y**2)
        arc_dist = curr_r * delta_rad
        return f['cx'] + arc_dist, f['cy'] + z

    mapped_pts = np.array([map_to_canvas(x, y, z) for x, y, z in pts])

    from from_antigravity_peanuts import get_geodesic_profile_distance
    s_vals_line = np.array([get_geodesic_profile_distance(0, z, a_true, b_true) for z in z_vals])
    s_vals_line = s_vals_line - s_vals_line[0]

    pts_z = Z.ravel()
    pts_th = TH.ravel()
    pts_r = R.ravel()
    pts_s = np.interp(pts_z, z_vals, s_vals_line)

    with np.errstate(divide='ignore', invalid='ignore'):
        dr_dz = (2*a_true**2*Z/np.sqrt(R_sq) - Z) / R
        dr_dz[np.isnan(dr_dz)] = 0.0
        
    dA = R * np.sqrt(1 + dr_dz**2) * (z_vals[1] - z_vals[0]) * (th_vals[1] - th_vals[0])
    total_area = np.sum(dA)

    for ax_2d, (strat, color, title) in zip(axes, strategies):
        for fid, f in faces_map.items():
            cx, cy = f['cx'], f['cy']
            ax_2d.plot(cx + width_vals, cy + z_vals, 'k-')
            ax_2d.plot(cx - width_vals, cy + z_vals, 'k-')
            ax_2d.fill_betweenx(cy + z_vals, cx - width_vals, cx + width_vals, color='#a6806d', alpha=0.5)
            label = "Front (0 deg)" if fid == 0 else "Back (180 deg)"
            ax_2d.text(cx, cy + z_max_true + 0.3, label, ha='center', fontweight='bold', fontsize=16)

        try:
            strat_data = df_sim[df_sim['strategy'] == strat]['coords'].values[0]
        except IndexError:
            print(f"Strategy {strat} not found in df_sim")
            continue
            
        p_all = np.array(strat_data)[:max_steps*2]
        fronts = p_all[1::2]

        covered_mask = np.zeros(len(pts), dtype=bool)
        covered_count = np.zeros(len(pts), dtype=int)
        
        for p_clamp in fronts:
            xc, yc, zc = p_clamp
            c_z_idx = np.argmin(np.abs(z_vals - zc))
            c_S = s_vals_line[c_z_idx]
            c_th = np.arctan2(yc, xc)
            
            delta_s = np.abs(pts_s - c_S)
            diff_th = np.abs(pts_th - c_th)
            diff_th[diff_th > np.pi] = 2*np.pi - diff_th[diff_th > np.pi]
            delta_arc = pts_r * diff_th
            
            clamp_mask = (delta_s <= 0.35) & (delta_arc <= 0.20)
            covered_mask[clamp_mask] = True
            covered_count[clamp_mask] += 1
            
        cov_pts = mapped_pts[covered_mask]
        ax_2d.scatter(cov_pts[:,0], cov_pts[:,1], color=color, s=2, alpha=0.8)
        
        mapped_fronts = np.array([map_to_canvas(x, y, z) for x, y, z in fronts])
        
        for i in range(len(fronts)-1):
            x1, y1, z1 = fronts[i]
            x2, y2, z2 = fronts[i+1]
            
            idx1 = np.argmin(np.abs(z_vals - z1))
            idx2 = np.argmin(np.abs(z_vals - z2))
            s1, s2 = s_vals_line[idx1], s_vals_line[idx2]
            
            th1 = np.arctan2(y1, x1)
            th2 = np.arctan2(y2, x2)
            
            if th2 - th1 > np.pi: th1 += 2*np.pi
            elif th1 - th2 > np.pi: th2 += 2*np.pi
            
            num_interp = 20
            s_interp = np.linspace(s1, s2, num_interp)
            th_interp = np.linspace(th1, th2, num_interp)
            
            z_interp = np.interp(s_interp, s_vals_line, z_vals)
            R_sq_i = b_true**4 + 4*a_true**2*z_interp**2
            R_interp = np.sqrt(np.maximum(0, np.sqrt(R_sq_i) - z_interp**2 - a_true**2))
            x_interp = R_interp * np.cos(th_interp)
            y_interp = R_interp * np.sin(th_interp)
            
            pts_interp = np.column_stack((x_interp, y_interp, z_interp))
            mapped_interp = np.array([map_to_canvas(x, y, z) for x, y, z in pts_interp])
            
            for j in range(num_interp - 1):
                pA = mapped_interp[j]
                pB = mapped_interp[j+1]
                if np.linalg.norm(pA - pB) < 1.0: # avoid face-jumping lines
                    ax_2d.plot([pA[0], pB[0]], [pA[1], pB[1]], color='black', linestyle='-', linewidth=2, alpha=0.6)
                
        ax_2d.scatter(mapped_fronts[:,0], mapped_fronts[:,1], color='black', s=20, zorder=10)
        
        covered_area = np.sum(dA.ravel()[covered_mask])
        overlap_area = np.sum(dA.ravel()[covered_count >= 2])
        
        total_dist = 0.0
        for i in range(1, len(fronts)):
            x1, y1, z1 = fronts[i-1]
            x2, y2, z2 = fronts[i]
            
            idx1 = np.argmin(np.abs(z_vals - z1))
            idx2 = np.argmin(np.abs(z_vals - z2))
            s1, s2 = s_vals_line[idx1], s_vals_line[idx2]
            ds = s1 - s2
            
            th1 = np.arctan2(y1, x1)
            th2 = np.arctan2(y2, x2)
            dth = np.abs(th1 - th2)
            if dth > np.pi: dth = 2*np.pi - dth
            
            r1 = np.sqrt(x1**2 + y1**2)
            r2 = np.sqrt(x2**2 + y2**2)
            r_avg = (r1 + r2) / 2.0
            darc = dth * r_avg
            
            total_dist += np.sqrt(ds**2 + darc**2)
        
        print(f"\n--- {strat} ---")
        print(f"Total Peanut Surface Area: {total_area:.2f} cm^2")
        print(f"Total Geodesic Distance Traveled: {total_dist:.2f} cm")
        print(f"Area Covered: {covered_area:.2f} cm^2 ({(covered_area/total_area)*100:.1f}%)")
        print(f"Area Overlapped: {overlap_area:.2f} cm^2 ({(overlap_area/total_area)*100:.1f}%)")

        ax_2d.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax_2d.set_aspect('equal')
        ax_2d.set_xlim(-S*1.5, S*1.5)
        ax_2d.set_ylim(-2.5, 2.5)
        ax_2d.axis('off')

    plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print('Saved patchiness comparison to', output_file)
    plt.show()


def plot_real_peanut_and_simulation(
    sites_path,
    output_file=None,
    sampling_strategy='random',
    clamp_dims=(0.7, 0.3),
    show_bottom=True,
    show_front=False,
    show=True,
    target_std=None
):
    """
    Plots the real peanut surface and clamp sequence alongside a simulated clamp sequence.
    Left: Real clamp sequence.
    Right: Example simulation (random or markov) starting from the first actual clamp.
    """
    import os
    import json
    import glob
    import numpy as np

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: 'plotly' package is required for interactive 3D plots. Please install it with 'pip install plotly'.")
        return None

    # 1. Load Real Clamp Data
    if not os.path.exists(sites_path):
        print(f"Sites file not found: {sites_path}")
        return None

    sites_bottom = []
    json_version = "v2"
    with open(sites_path, 'r') as f:
        data = json.load(f)
        sites_bottom = data if isinstance(data, list) else data.get("sites", [])
        if isinstance(data, dict):
            json_version = data.get("version", "v2")

    sites_front = []
    path_base = sites_path.replace("_sites.json", "")
    front_path = path_base + "_front_sites.json"
    if os.path.exists(front_path):
        with open(front_path, 'r') as f:
            fd = json.load(f)
            sites_front = fd if isinstance(fd, list) else fd.get("sites", [])

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

    valid_bottom = parse_sites(sites_bottom) if show_bottom else []
    valid_front = parse_sites(sites_front) if show_front else []

    if not valid_bottom and not valid_front:
        print("No valid sites found.")
        return None

    # Scaling
    A_PARAM, B_PARAM = 1.2602, 1.3749
    is_hole = 'hole' in os.path.basename(sites_path).lower() or 'hole' in os.path.dirname(sites_path).lower()
    a_orig, b_orig = (1.0, 1.1) if is_hole else (1.0, 1.07)
    z_max_orig = np.sqrt(a_orig**2 + b_orig**2)
    z_max_true = np.sqrt(A_PARAM**2 + B_PARAM**2)
    scale_factor = z_max_true / z_max_orig

    def peanut_radius_sim(z, a, b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    # Hole Position
    if json_version == "v3":
        p_hole = np.array([0.0, 0.0, z_max_true])
    else:
        r_hole_orig = peanut_radius_sim(0.75, a_orig, b_orig)
        p_hole_orig = np.array([0.0, r_hole_orig, 0.75])
        p_hole = p_hole_orig * scale_factor

    z_h = p_hole[2]
    theta_h = np.arctan2(p_hole[1], p_hole[0])
    overlap_threshold = 0.50

    def get_distance_to_hole(p):
        z_c = p[2]
        theta_c = np.arctan2(p[1], p[0])
        dy = z_c - z_h
        dx_angle = (theta_c - theta_h + np.pi) % (2 * np.pi) - np.pi
        r_c = peanut_radius_sim(z_c, A_PARAM, B_PARAM)
        dx = r_c * dx_angle
        dist_x = max(0.0, abs(dx) - clamp_dims[1] / 2.0)
        dist_y = max(0.0, abs(dy) - clamp_dims[0] / 2.0)
        return np.sqrt(dist_x**2 + dist_y**2)

    def get_clamp_rectangle_3d_points(p, W=clamp_dims[1], L=clamp_dims[0]):
        z_c = p[2]
        theta_c = np.arctan2(p[1], p[0])
        boundary_pts = []
        z_val = z_c - L/2
        z_val = np.clip(z_val, -z_max_true + 0.01, z_max_true - 0.01)
        r_val = max(0.01, peanut_radius_sim(z_val, A_PARAM, B_PARAM))
        d_theta = (W / 2.0) / r_val
        for t in np.linspace(theta_c - d_theta, theta_c + d_theta, 5):
            boundary_pts.append([r_val * np.cos(t), r_val * np.sin(t), z_val])
        z_val = z_c + L/2
        z_val = np.clip(z_val, -z_max_true + 0.01, z_max_true - 0.01)
        r_val = max(0.01, peanut_radius_sim(z_val, A_PARAM, B_PARAM))
        d_theta = (W / 2.0) / r_val
        for t in np.linspace(theta_c + d_theta, theta_c - d_theta, 5):
            boundary_pts.append([r_val * np.cos(t), r_val * np.sin(t), z_val])
        boundary_pts.append(boundary_pts[0])
        return np.array(boundary_pts)

    def scale_point(p):
        return np.array([p[0] * scale_factor, p[1] * scale_factor, p[2] * scale_factor])

    actual_bottom = [scale_point(s['p']) for s in valid_bottom]
    actual_front = [scale_point(s['p']) for s in valid_front]
    
    # Combined real clamps in order of frames
    all_real_valid = []
    for s in valid_bottom:
        all_real_valid.append((scale_point(s['p']), s['frame'], 'bottom'))
    for s in valid_front:
        all_real_valid.append((scale_point(s['p']), s['frame'], 'front'))
    all_real_valid.sort(key=lambda x: x[1])
    real_coords = [x[0] for x in all_real_valid]

    # First clamp & subsequent count
    start_pt = real_coords[0]
    n_subsequent = len(real_coords) - 1

    # 2. Run Simulation
    sim_coords = [start_pt]
    if n_subsequent > 0:
        if sampling_strategy in ['markov', 'empirical_2d', 'markov_parametric']:
            _a_phys, _b_phys = 1.2602, 1.3749
            _z_max_phys = np.sqrt(_a_phys**2 + _b_phys**2)
            
            if sampling_strategy == 'markov_parametric':
                _pLL, _pLT = 0.65, 0.35
                _pTL, _pTT = 0.74, 0.26
                
                sd_L = target_std if target_std is not None else 0.560
                sd_T = target_std if target_std is not None else 0.432
                
                mag_L = np.random.normal(loc=0.767, scale=sd_L, size=10000)
                mag_L = np.clip(mag_L, 0.01, None)
                sign_L = np.random.choice([-1, 1], size=10000)
                _long_amps = mag_L * sign_L
                
                mag_T = np.random.normal(loc=0.664, scale=sd_T, size=10000)
                mag_T = np.clip(mag_T, 0.01, None)
                sign_T = np.random.choice([-1, 1], size=10000)
                _trans_amps = mag_T * sign_T
                
                _markov_params = dict(
                    prob_LL=_pLL, prob_LT=_pLT,
                    prob_TT=_pTT, prob_TL=_pTL,
                    long_amps=_long_amps,
                    trans_amps=_trans_amps,
                    all_2d_steps=[(0.3, 0.3)],
                    a_phys=_a_phys, b_phys=_b_phys, z_max_phys=_z_max_phys
                )
            else:
                # Run Markov param calculation from empirical data
                _sites_dir = os.path.dirname(sites_path)
                _bottom_files = sorted([
                    f for f in glob.glob(os.path.join(_sites_dir, "*_sites.json"))
                    if not f.endswith("_front_sites.json")
                ])
                def _pr_markov(z, a=1.0, b=1.1):
                    t1 = np.sqrt(b**4 + 4*a**2*z**2)
                    return np.sqrt(np.maximum(0, t1 - z**2 - a**2))
                _scale_m = _z_max_phys / np.sqrt(1.0**2 + 1.1**2)
                _trans = {'LL': 0, 'LT': 0, 'TL': 0, 'TT': 0}
                _counts = {'L': 0, 'T': 0}
                _long_amps, _trans_amps = [], []
                for _bf in _bottom_files:
                    for _gpath in [_bf, _bf.replace('_sites.json', '_front_sites.json')]:
                        if not os.path.exists(_gpath): continue
                        try:
                            with open(_gpath) as _fh:
                                _gd = json.load(_fh)
                                _graw = _gd if isinstance(_gd, list) else _gd.get('sites', [])
                                _gch = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3])}
                                        for s in _graw if len(s) >= 4]
                            _gch.sort(key=lambda x: x['frame'])
                            _gseq = []
                            for _gi in range(len(_gch) - 1):
                                _gp1, _gp2 = _gch[_gi]['p'], _gch[_gi+1]['p']
                                _gdz_signed = get_geodesic_profile_distance(_gp1[2], _gp2[2], a=1.0, b=1.1)
                                _gaz   = (_gp1[2] + _gp2[2]) / 2
                                _gr    = _pr_markov(_gaz)
                                _ga1   = np.arctan2(_gp1[1], _gp1[0])
                                _ga2   = np.arctan2(_gp2[1], _gp2[0])
                                _gdarc_signed = _gr * (((_ga2 - _ga1 + np.pi) % (2 * np.pi)) - np.pi)
                                _gseq.append(('L', _gdz_signed, _gdarc_signed) if abs(_gdz_signed) >= abs(_gdarc_signed) else ('T', _gdz_signed, _gdarc_signed))
                            for _gi, (_gm, _gdz_val, _gdarc_val) in enumerate(_gseq):
                                if _gm == 'L': _long_amps.append(_gdz_val * _scale_m)
                                else:          _trans_amps.append(_gdarc_val * _scale_m)
                                if _gi < len(_gseq) - 1:
                                    _gpair = _gm + _gseq[_gi+1][0]
                                    _trans[_gpair] += 1
                                    _counts[_gm]   += 1
                        except: pass
                        
                # Also collect exact 2D steps for empirical_2d sampling
                _all_2d_steps = []
                for _bf in _bottom_files:
                    for _gpath in [_bf, _bf.replace('_sites.json', '_front_sites.json')]:
                        if not os.path.exists(_gpath): continue
                        try:
                            with open(_gpath) as _fh:
                                _gd = json.load(_fh)
                                _graw = _gd if isinstance(_gd, list) else _gd.get('sites', [])
                                _gch = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3])}
                                        for s in _graw if len(s) >= 4]
                            _gch.sort(key=lambda x: x['frame'])
                            for _gi in range(len(_gch) - 1):
                                _gp1, _gp2 = _gch[_gi]['p'], _gch[_gi+1]['p']
                                _gdz = get_geodesic_profile_distance(_gp1[2], _gp2[2], a=1.0, b=1.1)
                                _gaz = (_gp1[2] + _gp2[2]) / 2
                                _gr = _pr_markov(_gaz)
                                _ga1 = np.arctan2(_gp1[1], _gp1[0])
                                _ga2 = np.arctan2(_gp2[1], _gp2[0])
                                _gdarc = _gr * (((_ga2 - _ga1 + np.pi) % (2 * np.pi)) - np.pi)
                                _all_2d_steps.append((_gdz * _scale_m, _gdarc * _scale_m))
                        except: pass
                if not _all_2d_steps: _all_2d_steps = [(0.3, 0.3)]
                _long_amps = np.array(_long_amps) if len(_long_amps) > 0 else np.array([0.3])
                _trans_amps = np.array(_trans_amps) if len(_trans_amps) > 0 else np.array([0.3])
                
                if target_std is not None:
                    if len(_long_amps) > 1:
                        mu_l = np.mean(_long_amps)
                        sd_l = np.std(_long_amps)
                        if sd_l > 1e-6:
                            _long_amps = mu_l + (_long_amps - mu_l) * (target_std / sd_l)
                            _long_amps = np.clip(_long_amps, 0.01, None)
                    if len(_trans_amps) > 1:
                        mu_t = np.mean(_trans_amps)
                        sd_t = np.std(_trans_amps)
                        if sd_t > 1e-6:
                            _trans_amps = mu_t + (_trans_amps - mu_t) * (target_std / sd_t)
                            _trans_amps = np.clip(_trans_amps, 0.01, None)
                
                _pLL = _trans['LL'] / _counts['L'] if _counts['L'] > 0 else 0.5
                _pTT = _trans['TT'] / _counts['T'] if _counts['T'] > 0 else 0.5
                _markov_params = dict(
                    prob_LL=_pLL, prob_LT=1-_pLL,
                    prob_TT=_pTT, prob_TL=1-_pTT,
                    long_amps=np.array(_long_amps) if len(_long_amps) > 0 else np.array([0.3]),
                    trans_amps=np.array(_trans_amps) if len(_trans_amps) > 0 else np.array([0.3]),
                    all_2d_steps=_all_2d_steps,
                    a_phys=_a_phys, b_phys=_b_phys, z_max_phys=_z_max_phys
                )

            def _apply_z_move(z, th, dz, z_max):
                limit = z_max * 0.999
                next_z = z + dz
                if next_z > limit:
                    return limit - (next_z - limit), (th + np.pi) % (2 * np.pi)
                elif next_z < -limit:
                    return -limit + (-limit - next_z), (th + np.pi) % (2 * np.pi)
                return next_z, th

            def _markov_run(start_pt, n_steps, mp):
                _a_p = mp['a_phys']; _b_p = mp['b_phys']; _z_max_p = mp['z_max_phys']
                def _surf(z, th):
                    t1 = np.sqrt(_b_p**4 + 4*_a_p**2*z**2)
                    r  = np.sqrt(max(0.0, t1 - z**2 - _a_p**2))
                    return np.array([r*np.cos(th), r*np.sin(th), z])
                _cur_z = np.clip(start_pt[2], -_z_max_p*0.999, _z_max_p*0.999)
                _cur_th = np.arctan2(start_pt[1], start_pt[0])
                _cur_type = 'L' if np.random.random() < mp['prob_LL'] else 'T'
                pts = []
                for _ in range(n_steps):
                    if sampling_strategy == 'empirical_2d':
                        _step_idx = np.random.randint(len(mp['all_2d_steps']))
                        _dz, _darc_signed = mp['all_2d_steps'][_step_idx]
                        _cur_z, _cur_th = _apply_z_move(_cur_z, _cur_th, _dz, _z_max_p)
                        t1 = np.sqrt(_b_p**4 + 4*_a_p**2*_cur_z**2)
                        _r = max(1e-6, np.sqrt(max(0.0, t1 - _cur_z**2 - _a_p**2)))
                        _cur_th = (_cur_th + (_darc_signed/_r)) % (2*np.pi)
                    else:
                        if _cur_type == 'L':
                            _dz = np.random.choice(mp['long_amps'])
                            _cur_z, _cur_th = _apply_z_move(_cur_z, _cur_th, _dz, _z_max_p)
                        else:
                            _darc_signed = np.random.choice(mp['trans_amps'])
                            t1 = np.sqrt(_b_p**4 + 4*_a_p**2*_cur_z**2)
                            _r = max(1e-6, np.sqrt(max(0.0, t1 - _cur_z**2 - _a_p**2)))
                            _cur_th = (_cur_th + (_darc_signed/_r)) % (2*np.pi)
                        _cur_type = ('L' if np.random.random()<mp['prob_LL'] else 'T') if _cur_type=='L' else \
                                    ('T' if np.random.random()<mp['prob_TT'] else 'L')
                    pts.append(_surf(_cur_z, _cur_th))
                return pts

            sim_coords += _markov_run(start_pt, n_subsequent, _markov_params)
        else:
            # Run Random Surface Sampling
            z_lin = np.linspace(-z_max_true * 0.999, z_max_true * 0.999, 100)
            theta_lin = np.linspace(0, 2*np.pi, 60, endpoint=False)
            Z_mesh_s, Theta_mesh_s = np.meshgrid(z_lin, theta_lin, indexing='ij')
            R_mesh_s = peanut_radius_sim(Z_mesh_s, A_PARAM, B_PARAM)
            X_mesh_s = R_mesh_s * np.cos(Theta_mesh_s)
            Y_mesh_s = R_mesh_s * np.sin(Theta_mesh_s)
            faces = []
            face_areas = []
            rows, cols = X_mesh_s.shape
            for r in range(rows - 1):
                for c in range(cols):
                    p1 = np.array([X_mesh_s[r, c], Y_mesh_s[r, c], Z_mesh_s[r, c]])
                    p2 = np.array([X_mesh_s[r+1, c], Y_mesh_s[r+1, c], Z_mesh_s[r+1, c]])
                    nxt_c = (c + 1) % cols
                    p3 = np.array([X_mesh_s[r+1, nxt_c], Y_mesh_s[r+1, nxt_c], Z_mesh_s[r+1, nxt_c]])
                    p4 = np.array([X_mesh_s[r, nxt_c], Y_mesh_s[r, nxt_c], Z_mesh_s[r, nxt_c]])
                    c1 = (np.linalg.norm(np.cross(p2-p1, p3-p1)) + np.linalg.norm(np.cross(p3-p1, p4-p1))) / 2.0
                    faces.append((p1 + p2 + p3 + p4) / 4.0)
                    face_areas.append(c1)
            faces = np.array(faces)
            face_areas = np.array(face_areas)
            face_probabilities = face_areas / np.sum(face_areas)
            
            sampled_indices = np.random.choice(len(faces), size=n_subsequent, p=face_probabilities)
            sim_coords += list(faces[sampled_indices])

    # 3. Create 3D Peanut Mesh for Visualization
    z_lin = np.linspace(-z_max_true * 0.999, z_max_true * 0.999, 50)
    theta_lin = np.linspace(0, 2*np.pi, 45, endpoint=False)
    Z_mesh, Theta_mesh = np.meshgrid(z_lin, theta_lin, indexing='ij')
    R_mesh = peanut_radius_sim(Z_mesh, A_PARAM, B_PARAM)
    X_mesh = R_mesh * np.cos(Theta_mesh)
    Y_mesh = R_mesh * np.sin(Theta_mesh)

    dists_mesh = np.sqrt((X_mesh - p_hole[0])**2 + (Y_mesh - p_hole[1])**2 + (Z_mesh - p_hole[2])**2)
    hole_mask = (dists_mesh <= overlap_threshold).astype(float)

    # 4. Interactive Plotly Figure Setup
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Real Clamp Path', f'Simulated Path ({sampling_strategy.upper()})'),
        horizontal_spacing=0.03
    )

    def draw_scene(row, col, coords, miss_color):
        # A. Peanut Surface Shell
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            surfacecolor=hole_mask,
            colorscale=[[0.0, '#C89C64'], [1.0, '#1E1E1E']],  # shell vs hole region
            showscale=False,
            opacity=0.75,
            hoverinfo='skip',
            name='Peanut Shell'
        ), row=row, col=col)

        # B. Hole Target Center Marker
        fig.add_trace(go.Scatter3d(
            x=[p_hole[0]], y=[p_hole[1]], z=[p_hole[2]],
            mode='markers',
            marker=dict(size=8, color='#FF3333', symbol='circle'),
            name='Hole Center',
            showlegend=False
        ), row=row, col=col)

        # C. Hole Target Ring Outline
        ring_th = np.linspace(0, 2*np.pi, 60)
        h_th_coord, h_z_coord = np.arctan2(p_hole[1], p_hole[0]), p_hole[2]
        dr_dz_h = (peanut_radius_sim(h_z_coord + 1e-5, A_PARAM, B_PARAM) - peanut_radius_sim(h_z_coord - 1e-5, A_PARAM, B_PARAM)) / 2e-5
        t1 = np.array([-np.sin(h_th_coord), np.cos(h_th_coord), 0])
        t2_r = np.array([dr_dz_h * np.cos(h_th_coord), dr_dz_h * np.sin(h_th_coord), 1.0])
        t2 = t2_r / np.linalg.norm(t2_r)
        ring_pts = p_hole + overlap_threshold * (np.outer(np.cos(ring_th), t1) + np.outer(np.sin(ring_th), t2))
        fig.add_trace(go.Scatter3d(
            x=ring_pts[:, 0], y=ring_pts[:, 1], z=ring_pts[:, 2],
            mode='lines',
            line=dict(color='#FF3333', width=4),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

        # D. Connecting line
        x_pts = [p[0] for p in coords]
        y_pts = [p[1] for p in coords]
        z_pts = [p[2] for p in coords]
        fig.add_trace(go.Scatter3d(
            x=x_pts, y=y_pts, z=z_pts,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.4)', width=3),
            showlegend=False
        ), row=row, col=col)

        # E. Clamps (Rectangles and Markers)
        colors = []
        for i, p in enumerate(coords):
            dist = get_distance_to_hole(p)
            is_hit = dist <= overlap_threshold
            color = '#10B981' if is_hit else miss_color  # emerald green if hit, else miss_color
            colors.append(color)

            # Draw Rectangle Clamp
            rect_pts = get_clamp_rectangle_3d_points(p)
            fig.add_trace(go.Scatter3d(
                x=rect_pts[:, 0], y=rect_pts[:, 1], z=rect_pts[:, 2],
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)

        # F. Markers + Sequence Labels
        fig.add_trace(go.Scatter3d(
            x=x_pts, y=y_pts, z=z_pts,
            mode='markers+text',
            marker=dict(size=7, color=colors, line=dict(color='white', width=1)),
            text=[f"#{i}" for i in range(len(coords))],
            textposition="top center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            name='Clamps',
            showlegend=False
        ), row=row, col=col)

        # G. Highlight First Clamp uniquely
        fig.add_trace(go.Scatter3d(
            x=[coords[0][0]], y=[coords[0][1]], z=[coords[0][2]],
            mode='markers',
            marker=dict(size=12, color='#FFAA00', symbol='diamond', line=dict(color='white', width=1.5)),
            name='First Clamp',
            showlegend=False
        ), row=row, col=col)

    # Populate both subplots
    draw_scene(row=1, col=1, coords=real_coords, miss_color='#FFAA00')
    draw_scene(row=1, col=2, coords=sim_coords, miss_color='#00FFCC')

    # Dark Theme Layout Styling
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f0f1a',
        plot_bgcolor='#0f0f1a',
        width=1400, height=700,
        margin=dict(l=10, r=10, t=80, b=10)
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.6, y=-1.6, z=1.3)
    )
    limit = 2.0
    for scene_name in ['scene', 'scene2']:
        fig.update_layout({
            scene_name: dict(
                xaxis=dict(range=[-limit, limit], showbackground=False, visible=False),
                yaxis=dict(range=[-limit, limit], showbackground=False, visible=False),
                zaxis=dict(range=[-limit, limit], showbackground=False, visible=False),
                camera=camera,
                aspectmode='cube'
            )
        })

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if output_file.endswith('.html'):
            fig.write_html(output_file)
            print(f"Saved interactive 3D HTML to {output_file}")
        else:
            try:
                import kaleido
                fig.write_image(output_file)
                print(f"Saved static 3D comparison plot to {output_file}")
            except ImportError:
                html_path = os.path.splitext(output_file)[0] + ".html"
                fig.write_html(html_path)
                print(f"Plotly requires 'kaleido' for static images. Saved interactive HTML instead: {html_path}")

    if show:
        fig.show()


    return fig


def plot_random_clamp_maps_grid(
    sites_path, 
    ethogram_path=None, 
    show_bottom=True, 
    show_front=True, 
    show_lines=True, 
    simple_lines=True, 
    bottom_color='red', 
    front_color='blue', 
    output_file=None, 
    overlap_threshold=0.50,
    use_rectangle=False,
    clamp_dims=(0.7, 0.4),
    sampling_strategy='random'
):
    """
    Plots a 2x3 grid of 2D unrolled peanut maps:
    - Subplot 1: Actual Animal clamping sequence.
    - Subplots 2-6: Five different simulation runs (with same start point and clamp counts).
    
    Clamps are colored green if their 3D distance to the hole is <= overlap_threshold.

    sampling_strategy : 'random' (default) or 'markov'
        'random'  – uniform area-weighted surface sampling.
        'markov'  – biologically-informed Markov chain walk using empirical
                    L/T transition probabilities and amplitude distributions
                    computed from all sessions in the parent sites directory.
    """
    import os
    import json
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if not os.path.exists(sites_path):
        print(f"Sites file not found: {sites_path}")
        return

    # Load Bottom
    sites_bottom = []
    json_version = "v2"  # default
    if show_bottom:
        with open(sites_path, 'r') as f:
            data = json.load(f)
            sites_bottom = data if isinstance(data, list) else data.get("sites", [])
            if isinstance(data, dict):
                json_version = data.get("version", "v2")

    # Load Front
    sites_front = []
    if show_front:
        path_base = sites_path.replace("_sites.json", "")
        front_path = path_base + "_front_sites.json"
        if os.path.exists(front_path):
            with open(front_path, 'r') as f:
                fd = json.load(f)
                sites_front = fd if isinstance(fd, list) else fd.get("sites", [])

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

    valid_bottom = parse_sites(sites_bottom)
    valid_front = parse_sites(sites_front)

    if not valid_bottom and not valid_front:
        print("No valid sites found to plot.")
        return

    # Setup Physical Geometry and Scaling
    a_true, b_true = 1.2602, 1.3749
    is_hole = 'hole' in os.path.basename(sites_path).lower() or 'hole' in os.path.dirname(sites_path).lower()
    if is_hole:
        a_orig, b_orig = 1.0, 1.1
    else:
        a_orig, b_orig = 1.0, 1.07
    z_max_orig = np.sqrt(a_orig**2 + b_orig**2)
    z_max_true = np.sqrt(a_true**2 + b_true**2)
    scale_factor = z_max_true / z_max_orig

    # Determine Hole Position in scaled coordinates
    if json_version == "v3":
        p_hole = np.array([0.0, 0.0, z_max_true])
    else:
        def peanut_radius_orig(z, a=1.0, b=1.1):
            term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
            r2 = term1 - z**2 - a**2
            return np.sqrt(np.maximum(0, r2))
        r_hole_orig = peanut_radius_orig(0.75, a_orig, b_orig)
        p_hole_orig = np.array([0.0, r_hole_orig, 0.75])
        p_hole = p_hole_orig * scale_factor

    z_h = p_hole[2]
    theta_h = np.arctan2(p_hole[1], p_hole[0])

    def scale_point(p):
        return np.array([p[0] * scale_factor, p[1] * scale_factor, p[2] * scale_factor])

    # Extract first clamps and subsequent counts
    actual_bottom_coords = [scale_point(s['p']) for s in valid_bottom]
    actual_front_coords = [scale_point(s['p']) for s in valid_front]

    num_subsequent_bottom = max(0, len(valid_bottom) - 1)
    num_subsequent_front = max(0, len(valid_front) - 1)
    total_subsequent = num_subsequent_bottom + num_subsequent_front

    b_first_scaled = actual_bottom_coords[0] if len(actual_bottom_coords) > 0 else None
    f_first_scaled = actual_front_coords[0] if len(actual_front_coords) > 0 else None

    # Setup Peanut Mesh for simulation
    A_PARAM, B_PARAM = a_true, b_true
    def peanut_radius_sim(z, a, b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    z_max_sim = np.sqrt(A_PARAM**2 + B_PARAM**2)
    z_lin = np.linspace(-z_max_sim * 0.999, z_max_sim * 0.999, 100)
    theta_lin = np.linspace(0, 2*np.pi, 60, endpoint=False)
    Z_mesh, Theta_mesh = np.meshgrid(z_lin, theta_lin, indexing='ij')
    R_mesh = peanut_radius_sim(Z_mesh, A_PARAM, B_PARAM)
    X_mesh = R_mesh * np.cos(Theta_mesh)
    Y_mesh = R_mesh * np.sin(Theta_mesh)

    faces = []
    face_areas = []
    rows, cols = X_mesh.shape
    for r in range(rows - 1):
        for c in range(cols):
            p1 = np.array([X_mesh[r, c], Y_mesh[r, c], Z_mesh[r, c]])
            p2 = np.array([X_mesh[r+1, c], Y_mesh[r+1, c], Z_mesh[r+1, c]])
            nxt_c = (c + 1) % cols
            p3 = np.array([X_mesh[r+1, nxt_c], Y_mesh[r+1, nxt_c], Z_mesh[r+1, nxt_c]])
            p4 = np.array([X_mesh[r, nxt_c], Y_mesh[r, nxt_c], Z_mesh[r, nxt_c]])
            c1 = (np.linalg.norm(np.cross(p2-p1, p3-p1)) + np.linalg.norm(np.cross(p3-p1, p4-p1))) / 2.0
            faces.append((p1 + p2 + p3 + p4) / 4.0)
            face_areas.append(c1)

    faces = np.array(faces)
    face_areas = np.array(face_areas)
    face_probabilities = face_areas / np.sum(face_areas)

    # ------------------------------------------------------------------
    # Build empirical Markov parameters (used when sampling_strategy == 'markov')
    # Scans all *_sites.json files in the parent directory of sites_path.
    # ------------------------------------------------------------------
    _markov_params_grid = None
    if sampling_strategy == 'markov':
        _sites_dir_grid = os.path.dirname(sites_path)
        _bottom_files_grid = sorted([
            f for f in glob.glob(os.path.join(_sites_dir_grid, "*_sites.json"))
            if not f.endswith("_front_sites.json")
        ])
        def _pr_markov_grid(z, a=1.0, b=1.1):
            t1 = np.sqrt(b**4 + 4 * a**2 * z**2)
            return np.sqrt(np.maximum(0, t1 - z**2 - a**2))
        _a_phys_g, _b_phys_g = 1.2602, 1.3749
        _z_max_phys_g = np.sqrt(_a_phys_g**2 + _b_phys_g**2)
        _scale_g = _z_max_phys_g / np.sqrt(1.0**2 + 1.1**2)
        _trans_g  = {'LL': 0, 'LT': 0, 'TL': 0, 'TT': 0}
        _counts_g = {'L': 0, 'T': 0}
        _long_amps_g, _trans_amps_g = [], []
        for _bf_g in _bottom_files_grid:
            for _gpath in [_bf_g, _bf_g.replace('_sites.json', '_front_sites.json')]:
                if not os.path.exists(_gpath): continue
                try:
                    with open(_gpath) as _fh:
                        _gd = json.load(_fh)
                        _graw = _gd if isinstance(_gd, list) else _gd.get('sites', [])
                        _gch = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3])}
                                for s in _graw if len(s) >= 4]
                    _gch.sort(key=lambda x: x['frame'])
                    _gseq = []
                    for _gi in range(len(_gch) - 1):
                        _gp1, _gp2 = _gch[_gi]['p'], _gch[_gi+1]['p']
                        _gdz_signed = get_geodesic_profile_distance(_gp1[2], _gp2[2], a=1.0, b=1.1)
                        _gaz   = (_gp1[2] + _gp2[2]) / 2
                        _gr    = _pr_markov_grid(_gaz)
                        _ga1   = np.arctan2(_gp1[1], _gp1[0])
                        _ga2   = np.arctan2(_gp2[1], _gp2[0])
                        _gdarc_signed = _gr * (((_ga2 - _ga1 + np.pi) % (2 * np.pi)) - np.pi)
                        _gseq.append(('L', _gdz_signed, _gdarc_signed) if abs(_gdz_signed) >= abs(_gdarc_signed) else ('T', _gdz_signed, _gdarc_signed))
                    for _gi, (_gm, _gdz_val, _gdarc_val) in enumerate(_gseq):
                        if _gm == 'L': _long_amps_g.append(_gdz_val * _scale_g)
                        else:          _trans_amps_g.append(_gdarc_val * _scale_g)
                        if _gi < len(_gseq) - 1:
                            _gpair = _gm + _gseq[_gi+1][0]
                            _trans_g[_gpair] += 1
                            _counts_g[_gm]   += 1
                except: pass
        _pLL_g = _trans_g['LL'] / _counts_g['L'] if _counts_g['L'] > 0 else 0.5
        _pTT_g = _trans_g['TT'] / _counts_g['T'] if _counts_g['T'] > 0 else 0.5
        _markov_params_grid = dict(
            prob_LL=_pLL_g, prob_LT=1-_pLL_g,
            prob_TT=_pTT_g, prob_TL=1-_pTT_g,
            long_amps=np.array(_long_amps_g) if _long_amps_g else np.array([0.3]),
            trans_amps=np.array(_trans_amps_g) if _trans_amps_g else np.array([0.3]),
            a_phys=_a_phys_g, b_phys=_b_phys_g, z_max_phys=_z_max_phys_g
        )
        print(f"[Markov grid] P(L->L)={_pLL_g:.3f}  P(T->T)={_pTT_g:.3f}  "
              f"L-amps median={np.median(_markov_params_grid['long_amps']):.3f} cm  "
              f"T-amps median={np.median(_markov_params_grid['trans_amps']):.3f} cm")

    def _markov_run_grid(start_pt, n_steps, mp):
        """Generate n_steps positions via Markov walk from start_pt (physical coords)."""
        _a_p = mp['a_phys']; _b_p = mp['b_phys']; _z_max_p = mp['z_max_phys']
        def _surf(z, th):
            t1 = np.sqrt(_b_p**4 + 4*_a_p**2*z**2)
            r  = np.sqrt(max(0.0, t1 - z**2 - _a_p**2))
            return np.array([r*np.cos(th), r*np.sin(th), z])
        _cur_z = np.clip(start_pt[2], -_z_max_p*0.999, _z_max_p*0.999)
        _cur_th = np.arctan2(start_pt[1], start_pt[0])
        _cur_type = 'L' if np.random.random() < mp['prob_LL'] else 'T'
        pts = []
        for _ in range(n_steps):
            if _cur_type == 'L':
                _dz = np.random.choice(mp['long_amps'])
                _cur_z = np.clip(_cur_z + _dz, -_z_max_p*0.999, _z_max_p*0.999)
            else:
                _darc_signed = np.random.choice(mp['trans_amps'])
                t1 = np.sqrt(_b_p**4 + 4*_a_p**2*_cur_z**2)
                _r = max(1e-6, np.sqrt(max(0.0, t1 - _cur_z**2 - _a_p**2)))
                _cur_th = (_cur_th + (_darc_signed/_r)) % (2*np.pi)
            pts.append(_surf(_cur_z, _cur_th))
            _cur_type = ('L' if np.random.random()<mp['prob_LL'] else 'T') if _cur_type=='L' else \
                        ('T' if np.random.random()<mp['prob_TT'] else 'L')
        return pts

    # 2D projection parameters
    S = 1.85
    faces_proj = {
        0: {'az': 0, 'cx': -S/1.5, 'cy': 0},
        1: {'az': 180, 'cx': S/1.5, 'cy': 0}
    }
    z_vals = np.linspace(-z_max_true, z_max_true, 200)
    def peanut_radius_local(z, a=a_true, b=b_true):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))
    r_vals = peanut_radius_local(z_vals, a_true, b_true)
    width_vals = r_vals * (np.pi / 2) 

    def wrap_angle(val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def get_distance_to_hole(p_sc):
        if not use_rectangle:
            return np.linalg.norm(p_sc - p_hole)
        else:
            z_c = p_sc[2]
            theta_c = np.arctan2(p_sc[1], p_sc[0])
            dy = z_c - z_h
            dx_angle = wrap_angle(theta_c - theta_h)
            r_c = peanut_radius_sim(z_c, A_PARAM, B_PARAM)
            dx = r_c * dx_angle
            dist_x = max(0.0, abs(dx) - clamp_dims[1] / 2.0)
            dist_y = max(0.0, abs(dy) - clamp_dims[0] / 2.0)
            return np.sqrt(dist_x**2 + dist_y**2)

    def map_to_canvas_physical(p_physical):
        x_scaled, y_scaled, z_scaled = p_physical
        if is_hole:
            x_scaled, y_scaled = y_scaled, -x_scaled
        angle = np.degrees(np.arctan2(y_scaled, x_scaled)) % 360
        
        best_diff = 360; best_fid = -1
        for fid, f in faces_proj.items():
            diff = abs(angle - f['az'])
            if diff > 180: diff = 360 - diff
            if diff < best_diff: best_diff = diff; best_fid = fid
            
        f = faces_proj[best_fid]
        raw_diff = angle - f['az']
        if raw_diff > 180: raw_diff -= 360
        elif raw_diff < -180: raw_diff += 360
        
        delta_rad = np.radians(raw_diff)
        curr_r = np.sqrt(x_scaled**2 + y_scaled**2)
        arc_dist = curr_r * delta_rad
        return f['cx'] + arc_dist, f['cy'] + z_scaled

    # Helper function to draw a single map
    def draw_face_map(ax_subplot, title_text, bottom_coords, front_coords):
        for fid, f in faces_proj.items():
            cx, cy = f['cx'], f['cy']
            ax_subplot.plot(cx + width_vals, cy + z_vals, 'k-', linewidth=1.5)
            ax_subplot.plot(cx - width_vals, cy + z_vals, 'k-', linewidth=1.5)
            ax_subplot.fill_betweenx(cy + z_vals, cx - width_vals, cx + width_vals, color='#a6806d', alpha=0.5)
            
            if is_hole:
                if json_version == "v3":
                    z_hole_min = z_max_true - 0.20 * scale_factor
                    z_hole_max = z_max_true
                    z_hole_vals = np.linspace(z_hole_min, z_hole_max, 50)
                    r_hole_vals = peanut_radius_local(z_hole_vals, a_true, b_true)
                    width_hole_vals = r_hole_vals * (np.pi / 2)
                    ax_subplot.fill_betweenx(cy + z_hole_vals, cx - width_hole_vals, cx + width_hole_vals, color='black', alpha=0.85, zorder=8)
                else:
                    x_h, y_h = map_to_canvas_physical(p_hole)
                    if abs(x_h - cx) < S:
                        c_hole = Circle((x_h, y_h), radius=0.15, facecolor='black', edgecolor='black', zorder=18)
                        ax_subplot.add_patch(c_hole)
            
            label_text = f"Front ({f['az']}°)" if fid == 0 else f"Back ({f['az']}°)"
            ax_subplot.text(cx, cy + z_max_true + 0.3, label_text, ha='center', fontsize=10, fontweight='bold')

        # Draw lines
        if show_lines:
            if show_bottom and len(bottom_coords) > 1:
                mapped_b = [map_to_canvas_physical(p) for p in bottom_coords]
                for i in range(len(mapped_b) - 1):
                    p1, p2 = mapped_b[i], mapped_b[i+1]
                    ax_subplot.plot([p1[0], p2[0]], [p1[1], p2[1]], color=bottom_color, linestyle='--', linewidth=1.5, alpha=0.5, zorder=15)
            if show_front and len(front_coords) > 1:
                mapped_f = [map_to_canvas_physical(p) for p in front_coords]
                for i in range(len(mapped_f) - 1):
                    p1, p2 = mapped_f[i], mapped_f[i+1]
                    ax_subplot.plot([p1[0], p2[0]], [p1[1], p2[1]], color=front_color, linestyle='--', linewidth=1.5, alpha=0.5, zorder=15)

        # Draw Clamps
        marker_area = 0.5
        
        if show_bottom and len(bottom_coords) > 0:
            n_b = len(bottom_coords)
            r_b = np.sqrt((marker_area / n_b) / np.pi) if n_b > 0 else 0.1
            for i, p in enumerate(bottom_coords):
                X, Y = map_to_canvas_physical(p)
                dist_to_hole = get_distance_to_hole(p)
                is_contact = dist_to_hole <= overlap_threshold
                if is_contact and is_hole and json_version != "v3":
                    if X > 0:  # Back face, opposite the hole
                        is_contact = False
                color = 'green' if is_contact else bottom_color
                if not use_rectangle:
                    c = Circle((X, Y), radius=r_b, facecolor=color, edgecolor='black', zorder=20, alpha=0.7)
                    ax_subplot.add_patch(c)
                else:
                    rect = Rectangle(
                        (X - clamp_dims[1] / 2.0, Y - clamp_dims[0] / 2.0), 
                        clamp_dims[1], clamp_dims[0], 
                        facecolor=color, 
                        edgecolor='black', 
                        zorder=20, 
                        alpha=0.7
                    )
                    ax_subplot.add_patch(rect)
                ax_subplot.text(X, Y, str(i), fontsize=8, color='white', fontweight='bold', ha='center', va='center', zorder=25)

        if show_front and len(front_coords) > 0:
            n_f = len(front_coords)
            r_f = np.sqrt((marker_area / n_f) / np.pi) if n_f > 0 else 0.1
            for i, p in enumerate(front_coords):
                X, Y = map_to_canvas_physical(p)
                dist_to_hole = get_distance_to_hole(p)
                is_contact = dist_to_hole <= overlap_threshold
                if is_contact and is_hole and json_version != "v3":
                    if X > 0:  # Back face, opposite the hole
                        is_contact = False
                color = 'green' if is_contact else front_color
                if not use_rectangle:
                    c = Circle((X, Y), radius=r_f, facecolor=color, edgecolor='black', zorder=20, alpha=0.7)
                    ax_subplot.add_patch(c)
                else:
                    rect = Rectangle(
                        (X - clamp_dims[1] / 2.0, Y - clamp_dims[0] / 2.0), 
                        clamp_dims[1], clamp_dims[0], 
                        facecolor=color, 
                        edgecolor='black', 
                        zorder=20, 
                        alpha=0.7
                    )
                    ax_subplot.add_patch(rect)
                ax_subplot.text(X, Y, str(i), fontsize=8, color='white', fontweight='bold', ha='center', va='center', zorder=25)

        ax_subplot.set_aspect('equal')
        ax_subplot.set_xlim(-S*1.5, S*1.5)
        ax_subplot.set_ylim(-3.5, 3.5)
        ax_subplot.axis('off')
        ax_subplot.set_title(title_text, fontsize=12, pad=10)

    # Main Grid Setup (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Plot Actual Animal
    actual_min_dist = np.min([get_distance_to_hole(pt) for pt in actual_bottom_coords + actual_front_coords])
    actual_touch = actual_min_dist <= overlap_threshold
    actual_title = f"Actual Animal\n(Min Dist = {actual_min_dist:.3f} cm, Touch: {'Yes' if actual_touch else 'No'})"
    draw_face_map(axes[0, 0], actual_title, actual_bottom_coords, actual_front_coords)

    # 2. Plot 5 Simulations (random or Markov)
    _strat_label_grid = 'Markov Run' if sampling_strategy == 'markov' else 'Random Run'
    for sim_idx in range(1, 6):
        if sampling_strategy == 'markov' and _markov_params_grid is not None:
            sim_b = []
            if show_bottom and b_first_scaled is not None:
                walk_b = _markov_run_grid(b_first_scaled, num_subsequent_bottom, _markov_params_grid)
                sim_b = [b_first_scaled] + walk_b
            sim_f = []
            if show_front and f_first_scaled is not None:
                walk_f = _markov_run_grid(f_first_scaled, num_subsequent_front, _markov_params_grid)
                sim_f = [f_first_scaled] + walk_f
        else:
            # Original random sampling
            sampled_indices = np.random.choice(len(faces), size=total_subsequent, p=face_probabilities)
            sim_coords = faces[sampled_indices]
            sim_b = []
            if show_bottom and b_first_scaled is not None:
                sim_b = [b_first_scaled] + list(sim_coords[:num_subsequent_bottom])
            sim_f = []
            if show_front and f_first_scaled is not None:
                sim_f = [f_first_scaled] + list(sim_coords[num_subsequent_bottom:])
            
        # Compute stats for this simulation run
        all_sim_coords = sim_b + sim_f
        sim_min_dist = np.min([get_distance_to_hole(pt) for pt in all_sim_coords])
        sim_touch = sim_min_dist <= overlap_threshold
        
        row = (sim_idx) // 3
        col = (sim_idx) % 3
        
        sim_title = f"{_strat_label_grid} #{sim_idx}\n(Min Dist = {sim_min_dist:.3f} cm, Touch: {'Yes' if sim_touch else 'No'})"
        draw_face_map(axes[row, col], sim_title, sim_b, sim_f)

    _main_title = 'Markov Walk' if sampling_strategy == 'markov' else 'Random Search'
    plt.suptitle(f"Clamping Search Realizations: Animal vs. {_main_title}", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if output_file.endswith('.png'):
            plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
        elif output_file.endswith('.svg'):
            plt.savefig(output_file.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
        print(f"Saved random map comparison grid to {output_file}")

    plt.show()
    plt.close(fig)



def plot_random_clamp_maps_grid_3d(
    sites_path, 
    ethogram_path=None, 
    show_bottom=True, 
    show_front=True, 
    show_lines=True, 
    simple_lines=True, 
    bottom_color='red', 
    front_color='blue', 
    output_file=None, 
    overlap_threshold=0.50,
    use_rectangle=False,
    clamp_dims=(0.7, 0.4),
    sampling_strategy='random'
):
    """
    Plots a 2x3 grid of interactive 3D peanut models:
    - Subplot 1: Actual Animal clamping sequence.
    - Subplots 2-6: Five different simulation runs (with same start point and clamp counts).
    
    Clamps are colored green if their 3D distance to the hole is <= overlap_threshold.
    Returns the Plotly figure.

    sampling_strategy : 'random' (default) or 'markov'
        'random'  – uniform area-weighted surface sampling.
        'markov'  – biologically-informed Markov chain walk using empirical
                    L/T transition probabilities and amplitude distributions
                    computed from all sessions in the parent sites directory.
    """
    import os
    import sys
    import json
    import glob
    import numpy as np
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Error: 'plotly' package is required for interactive 3D plots. Please install it with 'pip install plotly'.")
        return None

    if not os.path.exists(sites_path):
        print(f"Sites file not found: {sites_path}")
        return None

    # Load Bottom
    sites_bottom = []
    json_version = "v2"
    with open(sites_path, 'r') as f:
        data = json.load(f)
        sites_bottom = data if isinstance(data, list) else data.get("sites", [])
        if isinstance(data, dict):
            json_version = data.get("version", "v2")

    # Load Front
    sites_front = []
    path_base = sites_path.replace("_sites.json", "")
    front_path = path_base + "_front_sites.json"
    if os.path.exists(front_path):
        with open(front_path, 'r') as f:
            fd = json.load(f)
            sites_front = fd if isinstance(fd, list) else fd.get("sites", [])

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

    valid_bottom = parse_sites(sites_bottom)
    valid_front = parse_sites(sites_front)

    if not valid_bottom and not valid_front:
        print("No valid sites found to plot.")
        return None

    # Setup Physical Geometry and Scaling
    a_true, b_true = 1.2602, 1.3749
    is_hole = 'hole' in os.path.basename(sites_path).lower() or 'hole' in os.path.dirname(sites_path).lower()
    if is_hole:
        a_orig, b_orig = 1.0, 1.1
    else:
        a_orig, b_orig = 1.0, 1.07
    z_max_orig = np.sqrt(a_orig**2 + b_orig**2)
    z_max_true = np.sqrt(a_true**2 + b_true**2)
    scale_factor = z_max_true / z_max_orig

    # Determine Hole Position in scaled coordinates
    if json_version == "v3":
        p_hole = np.array([0.0, 0.0, z_max_true])
    else:
        def peanut_radius_orig(z, a=1.0, b=1.1):
            term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
            r2 = term1 - z**2 - a**2
            return np.sqrt(np.maximum(0, r2))
        r_hole_orig = peanut_radius_orig(0.75, a_orig, b_orig)
        p_hole_orig = np.array([0.0, r_hole_orig, 0.75])
        p_hole = p_hole_orig * scale_factor

    z_h = p_hole[2]
    theta_h = np.arctan2(p_hole[1], p_hole[0])

    def scale_point(p):
        return np.array([p[0] * scale_factor, p[1] * scale_factor, p[2] * scale_factor])

    # Extract first clamps and subsequent counts
    actual_bottom_coords = [scale_point(s['p']) for s in valid_bottom]
    actual_front_coords = [scale_point(s['p']) for s in valid_front]

    num_subsequent_bottom = max(0, len(valid_bottom) - 1)
    num_subsequent_front = max(0, len(valid_front) - 1)
    total_subsequent = num_subsequent_bottom + num_subsequent_front

    b_first_scaled = actual_bottom_coords[0] if len(actual_bottom_coords) > 0 else None
    f_first_scaled = actual_front_coords[0] if len(actual_front_coords) > 0 else None

    # Setup Peanut Mesh for simulation
    A_PARAM, B_PARAM = a_true, b_true
    def peanut_radius_sim(z, a, b):
        term1 = np.sqrt(b**4 + 4 * a**2 * z**2)
        r2 = term1 - z**2 - a**2
        return np.sqrt(np.maximum(0, r2))

    def wrap_angle(val):
        return (val + np.pi) % (2 * np.pi) - np.pi

    def get_distance_to_hole(p_sc):
        if not use_rectangle:
            return np.linalg.norm(p_sc - p_hole)
        else:
            z_c = p_sc[2]
            theta_c = np.arctan2(p_sc[1], p_sc[0])
            dy = z_c - z_h
            dx_angle = wrap_angle(theta_c - theta_h)
            r_c = peanut_radius_sim(z_c, A_PARAM, B_PARAM)
            dx = r_c * dx_angle
            dist_x = max(0.0, abs(dx) - clamp_dims[1] / 2.0)
            dist_y = max(0.0, abs(dy) - clamp_dims[0] / 2.0)
            return np.sqrt(dist_x**2 + dist_y**2)

    def get_clamp_rectangle_3d_points(p, W=clamp_dims[1], L=clamp_dims[0]):
        z_c = p[2]
        theta_c = np.arctan2(p[1], p[0])
        boundary_pts = []
        
        # Bottom edge
        z_val = z_c - L/2
        z_val = np.clip(z_val, -z_max_true + 0.01, z_max_true - 0.01)
        r_val = max(0.01, peanut_radius_sim(z_val, A_PARAM, B_PARAM))
        d_theta = (W / 2.0) / r_val
        for t in np.linspace(theta_c - d_theta, theta_c + d_theta, 5):
            boundary_pts.append([r_val * np.cos(t), r_val * np.sin(t), z_val])
            
        # Top edge
        z_val = z_c + L/2
        z_val = np.clip(z_val, -z_max_true + 0.01, z_max_true - 0.01)
        r_val = max(0.01, peanut_radius_sim(z_val, A_PARAM, B_PARAM))
        d_theta = (W / 2.0) / r_val
        for t in np.linspace(theta_c + d_theta, theta_c - d_theta, 5):
            boundary_pts.append([r_val * np.cos(t), r_val * np.sin(t), z_val])
            
        boundary_pts.append(boundary_pts[0])
        return np.array(boundary_pts)

    z_max_sim = np.sqrt(A_PARAM**2 + B_PARAM**2)
    z_lin = np.linspace(-z_max_sim * 0.999, z_max_sim * 0.999, 100)
    theta_lin = np.linspace(0, 2*np.pi, 60, endpoint=False)
    Z_mesh, Theta_mesh = np.meshgrid(z_lin, theta_lin, indexing='ij')
    R_mesh = peanut_radius_sim(Z_mesh, A_PARAM, B_PARAM)
    X_mesh = R_mesh * np.cos(Theta_mesh)
    Y_mesh = R_mesh * np.sin(Theta_mesh)

    # Compute distance to hole for color mapping
    dists_mesh = np.sqrt((X_mesh - p_hole[0])**2 + (Y_mesh - p_hole[1])**2 + (Z_mesh - p_hole[2])**2)
    hole_mask = (dists_mesh <= overlap_threshold).astype(float)

    # Build faces list for random sampling
    faces = []
    face_areas = []
    rows, cols = X_mesh.shape
    for r in range(rows - 1):
        for c in range(cols):
            p1 = np.array([X_mesh[r, c], Y_mesh[r, c], Z_mesh[r, c]])
            p2 = np.array([X_mesh[r+1, c], Y_mesh[r+1, c], Z_mesh[r+1, c]])
            nxt_c = (c + 1) % cols
            p3 = np.array([X_mesh[r+1, nxt_c], Y_mesh[r+1, nxt_c], Z_mesh[r+1, nxt_c]])
            p4 = np.array([X_mesh[r, nxt_c], Y_mesh[r, nxt_c], Z_mesh[r, nxt_c]])
            c1 = (np.linalg.norm(np.cross(p2-p1, p3-p1)) + np.linalg.norm(np.cross(p3-p1, p4-p1))) / 2.0
            faces.append((p1 + p2 + p3 + p4) / 4.0)
            face_areas.append(c1)

    faces = np.array(faces)
    face_areas = np.array(face_areas)
    face_probabilities = face_areas / np.sum(face_areas)

    # ------------------------------------------------------------------
    # Build empirical Markov parameters for the 3D grid (when 'markov')
    # ------------------------------------------------------------------
    _markov_params_3d = None
    if sampling_strategy == 'markov':
        _sites_dir_3d = os.path.dirname(sites_path)
        _bottom_files_3d = sorted([
            f for f in glob.glob(os.path.join(_sites_dir_3d, "*_sites.json"))
            if not f.endswith("_front_sites.json")
        ])
        def _pr_markov_3d(z, a=1.0, b=1.1):
            t1 = np.sqrt(b**4 + 4*a**2*z**2)
            return np.sqrt(np.maximum(0, t1 - z**2 - a**2))
        _a_phys_3d, _b_phys_3d = 1.2602, 1.3749
        _z_max_phys_3d = np.sqrt(_a_phys_3d**2 + _b_phys_3d**2)
        _scale_3d = _z_max_phys_3d / np.sqrt(1.0**2 + 1.1**2)
        _trans_3d  = {'LL': 0, 'LT': 0, 'TL': 0, 'TT': 0}
        _counts_3d = {'L': 0, 'T': 0}
        _long_amps_3d, _trans_amps_3d = [], []
        for _bf_3d in _bottom_files_3d:
            for _gpath in [_bf_3d, _bf_3d.replace('_sites.json', '_front_sites.json')]:
                if not os.path.exists(_gpath): continue
                try:
                    with open(_gpath) as _fh:
                        _gd = json.load(_fh)
                        _graw = _gd if isinstance(_gd, list) else _gd.get('sites', [])
                        _gch = [{'p': [float(s[0]), float(s[1]), float(s[2])], 'frame': int(s[3])}
                                for s in _graw if len(s) >= 4]
                    _gch.sort(key=lambda x: x['frame'])
                    _gseq = []
                    for _gi in range(len(_gch) - 1):
                        _gp1, _gp2 = _gch[_gi]['p'], _gch[_gi+1]['p']
                        _gdz_signed = get_geodesic_profile_distance(_gp1[2], _gp2[2], a=1.0, b=1.1)
                        _gaz   = (_gp1[2] + _gp2[2]) / 2
                        _gr    = _pr_markov_3d(_gaz)
                        _ga1   = np.arctan2(_gp1[1], _gp1[0])
                        _ga2   = np.arctan2(_gp2[1], _gp2[0])
                        _gdarc_signed = _gr * (((_ga2 - _ga1 + np.pi) % (2 * np.pi)) - np.pi)
                        _gseq.append(('L', _gdz_signed, _gdarc_signed) if abs(_gdz_signed) >= abs(_gdarc_signed) else ('T', _gdz_signed, _gdarc_signed))
                    for _gi, (_gm, _gdz_val, _gdarc_val) in enumerate(_gseq):
                        if _gm == 'L': _long_amps_3d.append(_gdz_val * _scale_3d)
                        else:          _trans_amps_3d.append(_gdarc_val * _scale_3d)
                        if _gi < len(_gseq) - 1:
                            _trans_3d[_gm + _gseq[_gi+1][0]] += 1
                            _counts_3d[_gm] += 1
                except: pass
        _pLL_3d = _trans_3d['LL']/_counts_3d['L'] if _counts_3d['L']>0 else 0.5
        _pTT_3d = _trans_3d['TT']/_counts_3d['T'] if _counts_3d['T']>0 else 0.5
        _markov_params_3d = dict(
            prob_LL=_pLL_3d, prob_LT=1-_pLL_3d,
            prob_TT=_pTT_3d, prob_TL=1-_pTT_3d,
            long_amps=np.array(_long_amps_3d) if _long_amps_3d else np.array([0.3]),
            trans_amps=np.array(_trans_amps_3d) if _trans_amps_3d else np.array([0.3]),
            a_phys=_a_phys_3d, b_phys=_b_phys_3d, z_max_phys=_z_max_phys_3d
        )
        print(f"[Markov 3D] P(L->L)={_pLL_3d:.3f}  P(T->T)={_pTT_3d:.3f}")

    def _markov_run_3d(start_pt, n_steps, mp):
        """Generate n_steps positions via Markov walk (physical coords)."""
        _a_p=mp['a_phys']; _b_p=mp['b_phys']; _z_max_p=mp['z_max_phys']
        def _surf(z, th):
            t1=np.sqrt(_b_p**4+4*_a_p**2*z**2)
            r=np.sqrt(max(0.0,t1-z**2-_a_p**2))
            return np.array([r*np.cos(th),r*np.sin(th),z])
        _cur_z=np.clip(start_pt[2],-_z_max_p*0.999,_z_max_p*0.999)
        _cur_th=np.arctan2(start_pt[1],start_pt[0])
        _cur_type='L' if np.random.random()<mp['prob_LL'] else 'T'
        pts=[]
        for _ in range(n_steps):
            if _cur_type == 'L':
                _dz = np.random.choice(mp['long_amps'])
                _cur_z = np.clip(_cur_z + _dz, -_z_max_p * 0.999, _z_max_p * 0.999)
            else:
                _darc_signed = np.random.choice(mp['trans_amps'])
                t1 = np.sqrt(_b_p**4 + 4 * _a_p**2 * _cur_z**2)
                _r = max(1e-6, np.sqrt(max(0.0, t1 - _cur_z**2 - _a_p**2)))
                _cur_th = (_cur_th + (_darc_signed / _r)) % (2 * np.pi)
            pts.append(_surf(_cur_z, _cur_th))
            _cur_type = ('L' if np.random.random() < mp['prob_LL'] else 'T') if _cur_type == 'L' else \
                        ('T' if np.random.random() < mp['prob_TT'] else 'L')
        return pts

    # Generate the 5 simulations first to compile titles
    _strat_label_3d = 'Markov Run' if sampling_strategy == 'markov' else 'Random Run'
    sim_runs = []
    for sim_idx in range(1, 6):
        if sampling_strategy == 'markov' and _markov_params_3d is not None:
            sim_b = []
            if b_first_scaled is not None:
                sim_b = [b_first_scaled] + _markov_run_3d(b_first_scaled, num_subsequent_bottom, _markov_params_3d)
            sim_f = []
            if f_first_scaled is not None:
                sim_f = [f_first_scaled] + _markov_run_3d(f_first_scaled, num_subsequent_front, _markov_params_3d)
        else:
            sampled_indices = np.random.choice(len(faces), size=total_subsequent, p=face_probabilities)
            sim_coords = faces[sampled_indices]
            sim_b = []
            if b_first_scaled is not None:
                sim_b = [b_first_scaled] + list(sim_coords[:num_subsequent_bottom])
            sim_f = []
            if f_first_scaled is not None:
                sim_f = [f_first_scaled] + list(sim_coords[num_subsequent_bottom:])
            
        all_sim_coords = sim_b + sim_f
        sim_min_dist = np.min([get_distance_to_hole(pt) for pt in all_sim_coords])
        sim_touch = sim_min_dist <= overlap_threshold
        sim_runs.append({
            'b': sim_b,
            'f': sim_f,
            'min_dist': sim_min_dist,
            'touch': sim_touch
        })

    # Compile Titles
    actual_min_dist = np.min([get_distance_to_hole(pt) for pt in actual_bottom_coords + actual_front_coords])
    actual_touch = actual_min_dist <= overlap_threshold
    
    titles = [
        f"Actual Animal<br><sub>Min Dist: {actual_min_dist:.3f} cm | Touch: {'Yes' if actual_touch else 'No'}</sub>"
    ]
    for i, run in enumerate(sim_runs):
        titles.append(
            f"{_strat_label_3d} {i+1}<br><sub>Min Dist: {run['min_dist']:.3f} cm | Touch: {'Yes' if run['touch'] else 'No'}</sub>"
        )

    # Plotly Subplots configuration (2x3 scene subplots)
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
            [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]
        ],
        subplot_titles=titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.08
    )

    # Helper function to add a 3D peanut and clamps to a specific subplot
    def draw_3d_peanut_and_clamps(row, col, bottom_coords, front_coords):
        # 1. Add Peanut Surface
        fig.add_trace(go.Surface(
            x=X_mesh, y=Y_mesh, z=Z_mesh,
            surfacecolor=hole_mask,
            colorscale=[[0.0, '#C89C64'], [1.0, '#1E1E1E']],  # Tan shell, black hole
            cmin=0, cmax=1,
            showscale=False,
            opacity=0.75,
            hoverinfo='skip',
            name='Peanut Shell'
        ), row=row, col=col)

        # 2. Add Hole Target Center Marker
        fig.add_trace(go.Scatter3d(
            x=[p_hole[0]], y=[p_hole[1]], z=[p_hole[2]],
            mode='markers',
            marker=dict(size=8, color='#10B981', symbol='circle'), # Emerald green target center
            name='Hole Center',
            showlegend=False
        ), row=row, col=col)

        # 3. Add Bottom Clamps
        if len(bottom_coords) > 0:
            x_b = [p[0] for p in bottom_coords]
            y_b = [p[1] for p in bottom_coords]
            z_b = [p[2] for p in bottom_coords]
            colors_b = []
            for p in bottom_coords:
                dist = get_distance_to_hole(p)
                colors_b.append('#10B981' if dist <= overlap_threshold else '#EF4444')  # Green if hit, Red if miss

            # Lines connecting chronologically
            fig.add_trace(go.Scatter3d(
                x=x_b, y=y_b, z=z_b,
                mode='lines',
                line=dict(color='rgba(239, 68, 68, 0.6)', width=4),
                showlegend=False
            ), row=row, col=col)

            if use_rectangle:
                for p, color in zip(bottom_coords, colors_b):
                    rect_pts = get_clamp_rectangle_3d_points(p)
                    fig.add_trace(go.Scatter3d(
                        x=rect_pts[:, 0], y=rect_pts[:, 1], z=rect_pts[:, 2],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=row, col=col)

            # Markers with original indexes
            fig.add_trace(go.Scatter3d(
                x=x_b, y=y_b, z=z_b,
                mode='markers+text',
                marker=dict(size=7, color=colors_b, line=dict(color='white', width=1)),
                text=[str(i) for i in range(len(bottom_coords))],
                textposition="top center",
                textfont=dict(size=10, color='white', family='Arial Black'),
                name='Bottom Clamps',
                showlegend=False
            ), row=row, col=col)

        # 4. Add Front Clamps
        if len(front_coords) > 0:
            x_f = [p[0] for p in front_coords]
            y_f = [p[1] for p in front_coords]
            z_f = [p[2] for p in front_coords]
            colors_f = []
            for p in front_coords:
                dist = get_distance_to_hole(p)
                colors_f.append('#10B981' if dist <= overlap_threshold else '#3B82F6')  # Green if hit, Blue if miss

            # Lines connecting chronologically
            fig.add_trace(go.Scatter3d(
                x=x_f, y=y_f, z=z_f,
                mode='lines',
                line=dict(color='rgba(59, 130, 246, 0.6)', width=4),
                showlegend=False
            ), row=row, col=col)

            if use_rectangle:
                for p, color in zip(front_coords, colors_f):
                    rect_pts = get_clamp_rectangle_3d_points(p)
                    fig.add_trace(go.Scatter3d(
                        x=rect_pts[:, 0], y=rect_pts[:, 1], z=rect_pts[:, 2],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=row, col=col)

            # Markers with original indexes
            fig.add_trace(go.Scatter3d(
                x=x_f, y=y_f, z=z_f,
                mode='markers+text',
                marker=dict(size=7, color=colors_f, line=dict(color='white', width=1)),
                text=[str(i) for i in range(len(front_coords))],
                textposition="top center",
                textfont=dict(size=10, color='white', family='Arial Black'),
                name='Front Clamps',
                showlegend=False
            ), row=row, col=col)

    # 1. Plot Actual Animal
    draw_3d_peanut_and_clamps(1, 1, actual_bottom_coords, actual_front_coords)

    # 2. Plot 5 Random Simulations
    for sim_idx, run in enumerate(sim_runs):
        r = (sim_idx + 1) // 3 + 1
        c = (sim_idx + 1) % 3 + 1
        draw_3d_peanut_and_clamps(r, c, run['b'], run['f'])

    # Styling and Layout
    DARK_BG = '#09090E'
    
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.3, y=1.3, z=0.7)
    )

    _title_strat_3d = 'Markov Walk' if sampling_strategy == 'markov' else 'Random Search'
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        title=dict(
            text=f"Interactive 3D Clamping Sequences: Animal vs. {_title_strat_3d}",
            font=dict(size=22, color='white', family='Arial Black'),
            y=0.96, x=0.5, xanchor='center'
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        height=950,
        showlegend=False
    )

    # Apply aspect ratios and camera angles to all scenes, hiding background/grids
    for r in [1, 2]:
        for c in [1, 2, 3]:
            fig.update_scenes(
                dict(
                    xaxis=dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
                    yaxis=dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
                    zaxis=dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
                    aspectmode='data',
                    camera=camera
                ),
                row=r, col=c
            )

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        html_file = output_file
        if output_file.endswith('.svg') or output_file.endswith('.png'):
            base, _ = os.path.splitext(output_file)
            html_file = base + ".html"
        
        fig.write_html(html_file)
        print(f"Saved interactive 3D comparison grid to {html_file}")
        
        # Try writing static image if requested
        if output_file.endswith('.svg') or output_file.endswith('.png'):
            try:
                fig.write_image(output_file)
                print(f"Saved static 3D comparison grid to {output_file}")
            except Exception as e:
                print(f"Note: Could not save static image {output_file} (requires 'kaleido' package).")

    # Show figure if in Jupyter notebook or interactive environment
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        fig.show()

    return fig




def plot_real_peanut_3d_surface(a=1.2602, b=1.3749, colorscale='brwnyl', output_file=None, show=True):
    """
    Generates a beautiful interactive 3D surface plot of the Cassini ovoid peanut model
    with a waist and nothing else on it.
    
    Parameters:
        a : float, default 1.2602
            Parameter 'a' of the Cassini ovoid (neck-to-lobe geometry).
        b : float, default 1.3749
            Parameter 'b' of the Cassini ovoid (lobe size).
        colorscale : str, default 'brwnyl'
            Plotly colorscale for the peanut surface (e.g. 'brwnyl', 'peach', 'solar').
        output_file : str, optional
            Path to save the interactive HTML output file.
        show : bool, default True
            Whether to display the interactive plot in the notebook.
            
    Returns:
        plotly.graph_objects.Figure
    """
    import numpy as np
    import sys
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Error: 'plotly' package is required for interactive 3D plots. Please install it with 'pip install plotly'.")
        return None

    z_max = np.sqrt(a**2 + b**2)
    
    # Generate the Cassini ovoid meshgrid
    z_lin = np.linspace(-z_max * 0.999, z_max * 0.999, 150)
    theta_lin = np.linspace(0, 2 * np.pi, 150)
    Zm, Thm = np.meshgrid(z_lin, theta_lin)

    # Compute radius of Cassini ovoid (peanut with a waist)
    term1 = np.sqrt(b**4 + 4 * a**2 * Zm**2)
    r2 = term1 - Zm**2 - a**2
    Rm = np.sqrt(np.maximum(0, r2))
    
    Xm = Rm * np.cos(Thm)
    Ym = Rm * np.sin(Thm)

    # Plot surface
    fig = go.Figure(data=[go.Surface(
        x=Xm, y=Ym, z=Zm,
        colorscale=colorscale,
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5)
    )])

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.3, y=1.3, z=0.7)
    )

    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text="Interactive 3D Real Peanut Model (with Waist)",
            font=dict(size=20, color='white', family='Arial Black'),
            y=0.95, x=0.5, xanchor='center'
        ),
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
            yaxis=dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
            zaxis=dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
            aspectmode='data',
            camera=camera
        ),
        width=800,
        height=800
    )

    if output_file:
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.write_html(output_file)
        print(f"Saved interactive 3D peanut surface to {output_file}")

    if show:
        fig.show()

    return fig

