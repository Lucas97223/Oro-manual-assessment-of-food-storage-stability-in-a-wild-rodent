# Coherently organized Cassini ovoid 2D projection and patchiness mapping toolkit.

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

def get_geodesic_total_length(a=1.0, b=1.1):
    """Calculates the full tip-to-tip surface distance."""
    z_max = np.sqrt(a**2 + b**2)
    # Using small eps to avoid singularity at tips
    return abs(get_geodesic_profile_distance(-z_max + 1e-6, z_max - 1e-6, a, b))

