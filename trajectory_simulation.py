# Coherently organized trajectory tracking and simulation strategies on curved surfaces.

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

