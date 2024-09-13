#!/usr/bin/env python
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

exp_tag = 'exp02'
exps = {
    'act_np': exp_tag + '_act_ph.json',
    'act_ph': exp_tag + '_act_ph.json',
    'act_op': exp_tag + '_act_op.json',
    'dp_np': exp_tag + '_dp_ph.json',
    'dp_ph': exp_tag + '_dp_ph.json',
    'dp_op': exp_tag + '_dp_op.json',
    'dp_bi': exp_tag + '_dp_bi.json',
    'dp_gd': exp_tag + '_dp_gd.json',
    'dp_rd': exp_tag + '_dp_rd.json',
}


def get_alignment_strategy(exp):
    if 'np' in exp:
        return 'np'
    elif 'ph' in exp:
        return 'ph'
    elif 'op' in exp:
        return 'op'
    elif 'bi' in exp:
        return 'bi'
    elif 'gd' in exp:
        return 'gd'
    elif 'rd' in exp:
        return 'rd'
    else:
        raise ValueError(f'Unknown alignment strategy for experiment {exp}')

exp_dict = {}
for exp, file in exps.items():
    with open(file, 'r', buffering=1) as f:
        f.seek(0)
        trials = [json.loads(trial) for trial in f]
        exp_dict[exp] = trials

# assert trials are of same length for all exp keys
for exp, trials in exp_dict.items():
    assert len(trials) == len(exp_dict['act_np'])

# each trial is a dictionary with the following keys
# trial = {
#     "trial_idx" # int
#     "agent_pos" # int [x, y] 
#     "guide" # nested list of shape (guide_horizon, action_dim)
#     "pred_traj" # nested list of shape (batch, pred_horizon, action_dim)
#     "collisions" # list of bool
# }

def l2_dist(samples, guide):
    # samples: (B, pred_horizon, action_dim)
    # guide: (guide_horizon, action_dim)
    assert samples.shape[2] == 2 and guide.shape[1] == 2
    indices = np.linspace(0, guide.shape[0]-1, samples.shape[1], dtype=int)
    guide = np.expand_dims(guide[indices], axis=0) # (1, pred_horizon, action_dim)
    guide = np.tile(guide, (samples.shape[0], 1, 1)) # (B, pred_horizon, action_dim)
    dist = np.linalg.norm(samples[:, :] - guide[:, :], axis=2, ord=2).mean(axis=1) # (B,)
    # # sort the predictions based on scores, from smallest to largest, so that larger scores will be drawn on top
    sort_idx = np.argsort(dist)
    dist = dist[sort_idx]
    samples = samples[sort_idx]  
    return samples, dist

def dtw_dist(samples, guide):
    # samples: (B, pred_horizon, action_dim)
    # guide: (guide_horizon, action_dim)
    assert samples.shape[2] == 2 and guide.shape[1] == 2
    dist = []
    for sample in samples:
        distance, _ = fastdtw(sample, guide, dist=euclidean)
        dist.append(distance)
    dist = np.array(dist)
    sort_idx = np.argsort(dist)
    dist = dist[sort_idx]
    samples = samples[sort_idx]
    return samples, dist

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

def plot_dist_vs_collisions():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))  # 1x2 grid for the two plots
    
    # Create lists to store the mean statistics for each experiment
    mean_min_dists_l2 = []
    mean_avg_dists_l2 = []
    mean_collision_rates = []
    exps = []
    alignment_strategies = []
    markers = []

    for exp, trials in exp_dict.items():
        min_dists_l2 = []
        avg_dists_l2 = []
        collision_rates = []
        
        # Gather data for each trial in the current experiment
        for trial in trials:
            pred_traj = np.array(trial['pred_traj'])
            guide = np.array(trial['guide'])
            
            # Calculate l2 distances
            pred_traj_l2, dist_l2 = l2_dist(pred_traj, guide)
            if 'np' in exp:
                # Randomly choosing one as min
                rand_idx = np.random.randint(0, len(pred_traj_l2))
                min_dist_l2 = dist_l2[rand_idx]
            else:   
                min_dist_l2 = dist_l2.min()
            avg_dist_l2 = dist_l2.mean()
            min_dists_l2.append(min_dist_l2)
            avg_dists_l2.append(avg_dist_l2)
            
            # Calculate collision rates
            collision_rate = np.mean(trial['collisions'])
            collision_rates.append(collision_rate)
            
        # Extract alignment strategy and policy type from experiment name
        alignment_strategy = get_alignment_strategy(exp)
        
        # Add to lists
        alignment_strategies.append(alignment_strategy)
        markers.append(exp)
        
        # Compute the mean statistics across trials for this experiment
        mean_min_dists_l2.append(np.mean(min_dists_l2))
        mean_avg_dists_l2.append(np.mean(avg_dists_l2))
        mean_collision_rates.append(np.mean(collision_rates))
        exps.append(exp)

    # Convert to arrays for plotting
    mean_min_dists_l2 = np.array(mean_min_dists_l2)
    mean_avg_dists_l2 = np.array(mean_avg_dists_l2)
    mean_collision_rates = np.array(mean_collision_rates)
    alignment_strategies = np.array(alignment_strategies)
    markers = np.array(markers)

    # Set marker styles and colors
    marker_styles = {}
    alignment_color_map = {}
    color_map = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22"
    ]
    for exp in exp_dict.keys():
        if 'dp' in exp:
            marker_styles[exp] = 'o'
        elif 'act' in exp:
            marker_styles[exp] = 'D'
        else:
            raise ValueError(f'Unknown policy type for experiment {exp}')
        if 'act_np' in exp:
            alignment_color_map[exp] = color_map[0]
        elif 'act_op' in exp:
            alignment_color_map[exp] = color_map[1]
        elif 'act_ph' in exp:
            alignment_color_map[exp] = color_map[3]
        elif 'dp_np' in exp:
            alignment_color_map[exp] = color_map[2]
        elif 'dp_ph' in exp:
            alignment_color_map[exp] = color_map[4]
        elif 'dp_op' in exp:
            alignment_color_map[exp] = color_map[5]
        elif 'dp_bi' in exp:
            alignment_color_map[exp] = color_map[6]
        elif 'dp_gd' in exp:
            alignment_color_map[exp] = color_map[7]
        elif 'dp_rd' in exp:
            alignment_color_map[exp] = color_map[8]
        else:
            raise ValueError(f'Unknown alignment strategy for experiment {exp}')

    # Replace zero values with a small positive number for log scale plotting
    dist_normalization = 900
    adjusted_collision_rates = [x if x > 0 else 1e-3 for x in mean_collision_rates] 
    adjusted_min_dists_l2 = [y / dist_normalization if y / dist_normalization > 0 else 1e-3 for y in mean_min_dists_l2] 
    adjusted_avg_dists_l2 = [y / dist_normalization if y / dist_normalization > 0 else 1e-3 for y in mean_avg_dists_l2] 

    # Add a small jitter to the x values and y values to avoid overlapping points
    np.random.seed(1)
    adjusted_collision_rates = np.array(adjusted_collision_rates) + np.random.normal(0, 0.002, len(adjusted_collision_rates))
    adjusted_min_dists_l2 = np.array(adjusted_min_dists_l2) + np.random.normal(0, 0.002, len(adjusted_min_dists_l2))
    adjusted_avg_dists_l2 = np.array(adjusted_avg_dists_l2) + np.random.normal(0, 0.002, len(adjusted_avg_dists_l2))

    # Plot using Seaborn with increased marker size and specified palette
    sns.scatterplot(
        x=adjusted_collision_rates,  
        y=adjusted_min_dists_l2, 
        hue=exps, 
        style=markers, 
        markers=marker_styles,
        palette=alignment_color_map,  # Added palette parameter
        s=100,  # Increased marker size
        ax=ax[0], 
        legend=False  # Removed legend
    )
    sns.scatterplot(
        x=adjusted_collision_rates, 
        y=adjusted_avg_dists_l2, 
        hue=exps, 
        style=markers, 
        markers=marker_styles,
        palette=alignment_color_map,  # Added palette parameter
        s=100,  # Increased marker size
        ax=ax[1], 
        legend=False  # Removed legend
    )
    
    # For each exp, print the mean min dist and collision rate
    for exp, mean_min_dist, mean_collision_rate in zip(exps, mean_min_dists_l2, mean_collision_rates):
        print(f'{exp}: Mean Min Dist: {mean_min_dist:.3f}, Collision Rate: {mean_collision_rate:.3f}')

    # Set log scale for both axes
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[0].set_title('Min Distance vs Collision Rate')
    ax[0].set_xlabel('Collision Rate')
    ax[0].set_ylabel('Min Distance (Normalized L2)')
    ax[0].set_xlim(-0.01, 0.38)
    ax[0].set_ylim(0.08, 0.33)

    ax[1].set_title('Mean Distance vs Collision Rate')
    ax[1].set_xlabel('Collision Rate')
    ax[1].set_ylabel('Mean Distance (Normalized L2)')
    ax[1].set_xlim(-0.01, 0.38)
    ax[1].set_ylim(0.08, 0.33)

    # Annotate each point with experiment labels
    method_names = {
        'act_np': 'ACT: No Perturbation',
        'act_ph': 'ACT: Post-Hoc Ranking',
        'act_op': 'ACT: State Perturbation',
        'dp_np': 'DP: No Perturbation',
        'dp_ph': 'DP: Post-Hoc Ranking',
        'dp_op': 'DP: State Perturbation',
        'dp_bi': 'DP: Biased Initialization',
        'dp_gd': 'DP: Vanilla Diffusion',
        'dp_rd': 'DP: Stochastic Sampling',
    }
    method_offsets = {
        'act_np': (15, 15),
        'act_ph': (-20, 0),
        'act_op': (-5, -15),
        'dp_np': (15, 0),
        'dp_ph': (-5, 15),
        'dp_op': (15, -5),
        'dp_bi': (15, -15),
        'dp_gd': (15, 15),
        'dp_rd': (0, -15),
    }

    for i, txt in enumerate(exps):
        x = adjusted_collision_rates[i]
        y = adjusted_min_dists_l2[i]
        # Determine offset based on position
        offset_x, offset_y = method_offsets.get(txt, (0, 0))
        ha = 'left' if x < 0.1 else 'right'
        ax[0].annotate(
            method_names.get(txt, txt),
            (x, y), 
            textcoords="offset points", 
            xytext=(offset_x, offset_y), 
            ha=ha, 
            fontsize=10,
            color=alignment_color_map.get(txt, 'black')  # Set text color to match marker
        )
    
    method_offsets2 = {
        'act_np': (35, 10),
        'act_ph': (-15, -5),
        'act_op': (0, -20),
        'dp_np': (-10, 10),
        'dp_ph': (-15, -20),
        'dp_op': (10, 10),
        'dp_bi': (15, -10),
        'dp_gd': (15, 0),
        'dp_rd': (-5, -20),
    }    

    for i, txt in enumerate(exps):
        x = adjusted_collision_rates[i]
        y = adjusted_avg_dists_l2[i]
        # Determine offset based on position
        offset_x, offset_y = method_offsets2.get(txt, (0, 0))
        ha = 'left' if x < 0.1 else 'right'
        ax[1].annotate(
            method_names.get(txt, txt),
            (x, y), 
            textcoords="offset points", 
            xytext=(offset_x, offset_y), 
            ha=ha, 
            fontsize=10,
            color=alignment_color_map.get(txt, 'black')  # Set text color to match marker
        )

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    plot_dist_vs_collisions()



