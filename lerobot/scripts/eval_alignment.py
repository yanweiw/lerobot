#!/usr/bin/env python
import json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

exp_tag = 'exp00'
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
    # scores = 1 - scores / (scores.max() + 1e-6) # normalize
    # temperature = 20
    # scores = softmax(scores*temperature)
    # # print('scores:', [f'{score:.3f}' for score in scores])
    # # normalize the score to be between 0 and 1
    # scores = (scores - scores.min()) / (scores.max() - scores.min())
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


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_dist_vs_collisions():
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 grid for the four plots
    
    # Create lists to store the mean statistics for each experiment for both l2 and dtw
    mean_min_dists_l2 = []
    mean_avg_dists_l2 = []
    mean_min_dists_dtw = []
    mean_avg_dists_dtw = []
    mean_collision_rates = []
    exps = []
    alignment_strategies = []  # To store alignment strategies for color encoding
    markers = []               # To store full marker names like 'act_op', 'dp_op', etc.

    for exp, trials in exp_dict.items():
        min_dists_l2 = []
        avg_dists_l2 = []
        min_dists_dtw = []
        avg_dists_dtw = []
        collision_rates = []
        
        # Gather data for each trial in the current experiment
        for trial in trials:
            pred_traj = np.array(trial['pred_traj'])
            guide = np.array(trial['guide'])
            
            # Calculate l2 distances
            pred_traj_l2, dist_l2 = l2_dist(pred_traj, guide)
            if 'np' in exp:
                # randomly choosing one as min
                rand_idx = np.random.randint(0, len(pred_traj_l2))
                min_dist_l2 = dist_l2[rand_idx]
            else:   
                min_dist_l2 = dist_l2.min()
            avg_dist_l2 = dist_l2.mean()
            min_dists_l2.append(min_dist_l2)
            avg_dists_l2.append(avg_dist_l2)
            
            # Calculate dtw distances
            pred_traj_dtw, dist_dtw = dtw_dist(pred_traj, guide)
            if 'np' in exp:
                # randomly choosing one as min
                rand_idx = np.random.randint(0, len(pred_traj_dtw))
                min_dist_dtw = dist_dtw[rand_idx]
            min_dist_dtw = dist_dtw.min()
            avg_dist_dtw = dist_dtw.mean()
            min_dists_dtw.append(min_dist_dtw)
            avg_dists_dtw.append(avg_dist_dtw)
            
            # Calculate collision rates
            collision_rate = np.mean(trial['collisions'])
            collision_rates.append(collision_rate)
            
        # Extract alignment strategy and policy type from experiment name
        alignment_strategy = get_alignment_strategy(exp)  # e.g., 'op', 'np', 'bi'
        # policy_type = exp.split('_')[0]         # e.g., 'act', 'dp'
        
        # Add to lists
        alignment_strategies.append(alignment_strategy)
        markers.append(exp)  # Full name like 'act_op', 'dp_bi', etc.
        
        # Compute the mean statistics across trials for this experiment
        mean_min_dists_l2.append(np.mean(min_dists_l2))
        mean_avg_dists_l2.append(np.mean(avg_dists_l2))
        mean_min_dists_dtw.append(np.mean(min_dists_dtw))
        mean_avg_dists_dtw.append(np.mean(avg_dists_dtw))
        mean_collision_rates.append(np.mean(collision_rates))
        exps.append(exp)

    # Convert to arrays for plotting
    mean_min_dists_l2 = np.array(mean_min_dists_l2)
    mean_avg_dists_l2 = np.array(mean_avg_dists_l2)
    mean_min_dists_dtw = np.array(mean_min_dists_dtw)
    mean_avg_dists_dtw = np.array(mean_avg_dists_dtw)
    mean_collision_rates = np.array(mean_collision_rates)
    alignment_strategies = np.array(alignment_strategies)  # for color
    markers = np.array(markers)                            # for markers

    # set marker styles cirlce for dp and diamond for act
    # set different colors for different alignment strategies
    marker_styles = {}
    alignment_color_map = {}
    for exp in exp_dict.keys():
        if 'dp' in exp:
            marker_styles[exp] = 'o'
        elif 'act' in exp:
            marker_styles[exp] = 'D'
        else:
            raise ValueError(f'Unknown policy type for experiment {exp}')
        if 'np' in exp:
            alignment_color_map[exp] = 'blue'
        elif 'op' in exp:
            alignment_color_map[exp] = 'green'
        elif 'ph' in exp:
            alignment_color_map[exp] = 'red'
        elif 'bi' in exp:
            alignment_color_map[exp] = 'purple'
        elif 'rd' in exp:
            alignment_color_map[exp] = 'orange'
        elif 'gd' in exp:
            alignment_color_map[exp] = 'cyan'
        else:
            raise ValueError(f'Unknown alignment strategy for experiment {exp}')

    # Map the alignment strategy to colors manually
    # alignment_palette = [alignment_color_map[e] for e in exps]

    # Define marker styles based on 'act' and 'dp'
    # marker_styles = {
    #     'act': 'D',  # Diamond for 'act'
    #     'dp': 'o'    # Circle for 'dp'
    # }

    # Define the hue grouping based on alignment strategies
    hue_group = [exp.split('_')[1] for exp in exp_dict.keys()]  # This ensures same hue for the same alignment strategy

    # Use 'dp' and 'act' to control marker shapes
    style_group = [exp.split('_')[0] for exp in exp_dict.keys()]  # 'dp' or 'act' for marker shapes

    # Plot using Seaborn with 'hue' for color and 'style' for different markers
    sns.scatterplot(x=mean_collision_rates + np.random.normal(0, 0.01, size=len(mean_collision_rates)), y=mean_min_dists_l2, hue=exps, style=markers,
                    markers=marker_styles, ax=ax[0, 0], legend='full')
    sns.scatterplot(x=mean_collision_rates + np.random.normal(0, 0.01, size=len(mean_collision_rates)), y=mean_avg_dists_l2, hue=exps, style=markers,
                    markers=marker_styles, ax=ax[0, 1], legend='full')

    # Set log scale for both axes
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')

    ax[0, 0].set_title('L2: Mean Min Distance vs Mean Collision Rate')
    ax[0, 0].set_xlabel('Mean Collision Rate')
    ax[0, 0].set_ylabel('Mean Min Distance (L2)')
    
    ax[0, 1].set_title('L2: Mean Avg Distance vs Mean Collision Rate')
    ax[0, 1].set_xlabel('Mean Collision Rate')
    ax[0, 1].set_ylabel('Mean Avg Distance (L2)')

    # Plot for DTW distance
    sns.scatterplot(x=mean_collision_rates + np.random.normal(0, 0.01, size=len(mean_collision_rates)), y=mean_min_dists_dtw, hue=exps, style=markers,
                    markers=marker_styles, ax=ax[1, 0], legend='full')
    sns.scatterplot(x=mean_collision_rates + np.random.normal(0, 0.01, size=len(mean_collision_rates)), y=mean_avg_dists_dtw, hue=exps, style=markers,
                    markers=marker_styles, ax=ax[1, 1], legend='full')

    # Set log scale for both axes
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')

    ax[1, 0].set_title('DTW: Mean Min Distance vs Mean Collision Rate')
    ax[1, 0].set_xlabel('Mean Collision Rate')
    ax[1, 0].set_ylabel('Mean Min Distance (DTW)')
    
    ax[1, 1].set_title('DTW: Mean Avg Distance vs Mean Collision Rate')
    ax[1, 1].set_xlabel('Mean Collision Rate')
    ax[1, 1].set_ylabel('Mean Avg Distance (DTW)')

    plt.legend(title='Alignment Strategy and Marker')
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    plot_dist_vs_collisions()



