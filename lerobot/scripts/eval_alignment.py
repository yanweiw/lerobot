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

    # Create lists to store the mean and std statistics for each experiment
    mean_min_dists_l2 = []
    std_min_dists_l2 = []
    mean_avg_dists_l2 = []
    std_avg_dists_l2 = []
    mean_collision_rates = []
    std_collision_rates = []
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

        # Compute the mean and std statistics across trials for this experiment
        mean_min_dists_l2.append(np.mean(min_dists_l2))
        std_min_dists_l2.append(np.std(min_dists_l2))
        mean_avg_dists_l2.append(np.mean(avg_dists_l2))
        std_avg_dists_l2.append(np.std(avg_dists_l2))
        mean_collision_rates.append(np.mean(collision_rates))
        std_collision_rates.append(np.std(collision_rates))
        exps.append(exp)

    # The rest of your plotting code remains the same...
    # [Plotting code here...]

    # Print the mean and std of min dist, mean dist, and collision rate for each experiment
    print("Mean and Standard Deviation for Each Experiment:")
    for i, exp in enumerate(exps):
        print(f"\nExperiment: {exp}")
        print(f"  Min Distance L2: mean = {mean_min_dists_l2[i]:.3f}, std = {std_min_dists_l2[i]:.3f}")
        print(f"  Mean Distance L2: mean = {mean_avg_dists_l2[i]:.3f}, std = {std_avg_dists_l2[i]:.3f}")
        print(f"  Collision Rate: mean = {mean_collision_rates[i]:.3f}, std = {std_collision_rates[i]:.3f}")

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    plot_dist_vs_collisions()



