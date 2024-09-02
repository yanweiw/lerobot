import sys, os
import numpy as np
import pygame
import torch
import argparse
import matplotlib.pyplot as plt
import einops
from pathlib import Path
from huggingface_hub import snapshot_download
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.utils.utils import seeded_context, init_hydra_config
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.factory import make_dataset
from scipy.special import softmax
import json

fps = 10
batch_size = 32  # visualize this many trajectories per inference
renderer_background = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)

def similarity_score(samples, guide=None):
    # samples: (B, pred_horizon, action_dim)
    # guide: (guide_horizon, action_dim)
    if guide is None:
        return np.zeros(samples.shape[0])
    # print('caclulating similarity score, samples shape:', samples.shape, 'guide shape:', guide.shape)
    assert samples.shape[2] == 2 and guide.shape[1] == 2
    indices = np.linspace(0, guide.shape[0]-1, samples.shape[1], dtype=int)
    guide = np.expand_dims(guide[indices], axis=0) # (1, pred_horizon, action_dim)
    guide = np.tile(guide, (samples.shape[0], 1, 1)) # (B, pred_horizon, action_dim)
    scores = np.linalg.norm(samples[:, 1:] - guide[:, 1:], axis=2).mean(axis=1) # (B,)
    scores = 1 - scores / (scores.max() + 1e-6) # normalize
    # scores = scores.max() / (scores + 1e-6) 
    temperature = 20
    scores = softmax(scores*temperature)
    # print only 3 decimal places without scientific notation
    # print('scores:', [f'{score:.3f}' for score in scores])
    
    # normalize the score to be between 0 and 1
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores

def xy2gui(unnormed_xy):
    maze_shape = renderer_background.shape
    xy = unnormed_xy + 0.5  # Adjust normalization as necessary
    x = xy[0] * size[0] / (maze_shape[0])
    y = xy[1] * size[1] / (maze_shape[1])
    return np.array([y, x], dtype=float)

def gui2xy(gui):
    maze_shape = renderer_background.shape
    x = gui[1] / size[0] * maze_shape[0] - 0.5
    y = gui[0] / size[1] * maze_shape[1] - 0.5
    xy = np.array([x, y], dtype=float)
    return xy

def infer_target(policy_wrapped, obj_hist_xy, timestamp, batch_size=50, guide=None):
    # obj_hist_xy = obj_hist_xy[0]
    obj_hist_xy = np.vstack(obj_hist_xy)
    obs_batch = {
        "observation.state": einops.repeat(
            torch.from_numpy(obj_hist_xy).float().cuda(), "t d -> b t d", b=batch_size
        )
    }
    obs_batch["observation.environment_state"] = einops.repeat(
        torch.from_numpy(obj_hist_xy).float().cuda(), "t d -> b t d", b=batch_size
        )    
    if guide is not None:
        guide = torch.from_numpy(guide).float().cuda()
    with torch.autocast(device_type="cuda"), seeded_context(0):     
        # actions = policy_wrapped.provide_observation_get_actions(obs_batch, timestamp, timestamp).cpu().numpy().transpose(1, 0, 2)
        actions = policy_wrapped.policy.run_inference(obs_batch, guide=guide).cpu().numpy()
    return actions

def generate_time_color_map(num_steps):
    # Same function as before
    cmap = plt.get_cmap('rainbow')
    values = np.linspace(0, 1, num_steps)
    colors = cmap(values)
    return colors

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', type=str, help="Policy name")
    args = parser.parse_args()
    
    # Load policy from new codebase
    device = torch.device("cuda")

    pretrained_policy_path = Path(os.path.join(args.checkpoint, "pretrained_model"))  # Update path as necessary

    if args.policy in ["diffusion", "dp"]:
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    if args.policy in ["act"]:
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.cuda()
    policy.eval()
    policy_wrapped = PolicyRolloutWrapper(policy, fps=fps)

    # Initialize Pygame
    pygame.init()
    size = (1000, 1000)

    # Set colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GRAY = (150, 150, 150)  # New color for the drawing

    # Create the screen
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Reach Goal")
    clock = pygame.time.Clock()

    obj_pos = np.array([0, 0])  # Initialize the position of the red dot
    draw_traj = []
    drawing = False # whether the user is still drawing
    keep_drawing = False # whether delete the current drawing and start a new one
    obj_history_xy = []

    running = True
    t = 0
    file_name = "save_maze_interact.json"

    # Open the file in append mode
    file = open(file_name, "r+", buffering=1)

    def load_position_trajectory(index):
        # Move the file pointer to the beginning
        file.seek(0)
        
        # Load all entries as a list of dictionaries
        entries = [json.loads(line) for line in file]
        
        # Return the i-th item, if it exists
        if index < len(entries):
            return entries[index]
        else:
            return None  # Or raise an exception if the index is out of range

    traj_point = 0
    while running:
        mouse_pos = np.array(pygame.mouse.get_pos())

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                if not drawing:
                    drawing = True
                    draw_traj = []
                draw_traj.append(mouse_pos)
            else:    
                if drawing: # mouse released
                    drawing = False
                    keep_drawing = True

        
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    print("Hey,loading point!")
                    entry =load_position_trajectory(traj_point)
                    print(entry)
                    guide = np.array(entry['data'])
                    mouse_pos = xy2gui(np.array(entry['data'][0]))

        if keep_drawing:
            for point in draw_traj:
                pygame.draw.circle(screen, GRAY, (int(point[0]), int(point[1])), 3)
            # Check if mouse returns to the agent's location
            if np.linalg.norm(mouse_pos - obj_pos) < 10:  # Threshold distance to reactivate the agent
                keep_drawing = False
                draw_traj = []

        if not drawing and not keep_drawing: # delete previous conditioning and start a new one 
            # Real-time inference based on mouse position
            obj_history_xy.append(gui2xy(mouse_pos))
            if len(obj_history_xy) < policy.config.n_obs_steps:  # Copy the last element to fill the history
                obj_history_xy = obj_history_xy + [obj_history_xy[-1]] * (policy.config.n_obs_steps - len(obj_history_xy))
            # Keep only the last n_obs_steps    
            if len(obj_history_xy) > policy.config.n_obs_steps:
                obj_history_xy = obj_history_xy[-policy.config.n_obs_steps:]

            # Update the red dot position to match the mouse position
            obj_pos = mouse_pos

        if not drawing: 
            if len(draw_traj) > 0:
                guide = np.array([gui2xy(point) for point in draw_traj])
            else:
                guide = None
            
            policy_wrapped.reset()
            
            unnormed_obs = infer_target(policy_wrapped, np.array(obj_history_xy), t, batch_size=batch_size, guide=guide)

            scores = similarity_score(unnormed_obs, guide=guide)

        # Clear the screen
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(renderer_background[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, size)
        screen.blit(surface, (0, 0))

        # Draw future predictions with time-based colors as lines
        # print('shape of unnormed_obs:', unnormed_obs.shape, 'scores shape:', scores.shape)
        assert scores.shape[0] == unnormed_obs.shape[0]
        # sort the predictions based on scores, from smallest to largest, so that larger scores will be drawn on top
        sort_idx = np.argsort(scores)
        unnormed_obs = unnormed_obs[sort_idx]
        scores = scores[sort_idx]

        # use score to weight the line thickness
        if unnormed_obs.size > 0:  # Check if normed_obs is not empty
            time_colors = generate_time_color_map(unnormed_obs.shape[1])
            for i, pred in enumerate(unnormed_obs):
                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    # mixing 1/2 color with whiteness depending on scores
                    if scores[-1] < 0.1: # if the highest score is less than 0.1, use the default color
                        color = color
                    else:
                        color = color//3 + (color//3*2) * scores[i] + 255//3*2 * (1-scores[i])
                    start_pos = xy2gui(pred[step_idx])
                    end_pos = xy2gui(pred[step_idx + 1])
                    pygame.draw.line(screen, color, start_pos, end_pos, int(3 + 20*scores[i]))
                    # pygame.draw.circle(screen, color, start_pos, 3)


        # Draw the red dot at the current mouse position (starting point for the predictions)
        pygame.draw.circle(screen, RED, (int(obj_pos[0]), int(obj_pos[1])), 20)

        # Draw the recorded drawing
        if drawing or keep_drawing:
            for point in draw_traj:
                pygame.draw.circle(screen, GRAY, (int(point[0]), int(point[1])), 5)

        pygame.display.flip()
        clock.tick(30)
        t += 1/fps

    pygame.quit()
