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
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.utils.utils import seeded_context, init_hydra_config
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.factory import make_dataset

fps = 10
batch_size = 3  # visualize this many trajectories per inference
renderer_background = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)

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

def infer_target(policy_wrapped, obj_hist_xy, timestamp, batch_size=50):
    obj_hist_xy = obj_hist_xy[0]
    obs_batch = {
        "observation.state": einops.repeat(
            torch.from_numpy(obj_hist_xy).float().cuda(), "d -> b d", b=batch_size
        )
    }
    obs_batch["observation.environment_state"] = einops.repeat(
        torch.from_numpy(obj_hist_xy).float().cuda(), "d -> b d", b=batch_size
        )    
    
    with torch.inference_mode(), torch.autocast(device_type="cuda"), seeded_context(0):     
        actions = policy_wrapped.provide_observation_get_actions(obs_batch, timestamp, timestamp)
    actions = actions.cpu().numpy()  # (S, B, 2)
    # reshape to (B, S, 2)
    return actions.transpose(1, 0, 2)

def generate_time_color_map(num_steps):
    # Same function as before
    cmap = plt.get_cmap('rainbow')
    values = np.linspace(0, 1, num_steps)
    colors = cmap(values)
    return colors

if __name__ == "__main__":

    # Load policy from new codebase
    device = torch.device("cuda")

    pretrained_policy_path = Path("/mnt/data/lerobot/outputs/2024-08-27/12-10-04_maze2d_diffusion_default/checkpoints/130000/pretrained_model")  # Update path as necessary
    # hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"))
    # policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)
    # dataset = make_dataset(hydra_cfg)
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.config.noise_scheduler_type = "DDIM"
    policy.diffusion.num_inference_steps = 10
    policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    policy.cuda()
    policy.eval()
    policy_wrapped = PolicyRolloutWrapper(policy, fps=fps)

    # Initialize Pygame
    pygame.init()
    size = (1000, 1000)

    # Set colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    # Create the screen
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Reach Goal")
    clock = pygame.time.Clock()

    obj_pos = np.array([0, 0])  # Initialize the position of the red dot

    running = True
    t = 0
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # Real-time inference based on mouse position
        mouse_pos = np.array(pygame.mouse.get_pos())
        obj_history_xy = [gui2xy(mouse_pos)]
        
        policy_wrapped.reset()
        unnormed_obs = infer_target(policy_wrapped, np.array(obj_history_xy), t, batch_size=batch_size)

        # Update the red dot position to match the mouse position
        obj_pos = mouse_pos

        # Clear the screen
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(renderer_background[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, size)
        screen.blit(surface, (0, 0))

        # # create a random batch (batch size 50) of 16 lenth trajectory from dataset as unnormed_obs
        # unnormed_obs = np.zeros((batch_size, 64, 2))
        # for i in range(batch_size):
        #     rand_idx = np.random.randint(0, len(dataset.hf_dataset['action'])-64)
        #     unnormed_obs[i] = np.array(dataset.hf_dataset['action'][rand_idx:rand_idx+64])        
        # print('shape of unnormed_obs:', unnormed_obs.shape)

        # Draw future predictions with time-based colors
        if unnormed_obs.size > 0:  # Check if normed_obs is not empty
            time_colors = generate_time_color_map(unnormed_obs.shape[1])
            for pred in unnormed_obs:
                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    start_pos = xy2gui(pred[step_idx])
                    end_pos = xy2gui(pred[step_idx + 1])
                    # pygame.draw.line(screen, color, start_pos, end_pos, 3)
                    # draw a circle instead of line
                    pygame.draw.circle(screen, color, start_pos, 3)

        # Draw the red dot at the current mouse position (starting point for the predictions)
        pygame.draw.circle(screen, RED, (int(obj_pos[0]), int(obj_pos[1])), 8)

        pygame.display.flip()
        clock.tick(30)
        t += 1/fps

    pygame.quit()
