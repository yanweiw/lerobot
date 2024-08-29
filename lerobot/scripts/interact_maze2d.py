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
    
    with torch.inference_mode(), torch.autocast(device_type="cuda"), seeded_context(0):     
        # actions = policy_wrapped.provide_observation_get_actions(obs_batch, timestamp, timestamp)
        actions = policy_wrapped.policy.run_inference(obs_batch)
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
    drawing = False
    drawing_finished = False
    obj_history_xy = []

    running = True
    t = 0
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                if not drawing:
                    drawing = True
                    draw_traj = []
                draw_traj.append(np.array(pygame.mouse.get_pos()))
            else:    
                if drawing: # mouse released
                    drawing = False
                    drawing_finished = True

        mouse_pos = np.array(pygame.mouse.get_pos())

        if drawing_finished:
            # Keep the drawing on the screen
            for point in draw_traj:
                pygame.draw.circle(screen, GRAY, (int(point[0]), int(point[1])), 3)
            # Check if mouse returns to the agent's location
            if np.linalg.norm(mouse_pos - obj_pos) < 10:  # Threshold distance to reactivate the agent
                drawing_finished = False
                draw_traj = []

        if not drawing and not drawing_finished:
            # Real-time inference based on mouse position
            obj_history_xy.append(gui2xy(mouse_pos))
            if len(obj_history_xy) < policy.config.n_obs_steps:  # Copy the last element to fill the history
                obj_history_xy = obj_history_xy + [obj_history_xy[-1]] * (policy.config.n_obs_steps - len(obj_history_xy))
            # Keep only the last n_obs_steps    
            if len(obj_history_xy) > policy.config.n_obs_steps:
                obj_history_xy = obj_history_xy[-policy.config.n_obs_steps:]
            
            policy_wrapped.reset()
            unnormed_obs = infer_target(policy_wrapped, np.array(obj_history_xy), t, batch_size=batch_size)
            # switch axis of unnormed_obs from (S, B, 2) to (B, S, 2)
            unnormed_obs = np.swapaxes(unnormed_obs, 0, 1)
            
            # Update the red dot position to match the mouse position
            obj_pos = mouse_pos

        # Clear the screen
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(renderer_background[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, size)
        screen.blit(surface, (0, 0))

        # Draw future predictions with time-based colors as lines
        if unnormed_obs.size > 0:  # Check if normed_obs is not empty
            time_colors = generate_time_color_map(unnormed_obs.shape[1])
            for pred in unnormed_obs:
                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    start_pos = xy2gui(pred[step_idx])
                    end_pos = xy2gui(pred[step_idx + 1])
                    pygame.draw.line(screen, color, start_pos, end_pos, 3)
                    # pygame.draw.circle(screen, color, start_pos, 3)


        # Draw the red dot at the current mouse position (starting point for the predictions)
        pygame.draw.circle(screen, RED, (int(obj_pos[0]), int(obj_pos[1])), 8)

        # Draw the recorded drawing
        if drawing or drawing_finished:
            for point in draw_traj:
                pygame.draw.circle(screen, GRAY, (int(point[0]), int(point[1])), 5)


        pygame.display.flip()
        clock.tick(30)
        t += 1/fps

    pygame.quit()
