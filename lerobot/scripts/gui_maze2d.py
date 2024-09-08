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
maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)

offset = 0.5  # offset to put the object in the center of the cell in maze background
# GUI x coord 0 -> gui_size[0] #1200
# GUI y coord 0 
#         |
#         v
#       gui_size[1] #900
# xy is in the same coordinate system as the background
# bkg y coord 0 -> maze_shape[1] #12
# bkg x coord 0
#         |
#         v
#       maze_shape[0] #9
                

def check_collision(xy_traj):
    """
    input xy is in the same coordinate system as the maze background
    """
    assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
    batch_size, num_steps, _ = xy_traj.shape
    xy_traj = xy_traj.reshape(-1, 2)  # Shape (batch_size * num_steps, 2)

    # Convert GUI coordinates to maze coordinates for all trajectories
    xy_traj = np.clip(xy_traj, [0, 0], [maze.shape[0] - 1, maze.shape[1] - 1])  # Clip values to valid maze bounds

    maze_x = np.round(xy_traj[:, 0]).astype(int)  # Convert to integer indices
    maze_y = np.round(xy_traj[:, 1]).astype(int)

    # Check for collisions by indexing the background
    collisions = maze[maze_x, maze_y] 

    # Reshape the result back to (batch_size, num_steps)
    collisions = collisions.reshape(batch_size, num_steps)

    # If any point in a trajectory collides, mark the entire trajectory as colliding
    return np.any(collisions, axis=1)

def blend_with_white(color, factor=0.5):
    """
    Blends the input color with white
    """
    white = np.array([255, 255, 255])
    blended_color = (1 - factor) * np.array(color) + factor * white
    return blended_color.astype(int)

def report_collision_percentage(collisions):
    num_trajectories = collisions.shape[0]
    num_collisions = np.sum(collisions)
    collision_percentage = (num_collisions / num_trajectories) * 100
    print(f"{num_collisions}/{num_trajectories} trajectories are in collision ({collision_percentage:.2f}%).")
    return collision_percentage

def xy2gui(xy):
    xy = xy + offset  # Adjust normalization as necessary
    x = xy[0] * gui_size[1] / (maze.shape[0])
    y = xy[1] * gui_size[0] / (maze.shape[1])
    return np.array([y, x], dtype=float)

def gui2xy(gui):
    x = gui[1] / gui_size[1] * maze.shape[0] - offset
    y = gui[0] / gui_size[0] * maze.shape[1] - offset
    return np.array([x, y], dtype=float)

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

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', type=str, help="Policy name")
    args = parser.parse_args()
    
    # Load policy from new codebase
    device = torch.device("cuda")

    pretrained_policy_path = Path(os.path.join(args.checkpoint, "pretrained_model"))  # Update path as necessary
    # hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"))
    # policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)
    # dataset = make_dataset(hydra_cfg)
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
    gui_size = (1200, 900)

    # Set colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    # Create the screen
    screen = pygame.display.set_mode(gui_size)
    pygame.display.set_caption("Maze")
    clock = pygame.time.Clock()

    obj_pos = np.array([0, 0]).reshape(1, 2)  # Initialize the position of the red dot

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

        # Check if the mouse is in collision
        mouse_in_collision = check_collision(obj_history_xy[-1].reshape(1, 1, 2))[0]

        # Change mouse color based on collision status
        if mouse_in_collision:
            # blend red with white
            mouse_color = (255, 0, 0)
            mouse_color = blend_with_white(mouse_color, 0.8)
        else:
            mouse_color = (255, 0, 0)  # Red if not in collision
        
        policy_wrapped.reset()
        xy_pred = infer_target(policy_wrapped, obj_history_xy, t, batch_size=batch_size)

        # Update the mouse dot position
        obj_pos = mouse_pos

        # Clear the screen
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(maze[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, gui_size)
        screen.blit(surface, (0, 0))

        # Draw future predictions with time-based colors
        if xy_pred.size > 0:  # Check if pred is not empty
            time_colors = generate_time_color_map(xy_pred.shape[1])

            # Check for collisions in parallel
            collisions = check_collision(xy_pred)

            # Report the percentage of trajectories in collision
            report_collision_percentage(collisions)

            for idx, pred in enumerate(xy_pred):
                collision_detected = collisions[idx]

                # Blend the colors with white only if the trajectory has any collisions
                whiteness_factor = 0.8 if collision_detected else 0.0  # Make the trajectory paler if there's a collision
                circle_size = 10 if collision_detected else 5

                for step_idx in range(len(pred) - 1):
                    # Get the time-based color and adjust it
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    color = blend_with_white(color, whiteness_factor)

                    # Draw the trajectory step
                    start_pos = xy2gui(pred[step_idx])
                    end_pos = xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(screen, color, start_pos, circle_size)

        # Draw the mouse dot with the appropriate color based on collision detection
        pygame.draw.circle(screen, mouse_color, (int(obj_pos[0]), int(obj_pos[1])), 20)

        pygame.display.flip()
        clock.tick(30)
        t += 1/fps

    pygame.quit()
