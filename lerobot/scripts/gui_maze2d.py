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

class MazeEnv:
    def __init__(self):
        self.maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
        self.gui_size = (1200, 900)
        self.fps = 10
        self.batch_size = 32        
        self.offset = 0.5  # Offset to put object in the center of the cell

        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.mouse_color = self.RED

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.gui_size)
        pygame.display.set_caption("Maze")
        self.clock = pygame.time.Clock()
        self.agent_gui_pos = np.array([0, 0])  # Initialize the position of the red dot
        self.running = True

    def check_collision(self, xy_traj):
        assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
        batch_size, num_steps, _ = xy_traj.shape
        xy_traj = xy_traj.reshape(-1, 2)
        xy_traj = np.clip(xy_traj, [0, 0], [self.maze.shape[0] - 1, self.maze.shape[1] - 1])
        maze_x = np.round(xy_traj[:, 0]).astype(int)
        maze_y = np.round(xy_traj[:, 1]).astype(int)
        collisions = self.maze[maze_x, maze_y]
        collisions = collisions.reshape(batch_size, num_steps)
        return np.any(collisions, axis=1)

    def blend_with_white(self, color, factor=0.5):
        white = np.array([255, 255, 255])
        blended_color = (1 - factor) * np.array(color) + factor * white
        return blended_color.astype(int)

    def report_collision_percentage(self, collisions):
        num_trajectories = collisions.shape[0]
        num_collisions = np.sum(collisions)
        collision_percentage = (num_collisions / num_trajectories) * 100
        print(f"{num_collisions}/{num_trajectories} trajectories are in collision ({collision_percentage:.2f}%).")
        return collision_percentage

    def xy2gui(self, xy):
        xy = xy + self.offset  # Adjust normalization as necessary
        x = xy[0] * self.gui_size[1] / (self.maze.shape[0])
        y = xy[1] * self.gui_size[0] / (self.maze.shape[1])
        return np.array([y, x], dtype=float)

    def gui2xy(self, gui):
        x = gui[1] / self.gui_size[1] * self.maze.shape[0] - self.offset
        y = gui[0] / self.gui_size[0] * self.maze.shape[1] - self.offset
        return np.array([x, y], dtype=float)

    def generate_time_color_map(self, num_steps):
        cmap = plt.get_cmap('rainbow')
        values = np.linspace(0, 1, num_steps)
        colors = cmap(values)
        return colors

    def draw_maze_background(self):
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(self.maze[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, self.gui_size)
        self.screen.blit(surface, (0, 0))

    def update_screen(self, xy_pred=None):
        self.draw_maze_background()
        if xy_pred is not None:
            time_colors = self.generate_time_color_map(xy_pred.shape[1])
            collisions = self.check_collision(xy_pred)
            self.report_collision_percentage(collisions)

            for idx, pred in enumerate(xy_pred):
                collision_detected = collisions[idx]
                whiteness_factor = 0.8 if collision_detected else 0.0
                circle_size = 10 if collision_detected else 5

                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    color = self.blend_with_white(color, whiteness_factor)

                    start_pos = self.xy2gui(pred[step_idx])
                    end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.mouse_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        pygame.display.flip()

class UnconditionalMaze(MazeEnv):
    def __init__(self, policy):
        super().__init__()
        self.mouse = None
        self.mouse_in_collision = False
        self.agent_history_xy = []
        self.policy_wrapped = policy

    def infer_target(self, timestamp):
        agent_hist_xy = self.agent_history_xy[-1]
        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "d -> b d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "d -> b d", b=self.batch_size
        )
        with torch.inference_mode(), torch.autocast(device_type="cuda"), seeded_context(0):
            actions = policy_wrapped.provide_observation_get_actions(obs_batch, timestamp, timestamp)
        actions = actions.cpu().numpy()
        return actions.transpose(1, 0, 2)

    def update_agent_gui_pos(self, history_len=1):
        self.mouse = np.array(pygame.mouse.get_pos())
        self.mouse_in_collision = self.check_collision(self.gui2xy(self.mouse).reshape(1, 1, 2))[0]
        if self.mouse_in_collision:
            self.mouse_color = self.blend_with_white(self.RED, 0.8)
        else:
            self.mouse_color = self.RED

        self.agent_gui_pos = self.mouse.copy()
        self.agent_history_xy.append(self.gui2xy(self.agent_gui_pos))
        self.agent_history_xy = self.agent_history_xy[-history_len:]

    def run(self):
        t = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            self.update_agent_gui_pos()
            self.policy_wrapped.reset()
            xy_pred = self.infer_target(t)
            self.update_screen(xy_pred)
            self.clock.tick(30)
            t += 1 / self.fps

        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', type=str, help="Policy name")
    args = parser.parse_args()

    # Load policy from the new codebase
    pretrained_policy_path = Path(os.path.join(args.checkpoint, "pretrained_model"))
    
    # Create and load the policy
    device = torch.device("cuda")

    if args.policy in ["diffusion", "dp"]:
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    elif args.policy in ["act"]:
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)

    policy.cuda()
    policy.eval()
    policy_wrapped = PolicyRolloutWrapper(policy, fps=10)  # fps and other params can be adjusted

    interactiveMaze = UnconditionalMaze(policy_wrapped)
    interactiveMaze.run()
