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

class MazeEnv:
    def __init__(self):
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
        self.GRAY = (128, 128, 128)
        self.agent_color = self.RED

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

    def update_screen(self, xy_pred=None, scores=None, keep_drawing=False):
        self.draw_maze_background()
        if xy_pred is not None:
            time_colors = self.generate_time_color_map(xy_pred.shape[1])
            collisions = self.check_collision(xy_pred)
            # self.report_collision_percentage(collisions)
            for idx, pred in enumerate(xy_pred):
                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    if scores is None: # whiteness indicates collision
                        whiteness_factor = 0.8 if collisions[idx] else 0.0
                        color = self.blend_with_white(color, whiteness_factor)
                        circle_size = 10 if collisions[idx] else 5
                    else: # whiteness indicates similarity score
                        color = color//3 + (color//3*2) * scores[idx] + 255//3*2 * (1-scores[idx])
                        circle_size = int(3 + 20 * scores[idx])
                    start_pos = self.xy2gui(pred[step_idx])
                    end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        if keep_drawing: # visualize the human drawing input
            # for point in self.draw_traj:
                # pygame.draw.circle(self.screen, self.GRAY, (int(point[0]), int(point[1])), 5)
            # draw lines
            for i in range(len(self.draw_traj) - 1):
                pygame.draw.line(self.screen, self.GRAY, self.draw_traj[i], self.draw_traj[i + 1], 10)

  
        pygame.display.flip()

    def similarity_score(self, samples, guide=None):
        # samples: (B, pred_horizon, action_dim)
        # guide: (guide_horizon, action_dim)
        if guide is None:
            return samples, None
        assert samples.shape[2] == 2 and guide.shape[1] == 2
        indices = np.linspace(0, guide.shape[0]-1, samples.shape[1], dtype=int)
        guide = np.expand_dims(guide[indices], axis=0) # (1, pred_horizon, action_dim)
        guide = np.tile(guide, (samples.shape[0], 1, 1)) # (B, pred_horizon, action_dim)
        scores = np.linalg.norm(samples[:, 1:] - guide[:, 1:], axis=2).mean(axis=1) # (B,)
        scores = 1 - scores / (scores.max() + 1e-6) # normalize
        temperature = 20
        scores = softmax(scores*temperature)
        # print('scores:', [f'{score:.3f}' for score in scores])
        # normalize the score to be between 0 and 1
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        # sort the predictions based on scores, from smallest to largest, so that larger scores will be drawn on top
        sort_idx = np.argsort(scores)
        samples = samples[sort_idx]
        scores = scores[sort_idx]  
        return samples, scores

class UnconditionalMaze(MazeEnv):
    def __init__(self, policy):
        super().__init__()
        self.mouse_pos = None
        self.agent_in_collision = False
        self.agent_history_xy = []
        self.policy_wrapped = policy

    def infer_target(self, timestamp, guide=None, visualizer=None):
        self.policy_wrapped.reset()
        agent_hist_xy = self.agent_history_xy[-1] # rely on policy wrapper to fill the history
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2).repeat(2, axis=0)

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
        )
        
        if guide is not None:
            guide = torch.from_numpy(guide).float().cuda()

        with torch.autocast(device_type="cuda"), seeded_context(0):
            # actions = self.policy_wrapped.provide_observation_get_actions(obs_batch, timestamp, timestamp, guide=guide, visualizer=None).transpose(1, 0, 2).cpu().numpy()
            actions = policy_wrapped.policy.run_inference(obs_batch, guide=guide, visualizer=visualizer).cpu().numpy() # directly call the policy in order to visualize the intermediate steps
        return actions

    def update_mouse_pos(self):
        self.mouse_pos = np.array(pygame.mouse.get_pos())
        self.agent_in_collision = self.check_collision(self.gui2xy(self.agent_gui_pos).reshape(1, 1, 2))[0]
        if self.agent_in_collision:
            self.agent_color = self.blend_with_white(self.RED, 0.8)
        else:
            self.agent_color = self.RED

    def update_agent_pos(self, history_len=1):
        self.agent_gui_pos = self.mouse_pos.copy()
        self.agent_history_xy.append(self.gui2xy(self.agent_gui_pos))
        self.agent_history_xy = self.agent_history_xy[-history_len:]

    def run(self):
        t = 0
        while self.running:
            self.update_mouse_pos()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            self.update_agent_pos()
            xy_pred = self.infer_target(t)
            self.update_screen(xy_pred)
            self.clock.tick(30)
            t += 1 / self.fps

        pygame.quit()


class ConditionalMaze(UnconditionalMaze):
    def __init__(self, policy):
        super().__init__(policy)
        self.drawing = False
        self.keep_drawing = False
        self.draw_traj = []

    def run(self):
        t = 0
        while self.running:
            self.update_mouse_pos()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                    if not self.drawing:
                        self.drawing = True
                        self.draw_traj = []
                    self.draw_traj.append(self.mouse_pos)
                else: # mouse released
                    if self.drawing: 
                        self.drawing = False # finish drawing action
                        self.keep_drawing = True # keep visualizing the drawing

            if self.keep_drawing: # visualize the human drawing input
                # Check if mouse returns to the agent's location
                if np.linalg.norm(self.mouse_pos - self.agent_gui_pos) < 20:  # Threshold distance to reactivate the agent
                    self.keep_drawing = False # delete the drawing
                    self.draw_traj = []

            if not self.drawing: # inference mode
                if not self.keep_drawing:
                    self.update_agent_pos()
                if len(self.draw_traj) > 0:
                    guide = np.array([self.gui2xy(point) for point in self.draw_traj])
                else:
                    guide = None
                xy_pred = self.infer_target(t, guide, visualizer=self)
                scores = None
                # xy_pred, scores = self.similarity_score(xy_pred, guide)
            
            self.update_screen(xy_pred, scores, (self.keep_drawing or self.drawing))
            self.clock.tick(30)
            t += 1 / self.fps

        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', type=str, help="Policy name")
    parser.add_argument('-u', '--unconditional', action='store_true', help="Unconditional Maze")
    parser.add_argument('-ph', '--post-hoc', action='store_true', help="Post-hoc alignment")
    parser.add_argument('-bi', '--biased-initialization', action='store_true', help="Biased initialization")
    parser.add_argument('-gd', '--guided-diffusion', action='store_true', help="Guided diffusion")
    parser.add_argument('-rd', '--recurrent-diffusion', action='store_true', help="Recurrent diffusion")
    args = parser.parse_args()

    # Load policy from the new codebase
    pretrained_policy_path = Path(os.path.join(args.checkpoint, "pretrained_model"))
    
    # Create and load the policy
    device = torch.device("cuda")

    alignment_strategy = None
    if args.post_hoc:
        alignment_strategy = 'post-hoc'
    elif args.biased_initialization:
        alignment_strategy = 'biased-initialization'
    elif args.guided_diffusion:
        alignment_strategy = 'guided-diffusion'
    elif args.recurrent_diffusion:
        alignment_strategy = 'recurrent-diffusion'

    if args.policy in ["diffusion", "dp"]:
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, alignment_strategy=alignment_strategy)
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    elif args.policy in ["act"]:
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)

    policy.cuda()
    policy.eval()
    policy_wrapped = PolicyRolloutWrapper(policy, fps=10)  # fps and other params can be adjusted

    if args.unconditional:
        interactiveMaze = UnconditionalMaze(policy_wrapped)
    else:
        interactiveMaze = ConditionalMaze(policy_wrapped)
    interactiveMaze.run()
