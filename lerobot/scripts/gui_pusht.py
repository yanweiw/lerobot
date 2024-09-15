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
import time
import json
import gym_pusht
import gymnasium as gym
import cv2

class PushTEnv:
    def __init__(self, policy_tag=None):
        # GUI x coord 0 -> gui_size[0] #512
        # GUI y coord 0 
        #         |
        #         v
        #       gui_size[1] #512
        # xy is the same

        self.seed = 0
        self.gui_size = (512, 512)
        self.SPACE_SIZE = 512
        self.fps = 10
        self.batch_size = 50        
        self.offset = 0.5  # Offset to put object in the center of the cell

        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.agent_color = self.RED

        # Initialize environment
        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_environment_state_agent_pos",
            visualization_width=self.gui_size[0],
            visualization_height=self.gui_size[1],
        )       
        self.obs, _ = self.env.reset()
        self.action = self.obs["agent_pos"]

        # Initialize policies
        self.dp_img = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
        self.dp_img.config.noise_scheduler_type = "DDIM"
        self.dp_img.diffusion.num_inference_steps = 10
        self.dp_img.config.n_action_steps = self.dp_img.config.horizon - self.dp_img.config.n_obs_steps + 1
        self.dp_img.cuda()
        self.dp_img.eval()
        self.act = ACTPolicy.from_pretrained("alexandersoare/act_pusht_keypoints")
        self.act.cuda()
        self.act.eval()

        # Set policies with window names
        if policy_tag == 'both':
            self.ls_window_names_and_policies = [
                ("Diffusion Policy (image)", self.dp_img),
                ("Action Chunking Transformer", self.act),
            ]
        elif policy_tag == 'dp':
            self.ls_window_names_and_policies = [
                ("Diffusion Policy (image)", self.dp_img),
            ]
        elif policy_tag == 'act':
            self.ls_window_names_and_policies = [
                ("Action Chunking Transformer", self.act),
            ]
        else:
            raise ValueError(f"Invalid policy tag: {policy_tag}")

        # Initialize variables
        self.mouse_pos = None
        # self.agent_gui_pos = None
        # self.agent_history_xy = []
        self.running = True

        # Set up OpenCV windows
        for window_name, _ in self.ls_window_names_and_policies:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, self.mouse_callback)

    def mouse_callback(self, event: int, x: int, y: int, flags: int = 0, *_):
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = np.array([x, y])

    # def update_mouse_pos(self):
        # In OpenCV, the mouse position is updated via the mouse_callback
        # pass  # The mouse position is already updated in the callback

    def update_agent_pos(self, history_len=1):
        # self.agent_gui_pos = np.array(new_agent_pos)
        # self.action = self.agent_gui_pos / self.gui_size[0] * self.SPACE_SIZE
        # self.agent_history_xy.append(self.action)
        # self.agent_history_xy = self.agent_history_xy[-history_len:]

        # self.obs, *_ = self.env.step(self.action)
        # from IPython import embed; embed()
        self.action = self.mouse_pos / self.gui_size[0] * self.SPACE_SIZE

    def generate_time_color_map(self, num_steps):
        cmap = plt.get_cmap('rainbow')
        values = np.linspace(0, 1, num_steps)
        colors = cmap(values)
        # Reverse the color ordering so that later steps are redder
        colors = colors[::-1]
        return colors

    def similarity_score(self, samples, guide=None):
        # samples: (B, pred_horizon, action_dim)
        # guide: (guide_horizon, action_dim)
        if guide is None:
            return samples, None
        assert samples.shape[2] == 2 and guide.shape[1] == 2
        indices = np.linspace(0, guide.shape[0]-1, samples.shape[1], dtype=int)
        guide = np.expand_dims(guide[indices], axis=0)  # (1, pred_horizon, action_dim)
        guide = np.tile(guide, (samples.shape[0], 1, 1))  # (B, pred_horizon, action_dim)
        scores = np.linalg.norm(samples - guide, axis=2, ord=2).mean(axis=1)  # (B,)
        scores = 1 - scores / (scores.max() + 1e-6)  # normalize
        temperature = 20
        scores = softmax(scores * temperature)
        # Normalize the score to be between 0 and 1
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        # Sort the predictions based on scores
        sort_idx = np.argsort(scores)
        samples = samples[sort_idx]
        scores = scores[sort_idx]  
        return samples, scores

    def update_screen(self, img, window_name, xy_pred=None, collisions=None, scores=None, keep_drawing=False, traj_in_gui_space=False):
        img_ = img.copy()
        if xy_pred is not None:
            time_colors = self.generate_time_color_map(xy_pred.shape[1])
            for idx, pred in enumerate(xy_pred):
                for step_idx in range(len(pred) - 1):
                    color_rgba = time_colors[step_idx]
                    # Convert RGBA to RGB and scale to [0, 255]
                    color_rgb = (color_rgba[:3] * 255).astype(int)
                    # Convert RGB to BGR for OpenCV
                    color_bgr = color_rgb[::-1]
                    # Ensure color values are standard Python integers
                    color = tuple(map(int, color_bgr))
                    if scores is None:
                        circle_size = 5  # Increased base size from 2 to 5
                    else:
                        circle_size = int(5 + 30 * scores[idx])  # Adjusted scaling factor for larger circles

                    if traj_in_gui_space:
                        start_pos = pred[step_idx]
                        end_pos = pred[step_idx + 1]
                    else:
                        start_pos = pred[step_idx] / self.SPACE_SIZE * self.gui_size[0]
                        end_pos = pred[step_idx + 1] / self.SPACE_SIZE * self.gui_size[0]
                    start_pos = tuple(np.round(start_pos).astype(int))
                    end_pos = tuple(np.round(end_pos).astype(int))
                    cv2.circle(
                        img_,
                        start_pos,
                        radius=circle_size,
                        color=color,
                        thickness=-1,
                    )
        # Draw the agent
        if self.action is not None:
            agent_pos = tuple(np.round(self.action).astype(int))
            # Ensure agent color is in BGR format and standard integers
            agent_color_bgr = self.agent_color
            agent_color = tuple(map(int, agent_color_bgr))
            cv2.circle(
                img_,
                agent_pos,
                radius=23,
                color=agent_color,
                thickness=-1,
            )
        cv2.imshow(window_name, cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
        return img_

class UnconditionalEnv(PushTEnv):
    def __init__(self, policy_tag=None):
        super().__init__(policy_tag=policy_tag)
        self.clock = pygame.time.Clock()  # For consistent fps

    def infer_target(self, policy, policy_tag, guide=None, visualizer=None):
        # Use policy and policy_tag passed as parameters
        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(self.action).float().cuda(), "d -> b d", b=self.batch_size
            )
        }
        if policy_tag == 'dp': 
            # Create a temporal dimension after batch and before the state dimension
            obs_batch["observation.state"] = einops.repeat(obs_batch["observation.state"], "b ... -> b t ...", t=2)
        if "pixels" in self.obs:
            obs_batch["observation.image"] = einops.repeat(
                torch.from_numpy(self.obs["pixels"]).float().cuda().permute(2, 0, 1), "... -> b ...", b=self.batch_size
            )
            if policy_tag == 'dp': 
                # Create a temporal dimension after batch and before the state dimension
                obs_batch["observation.image"] = einops.repeat(obs_batch["observation.image"], "b ... -> b t ...", t=2)
        if "environment_state" in self.obs:
            obs_batch["observation.environment_state"] = einops.repeat(
                torch.from_numpy(self.obs["environment_state"]).float().cuda(), "d -> b d", b=self.batch_size
            )
            if policy_tag == 'dp':
                # Create a temporal dimension after batch and before the state dimension
                obs_batch["observation.environment_state"] = einops.repeat(
                    obs_batch["observation.environment_state"], "b ... -> b t ...", t=2
                )
    
        if guide is not None:
            guide = torch.from_numpy(guide).float().cuda()

        with torch.autocast(device_type="cuda"), seeded_context(0):
            if policy_tag == 'act':
                actions = policy.run_inference(obs_batch).cpu().numpy()
            elif policy_tag == 'dp':
                actions = policy.run_inference(obs_batch, guide=guide, visualizer=visualizer).cpu().numpy()
            else:
                raise ValueError(f"Invalid policy tag: {policy_tag}")
        return actions

    def run(self, seed=0):
        self.seed = seed
        self.obs, _ = self.env.reset(seed=self.seed)
        np.random.seed(self.seed)
        while self.running:
            k = cv2.waitKey(1)
            if k == ord("q"):
                self.running = False
                break
            # self.update_mouse_pos()
            if self.mouse_pos is not None:
                self.update_agent_pos()
            img = self.env.render()
            # For each policy, infer and plot predictions
            for window_name, policy in self.ls_window_names_and_policies:
                # Determine the policy tag based on the window name
                policy_tag = 'dp' if 'Diffusion' in window_name else 'act'
                # Create a copy of the image for each policy
                xy_pred = self.infer_target(policy, policy_tag)
                self.obs, *_ = self.env.step(self.action)
                self.update_screen(img, window_name, xy_pred)
            self.clock.tick(self.fps)



class ConditionalEnv(UnconditionalEnv):
    # For interactive guidance dataset collection
    def __init__(self, vis_dp_dynamics=False, savepath=None, alignment_strategy=None, policy_tag=None):
        super().__init__(policy_tag=policy_tag)
        self.drawing = False
        self.keep_drawing = False
        self.vis_dp_dynamics = vis_dp_dynamics
        self.savefile = None
        self.savepath = savepath
        self.draw_traj = []  # GUI coordinates
        self.xy_pred = {}  # numpy array
        self.scores = {}  # numpy array
        self.alignment_strategy = alignment_strategy
        if self.dp_img is not None:
            self.dp_img.diffusion.alignment_strategy = alignment_strategy
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
            self.trial_idx = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print('left button down')
            if not self.drawing:
                self.drawing = True
                self.draw_traj = []
            self.draw_traj.append(np.array([x, y]))
        elif event == cv2.EVENT_MOUSEMOVE:
            # print('mouse move')
            if self.drawing:
                self.draw_traj.append(np.array([x, y]))
            self.mouse_pos = np.array([x, y])
        elif event == cv2.EVENT_LBUTTONUP:
            # print('left button up')
            self.drawing = False
            self.keep_drawing = True
        else:
            # print('mouse move')
            self.mouse_pos = np.array([x, y])

    def run(self, seed=0):
        self.seed = seed
        self.obs, _ = self.env.reset(seed=self.seed)
        np.random.seed(self.seed)
        img = self.env.render()  # Initialize img before the loop
        while self.running:
            # print('self.drawing', self.drawing, 'self.keep_drawing', self.keep_drawing, 'self.mouse_pos', self.mouse_pos, 'self.action', self.action)
            k = cv2.waitKey(1)
            if k == ord('q'):
                self.running = False
                break
            elif k == ord('s') and self.savefile is not None:
                self.save_trials()
                self.obs, _ = self.env.reset(seed=self.seed)
                np.random.seed(self.seed)
                self.draw_traj = []
                self.keep_drawing = False
                self.drawing = False

            # Check if mouse returns to the agent's location to reset drawing
            if self.keep_drawing:
                if np.linalg.norm(self.mouse_pos - self.action) < 10:
                    self.keep_drawing = False
                    self.draw_traj = []         

            if not self.drawing: # inference mode
                if not self.keep_drawing and self.mouse_pos is not None:
                    self.update_agent_pos()
                if len(self.draw_traj) > 0:
                    guide = np.array(self.draw_traj) / self.gui_size[0] * self.SPACE_SIZE
                else:
                    guide = None

                for window_name, policy in self.ls_window_names_and_policies:
                    # Determine the policy tag based on the window name
                    policy_tag = 'dp' if 'Diffusion' in window_name else 'act'
                    # Always perform inference, guide can be None
                    xy_pred = self.infer_target(policy, policy_tag, guide=guide)
                    # Optionally, apply alignment strategy
                    scores = None
                    if self.alignment_strategy == 'post-hoc' and guide is not None:
                        xy_pred, scores = self.similarity_score(xy_pred, guide)
                    # Store predictions
                    self.xy_pred[policy_tag] = xy_pred
                    self.scores[policy_tag] = scores
                    # 
                    img = self.env.render()  # Update img only when not drawing
                    # Update the environment and display
                    self.obs, *_ = self.env.step(self.action)
                    self.update_screen(img, window_name, xy_pred, scores=scores, keep_drawing=(self.keep_drawing or self.drawing))
            else:
                # If drawing, just display the drawing without inference
                for window_name, _ in self.ls_window_names_and_policies:
                    img_ = img.copy()  # Use the last rendered image
                    policy_tag = 'dp' if 'Diffusion' in window_name else 'act'
                    img_ = self.update_screen(img_, window_name, self.xy_pred[policy_tag], scores=self.scores[policy_tag], keep_drawing=(self.keep_drawing or self.drawing))
                    # Draw the agent
                    if self.action is not None:
                        agent_pos = tuple(np.round(self.action).astype(int))
                        agent_color_bgr = self.agent_color[::-1]
                        agent_color = tuple(map(int, agent_color_bgr))
                        cv2.circle(
                            img_,
                            agent_pos,
                            radius=10,
                            color=agent_color,
                            thickness=-1,
                        )
                    # Draw the drawing trajectory
                    if len(self.draw_traj) > 1:
                        pts = np.array(self.draw_traj, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_, [pts], isClosed=False, color=(128, 128, 128), thickness=5)
                    cv2.imshow(window_name, cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))



            self.clock.tick(self.fps)

    def save_trials(self):
        if self.xy_pred: # empty dict evaluates to False
            policy_tag = list(self.xy_pred.keys())
            assert len(policy_tag) == 1
            policy_tag = policy_tag[0]
            assert policy_tag in ['dp', 'act']
            xy_pred = self.xy_pred[policy_tag]
            b, t, _ = xy_pred.shape
            xy_pred = xy_pred.reshape(b * t, 2)
            pred_gui_traj = xy_pred / self.SPACE_SIZE * self.gui_size[0]
            pred_gui_traj = pred_gui_traj.reshape(b, t, 2)
            entry = {
                "seed": self.seed, 
                "trial_idx": self.trial_idx,
                "agent_pos": self.action.tolist(),
                "guide": np.array(self.draw_traj).tolist(),
                "pred_traj": pred_gui_traj.astype(int).tolist(),
            }
            self.savefile.write(json.dumps(entry) + "\n")
            print(f"Trial {self.trial_idx} saved to {self.savepath}.")
            self.trial_idx += 1
            self.seed += 1


# class MazeExp(ConditionalMaze):
#     # for replaying the trials and benchmarking the alignment strategies
#     def __init__(self, policy, vis_dp_dynamics=False, savepath=None, alignment_strategy=None, policy_tag=None, loadpath=None):
#         super().__init__(policy, vis_dp_dynamics, savepath, policy_tag=policy_tag)
#         # Load saved trails
#         assert loadpath is not None
#         with open(args.loadpath, "r", buffering=1) as file:
#             file.seek(0)
#             trials = [json.loads(line) for line in file]
#             # set random seed and shuffle the trials
#             np.random.seed(0)
#             np.random.shuffle(trials)

#         self.trials = trials
#         self.trial_idx = 0
#         # if savepath is not None:
#         #     # append loadpath to the savepath as prefix
#         #     self.savepath = loadpath[:-5] + '_' + policy_tag + '_' + savepath
#         #     self.savefile = open(self.savepath, "a+", buffering=1)
#         #     self.trial_idx = 0
#         self.alignment_strategy = alignment_strategy
#         print(f"Alignment strategy: {alignment_strategy}")

#     def run(self):
#         if self.savepath is not None:
#             self.savefile = open(savepath, "w+", buffering=1)
#             self.trial_idx = 0

#         while self.trial_idx < len(self.trials):
#             # Load the trial
#             self.draw_traj = self.trials[self.trial_idx]["guide"]
            
#             # skip empty trials
#             if len(self.draw_traj) == 0: 
#                 print(f"Skipping trial {self.trial_idx} which has no guide.")
#                 self.trial_idx += 1
#                 continue
            
#             # skip trials with all collisions
#             first_collision_idx = self.find_first_collision_from_GUI(np.array(self.draw_traj))
#             if first_collision_idx <= 0: # no collision or all collisions
#                 if np.array(self.trials[self.trial_idx]["collisions"]).all():
#                     print(f"Skipping trial {self.trial_idx} which has all collisions.")
#                     self.trial_idx += 1
#                     continue

#             # initialize the agent position
#             if self.alignment_strategy == 'output-perturb':
#                 # find the location before the first collision to initialize the agent
#                 if first_collision_idx <= 0: # no collision or all collisions
#                     perturbed_pos = self.draw_traj[20]
#                 else:
#                     first_collision_idx = min(first_collision_idx, 20)
#                     perturbed_pos = self.draw_traj[first_collision_idx - 1]
#                 self.update_agent_pos(perturbed_pos)
#             else:
#                 self.update_agent_pos(self.trials[self.trial_idx]["agent_pos"])

#             # infer the target based on the guide
#             if self.policy is not None:
#                 guide = np.array([self.gui2xy(point) for point in self.draw_traj])
#                 self.xy_pred = self.infer_target(guide, visualizer=(self if self.vis_dp_dynamics else None))
#                 if self.alignment_strategy in ['output-perturb', 'post-hoc']:
#                     self.xy_pred, scores = self.similarity_score(self.xy_pred, guide)
#                 else:
#                     scores = None
#                 self.collisions = self.check_collision(self.xy_pred)
#                 self.update_screen(self.xy_pred, self.collisions, scores=scores, keep_drawing=True, traj_in_gui_space=False)
#                 if self.vis_dp_dynamics:
#                     time.sleep(1)
                    
#                 # save the experiment trial
#                 if self.savepath is not None:
#                     self.save_trials()

#             # just replay the trials without inference    
#             else:
#                 collisions = self.trials[self.trial_idx]["collisions"]
#                 pred_traj = np.array(self.trials[self.trial_idx]["pred_traj"])
#                 if self.alignment_strategy in ['output-perturb', 'post-hoc']:
#                     _, scores = self.similarity_score(pred_traj, np.array(self.trials[self.trial_idx]["guide"])) # this is a hack as both pred_traj and guide are in gui space, don't use this score for absolute statistics calculation
#                 else:
#                     scores = None
#                 self.update_screen(pred_traj, collisions, scores=scores, keep_drawing=True, traj_in_gui_space=True)

#             # Handle events
#             for event in pygame.event.get():
#                 if event.type == pygame.KEYDOWN:
#                     assert self.savefile is None
#                     if event.key == pygame.K_n and self.savefile is None: # visualization mode rather than saving mode
#                         print("manual skip to the next trial")
#                         self.trial_idx += 1

#             self.clock.tick(10)

#         pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', default=None, type=str, help="Policy name")
    parser.add_argument('-u', '--unconditional', action='store_true', help="Unconditional Maze")
    parser.add_argument('-op', '--output-perturb', action='store_true', help="Output perturbation")
    parser.add_argument('-ph', '--post-hoc', action='store_true', help="Post-hoc alignment")
    parser.add_argument('-bi', '--biased-initialization', action='store_true', help="Biased initialization")
    parser.add_argument('-gd', '--guided-diffusion', action='store_true', help="Guided diffusion")
    parser.add_argument('-rd', '--recurrent-diffusion', action='store_true', help="Recurrent diffusion")
    parser.add_argument('-v', '--vis_dp_dynamics', action='store_true', help="Visualize dynamics in DP")
    parser.add_argument('-s', '--savepath', type=str, default=None, help="Filename to save the drawing")
    parser.add_argument('-l', '--loadpath', type=str, default=None, help="Filename to load the drawing")

    args = parser.parse_args()

    # Create and load the policy
    device = torch.device("cuda")

    alignment_strategy = 'post-hoc'
    if args.post_hoc:
        alignment_strategy = 'post-hoc'
    elif args.output_perturb:
        alignment_strategy = 'output-perturb'
    elif args.biased_initialization:
        alignment_strategy = 'biased-initialization'
    elif args.guided_diffusion:
        alignment_strategy = 'guided-diffusion'
    elif args.recurrent_diffusion:
        alignment_strategy = 'recurrent-diffusion'

    if args.policy in ["diffusion", "dp"]:
        policy_tag = 'dp'
    elif args.policy in ["act"]:
        policy_tag = 'act'
    else:
        policy_tag = 'both'
        
    if args.unconditional:
        interactiveEnv = UnconditionalEnv(policy_tag=policy_tag)
    # elif args.loadpath is not None:
    #     if args.savepath is None:
    #         savepath = None
    #     else:
    #         alignment_tag = 'ph'
    #         if alignment_strategy == 'output-perturb':
    #             alignment_tag = 'op'
    #         elif alignment_strategy == 'biased-initialization':
    #             alignment_tag = 'bi'
    #         elif alignment_strategy == 'guided-diffusion':
    #             alignment_tag = 'gd'
    #         elif alignment_strategy == 'recurrent-diffusion':
    #             alignment_tag = 'rd'
    #         savepath = f"{args.loadpath[:-5]}_{policy_tag}_{alignment_tag}{args.savepath}"
    #     interactiveMaze = MazeExp(policy, args.vis_dp_dynamics, savepath, alignment_strategy, policy_tag=policy_tag, loadpath=args.loadpath)
    else:
        interactiveEnv = ConditionalEnv(args.vis_dp_dynamics, args.savepath, alignment_strategy, policy_tag=policy_tag)
    interactiveEnv.run(seed=2)
