"""
Run from the root directory of the project with `python lerobot/scripts/interactive_multimodality.py`.

This script:
    - Sets up a PushT environment (custom branch: https://github.com/alexander-soare/gym-pusht/tree/add_all_mode)
    - Loads up 3 policies from pretrained models on the hub: Diffusion Policy, ACT, and VQ-BeT.
        - You can switch between ACT with and without VAE (see comments below).
    - Runs the environment in a loop showing visualizations for each of the policies. You can mouse over the
      first window to control the robot.
        - You can comment in/out policies to show.
        - You can noise the observations prior to input to the policies.

NOTE about setup. You can follow the setup instructions in the main LeRobot README.
"""

import cv2
import einops
import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.utils.utils import seeded_context

SPACE_SIZE = 512

batch_size = 50  # visualize this many trajectories per inference
vis_size = 512

env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_environment_state_agent_pos",
    visualization_width=vis_size,
    visualization_height=vis_size,
)
fps = env.unwrapped.metadata["render_fps"]
obs, _ = env.reset()
action = obs["agent_pos"]

dp_img = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
dp_img.config.noise_scheduler_type = "DDIM"
dp_img.diffusion.num_inference_steps = 10
dp_img.config.n_action_steps = dp_img.config.horizon - dp_img.config.n_obs_steps + 1
dp_img.cuda()
dp_img.eval()
dp_img_wrapped = PolicyRolloutWrapper(dp_img, fps=fps)

dp_kp = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht_keypoints")
dp_kp.config.noise_scheduler_type = "DDIM"
dp_kp.diffusion.num_inference_steps = 10
dp_kp.config.n_action_steps = dp_kp.config.horizon - dp_kp.config.n_obs_steps + 1
dp_kp.cuda()
dp_kp.eval()
dp_kp_wrapped = PolicyRolloutWrapper(dp_kp, fps=fps)


act = ACTPolicy.from_pretrained("alexandersoare/act_pusht_keypoints")
# act = ACTPolicy.from_pretrained("alexandersoare/act_nvae_pusht_keypoints")  # no VAE version
act.cuda()
act.eval()
act_wrapped = PolicyRolloutWrapper(act, fps=fps)

vqbet = VQBeTPolicy.from_pretrained("lerobot/vqbet_pusht")
vqbet.config.bet_softmax_temperature = 1.0
vqbet.cuda()
vqbet.eval()
vqbet_wrapped = PolicyRolloutWrapper(vqbet, fps=fps)


def run_inference(policy_wrapped, obs, timestamp, noise_std=0):
    obs_batch = {
        "observation.state": einops.repeat(
            torch.from_numpy(obs["agent_pos"]).float().cuda(), "d -> b d", b=batch_size
        )
    }
    with seeded_context(0):
        obs_batch["observation.state"] = (
            obs_batch["observation.state"] + torch.randn_like(obs_batch["observation.state"]) * noise_std
        )
    if "pixels" in obs:
        obs_batch["observation.image"] = einops.repeat(
            torch.from_numpy(obs["pixels"]).float().cuda().permute(2, 0, 1), "... -> b ...", b=batch_size
        )
    if "environment_state" in obs:
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(obs["environment_state"]).float().cuda(), "d -> b d", b=batch_size
        )
        with seeded_context(0):
            obs_batch["observation.environment_state"] = (
                obs_batch["observation.environment_state"]
                + torch.randn_like(obs_batch["observation.environment_state"]) * noise_std
            )
    with torch.inference_mode(), torch.autocast(device_type="cuda"), seeded_context(0):
        actions = policy_wrapped.provide_observation_get_actions(obs_batch, timestamp, timestamp)
    return actions.cpu().numpy()  # (S, B, 2)


# Uncomment/comment pairs of policies and window names.
ls_window_names_and_policies = [
    # ("Diffusion Policy (image)", dp_img_wrapped),
    ("Diffusion Policy (keypoints)", dp_kp_wrapped),
    ("Action Chunking Transformer", act_wrapped),
    # ("VQ-BeT", vqbet_wrapped),
]


def mouse_callback(event: int, x: int, y: int, flags: int = 0, *_):
    global action
    action = np.array([x / vis_size * SPACE_SIZE, y / vis_size * SPACE_SIZE])


img = env.render()
for window_name, _ in ls_window_names_and_policies:
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.setMouseCallback(ls_window_names_and_policies[0][0], mouse_callback)

quit_ = False
t = 0

while not quit_:
    k = cv2.waitKey(1)

    if k == ord("q"):
        quit_ = True

    img = env.render()

    for window_name, policy_wrapped in ls_window_names_and_policies:
        img_ = img.copy()
        # Uncomment this one (and comment out the next one) if you want to just clear the action cache but
        # not the observation cache.
        # policy_wrapped.invalidate_action_cache()
        # Uncomment this one (and comment out the last one) if you want to clear both the observation cache
        # and the action cache.
        policy_wrapped.reset()
        # Set noise_std to a non-zero value to noise the observations prior to input to the policies. 4 is
        # a good value.
        policy_batch_actions = run_inference(policy_wrapped, obs, t, noise_std=0)

        obs, *_ = env.step(action)

        for b in range(policy_batch_actions.shape[1]):
            policy_actions = policy_batch_actions[:, b] / 512 * img_.shape[:2]  # (horizon, 2)
            policy_actions = np.round(policy_actions).astype(int)
            for k, policy_action in enumerate(policy_actions[::-1]):
                cv2.circle(
                    img_,
                    tuple(policy_action),
                    radius=2,
                    color=(
                        int(255 * k / len(policy_actions)),
                        0,
                        int(255 * (len(policy_actions) - k) / len(policy_actions)),
                    ),
                    thickness=1,
                )
        cv2.imshow(window_name, cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))

    t += 1 / fps
