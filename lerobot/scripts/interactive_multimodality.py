import cv2
import einops
import gymnasium as gym
import numpy as np
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.utils.utils import seeded_context

SPACE_SIZE = 512

batch_size = 50
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

dp = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht_keypoints")
dp.config.noise_scheduler_type = "DDIM"
dp.config.num_inference_steps = 10
dp.config.n_action_steps = dp.config.horizon - dp.config.n_obs_steps + 1
dp.cuda()
dp.eval()
dp_wrapped = PolicyRolloutWrapper(dp, fps=fps)

act = ACTPolicy.from_pretrained("outputs/act_pusht_keypoints_4dec_novae")
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
policies_wrapped = [
    # dp_wrapped,
    # act_wrapped,
    vqbet_wrapped,
]
window_names = [
    # "Diffusion Policy",
    # "Action Chunking Transformer",
    "VQ-BeT",
]


def mouse_callback(event: int, x: int, y: int, flags: int = 0, *_):
    global action
    action = np.array([x / vis_size * SPACE_SIZE, y / vis_size * SPACE_SIZE])


img = env.render()
for window_name in window_names:
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.setMouseCallback(window_names[0], mouse_callback)

quit_ = False
t = 0

while not quit_:
    k = cv2.waitKey(1)

    if k == ord("q"):
        quit_ = True

    img = env.render()

    for window_name, policy_wrapped in zip(window_names, policies_wrapped, strict=True):
        img_ = img.copy()
        # Uncomment this one (and comment out the next one) if you want to just clear the action cache but
        # not the observation cache.
        policy_wrapped.invalidate_action_cache()
        # Uncomment this one (and comment out the last one) if you want to clear both the observation cache
        # and the action cache.
        # policy_wrapped.reset()
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
