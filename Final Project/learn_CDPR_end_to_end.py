from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from gymnasium import utils

from CDPR_env_end_to_end import CableControlEnv
from callback import RenderCallback

env = CableControlEnv(render_mode="rgb_array")

# Check the environment
check_env(env)

# Define the model and training parameters
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ddpg_cable_control_tensorboard/")

# Train the model
model.learn(total_timesteps=400000)

# Save the trained model
model.save("DDPG_cable_control_end_to_endnew1")