from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from gymnasium import utils
from stable_baselines3.common.env_checker import check_env
from CDPR_env import CableControlEnv
from callback import RenderCallback

# Initialize your environment
env = CableControlEnv(render_mode="rgb_array")
check_env(env)

# render_callback = RenderCallback()

# Define and train the model
model_save_path = '/models/DDPG_cable_control_circle100_3.zip'
model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ddpg_cable_control_tensorboard/")
model.learn(total_timesteps=400000, log_interval=4) 
#             # callback=render_callback)
model.save(model_save_path)
# %tensorboard --logdir ./ppo_cable_control_tensorboard/ --port 6007