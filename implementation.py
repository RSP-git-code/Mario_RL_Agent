#Import necessary libraries
import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# 1. Create Mario Environment
def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    return env

env = DummyVecEnv([make_env])


# 2. Callback for Saving Models

# class TrainAndLoggingCallback(BaseCallback):
#     def __init__(self, check_freq, save_path, verbose=1):
#         super().__init__(verbose)
#         self.check_freq = check_freq
#         self.save_path = save_path

#     def _init_callback(self):
#         os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self):
#         if self.n_calls % self.check_freq == 0:
#             model_path = os.path.join(
#                 self.save_path, f"ppo_mario_{self.n_calls}"
#             )
#             self.model.save(model_path)
#             print(f"Saved model at step {self.n_calls}")
#         return True

# checkpoint_dir = "./mario_models/"
# log_dir = "./mario_logs/"
# callback = TrainAndLoggingCallback(
#     check_freq=5000,
#     save_path=checkpoint_dir
# )

# --------------------------
# 3. PPO Model
# --------------------------
# model = PPO(
#     "CnnPolicy",
#     env,
#     verbose=1,
#     learning_rate=2.5e-4,
#     n_steps=512,
#     batch_size=64,
#     gamma=0.99,
#     gae_lambda=0.95,
#     ent_coef=0.01,
#     tensorboard_log=log_dir,
#     device="cpu"  # force CPU for stability
# )

# --------------------------
# 4. Train the Agent
# --------------------------
# model.learn(
#     total_timesteps=100000,
#     callback=callback
# )

# # --------------------------
# # 5. Save Final Model
# # --------------------------
# model.save("ppo_mario_final")

# print("Training finished successfully ")
model = PPO.load("ppo_mario_final", env=env)


# # Continue training
# model.learn(
#     total_timesteps=500000,
#     reset_num_timesteps=False
# )

# Save updated model
# model.save("ppo_mario_400_final")
# print("Continued training finished successfully ")

# model = PPO.load("ppo_mario_400_final", env=env)


# model.learn(total_timesteps=500000, reset_num_timesteps=False)

# 3. Save the new progress
# model.save("mario_models/ppo_mario_resumed_final")

# # 3. Test the Model

state = env.reset()

while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)

    # IMPORTANT for DummyVecEnv
    env.render()













