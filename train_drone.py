from stable_baselines3 import PPO
from airsim_gym import AirSimDroneEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap the environment
env = DummyVecEnv([lambda: AirSimDroneEnv()])

# Instantiate the model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_airsim_tensorboard/")

# Train the model
try:
    model.learn(total_timesteps=100_000)
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save("ppo_airsim_drone")
