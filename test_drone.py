from stable_baselines3 import PPO
from airsim_gym import AirSimDroneEnv

env = AirSimDroneEnv()
model = PPO.load("ppo_airsim_tensorboard")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
