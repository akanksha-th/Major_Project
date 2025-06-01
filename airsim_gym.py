import airsim
import gym
from gym import spaces
import numpy as np
import cv2
import time
import base64

class AirSimDroneEnv(gym.Env):
    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Define action and observation space
        # Actions: [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Observations: resized camera image
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.goal_position = airsim.Vector3r(50, 0, -10)
        self.previous_position = None

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(1)
        
        self.goal_position = airsim.Vector3r(
            np.random.uniform(-50, 50), 
            np.random.uniform(-50, 50), 
            np.random.uniform(-20, -5)
        )

        # Reset previous position tracker
        self.previous_position = None
        
        return self._get_obs()

    def step(self, action):
        vx, vy, vz, yaw_rate = map(float, action)
        duration = 0.5

        self.client.moveByVelocityAsync(vx, vy, vz, duration, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)).join()

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()

        return obs, reward, done, {}

    def _get_obs(self):
        response = self.client.simGetImage("0", airsim.ImageType.Scene)
        if response is None:
            # Return a blank image if no data received
            return np.zeros((84, 84, 3), dtype=np.uint8)

        img_bytes = base64.b64decode(response)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)

        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img_cv2 is None:
            # Could not decode image, return blank image
            return np.zeros((84, 84, 3), dtype=np.uint8)

        img_resized = cv2.resize(img_cv2, (84, 84))
        return img_resized



    def _compute_reward(self):
        # Distance to goal
        current_pos = self.client.getMultirotorState().kinematics_estimated.position
        dist_to_goal = self._distance(current_pos, self.goal_position)

        # Delta movement (encourage smoothness)
        if self.previous_position is not None:
            move_delta = self._distance(current_pos, self.previous_position)
        else:
            move_delta = 0.0
        self.previous_position = current_pos

        # Collision check
        collision = self.client.simGetCollisionInfo().has_collided

        # Reward components
        reward = 0.0
        if collision:
            reward -= 20.0
        else:
            reward += 1.0  # Base reward for surviving

            # Encourage smooth, minimal movement
            reward -= move_delta * 2.0

            # Reward for approaching goal
            reward += max(0, 10.0 - dist_to_goal)

            # Bonus for reaching goal
            if dist_to_goal < 1.0:
                reward += 50.0

        return reward

    def _is_done(self):
        collision = self.client.simGetCollisionInfo().has_collided
        current_pos = self.client.getMultirotorState().kinematics_estimated.position
        goal_reached = self._distance(current_pos, self.goal_position) < 1.0
        out_of_bounds = (
            abs(current_pos.x_val) > 10000000 or
            abs(current_pos.y_val) > 10000000 or
            abs(current_pos.z_val) > 500000
            )
        return collision or goal_reached or out_of_bounds

    def _distance(self, pos1, pos2):
        return np.sqrt(
            (pos1.x_val - pos2.x_val) ** 2 +
            (pos1.y_val - pos2.y_val) ** 2 +
            (pos1.z_val - pos2.z_val) ** 2
        )