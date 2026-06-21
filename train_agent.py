from stable_baselines3 import PPO
from ur3_env import UR3ObstacleEnv

# 1. Initialize your custom ROS 2 / Gazebo environment
print("Connecting to Gazebo...")
env = UR3ObstacleEnv()

# 2. Build the PPO Neural Network (Using a Multi-Layer Perceptron policy)
# We also enable TensorBoard so you can track the learning curve live
print("Initializing PPO Agent...")
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ur3_tensorboard/"
)

# 3. Train the Agent
# 100,000 steps is a good quick test to ensure the math doesn't crash. 
# For your actual thesis, this will likely be 1,000,000 to 5,000,000 steps.
print("Starting Training Loop...")
model.learn(total_timesteps=100000)

# 4. Save the trained brain to your hard drive
model.save("ppo_ur3_obstacle_avoidance")
print("Training Finished and Model Saved!")