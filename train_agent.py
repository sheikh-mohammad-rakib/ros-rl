import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# CRITICAL FIX: Importing your exact class name
from ur3_env import UR3ObstacleEnv 

def main():
    # 1. Initialize the Environment
    print("Connecting to Gazebo UR3 Environment...")
    env = UR3ObstacleEnv()

    # 2. Build the PPO Agent (The "Brain")
    print("Initializing PPO Model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ur3_tensorboard/"
    )

    # 3. Setup the Auto-Save Checkpoint
    # This saves a backup of the brain every 10,000 steps so you never lose lab progress
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints/',
        name_prefix='ur3_ppo_checkpoint'
    )

    # 4. Start the Training Loop
    print("Starting Training (Target: 500,000 steps)...")
    model.learn(
        total_timesteps=500000, 
        callback=checkpoint_callback,
        reset_num_timesteps=False 
    )

    # 5. Final Save (If it completes all 500k steps)
    print("Training Complete! Saving final model...")
    model.save("ur3_ppo_final")

if __name__ == '__main__':
    main()