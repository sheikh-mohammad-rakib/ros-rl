import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from ur3_env import UR3ObstacleEnv 

def main():
    # 1. Initialize the Environment
    print("Connecting to Gazebo UR3 Environment...")
    env = UR3ObstacleEnv()

    # 2. LOAD THE VETERAN BRAIN (190k Steps)
    print("Waking up the saved PPO Model from 190,000 steps...")
    checkpoint_path = "./models/checkpoints/ur3_ppo_checkpoint_190000_steps" 
    
    model = PPO.load(
        checkpoint_path, 
        env=env,
        tensorboard_log="./ur3_tensorboard/"
    )

    # 3. Setup the Auto-Save Checkpoint (Keep this active!)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints/',
        name_prefix='ur3_ppo_checkpoint'
    )

    # 4. Resume the Training Loop (310k remaining to reach 500k total)
    print("Resuming Training for the final 310,000 steps...")
    model.learn(
        total_timesteps=310000, 
        callback=checkpoint_callback,
        reset_num_timesteps=False # CRITICAL: This connects your new data to the old 190k graph!
    )

    # 5. Final Save
    print("Training Complete! Saving final 500k model...")
    model.save("ur3_ppo_final_500k")

if __name__ == '__main__':
    main()