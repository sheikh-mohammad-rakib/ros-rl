import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading
from math import cos, sin, pi
import time

# 1. The ROS 2 Node
class UR3ROSNode(Node):
    def __init__(self):
        super().__init__('ur3_rl_bridge')
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.current_joint_angles = np.zeros(6)

    def joint_state_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joint_angles = np.array(msg.position[:6])

# 2. The RL Environment
class UR3ObstacleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = UR3ROSNode()
        self.executor_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
        self.executor_thread.start()

        # Action space normalized [-1, 1]. Highly recommended for PPO.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        
        # Max radians the arm is allowed to move per 0.1s step
        self.max_joint_step = 0.05 
        
        # Scenario coordinates matching the Gazebo spheres
        self.target_position = np.array([0.3, -0.2, 0.1]) 
        self.obstacle_center = np.array([0.2, 0.0, 0.2])
        self.obstacle_radius = 0.12 
        
        # Episode Constraints
        self.max_steps = 300
        self.current_step = 0

    def get_all_joint_positions(self, joint_angles):
        dh_params = [
            [0, 0.1519, pi/2],
            [-0.24365, 0, 0],
            [-0.21325, 0, 0],
            [0, 0.11235, pi/2],
            [0, 0.08535, -pi/2],
            [0, 0.0819, 0]
        ]
        T = np.eye(4)
        positions = [np.array([0, 0, 0])] 
        
        for i in range(6):
            theta = joint_angles[i]
            a, d, alpha = dh_params[i]
            A_i = np.array([
                [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                [0,           sin(alpha),             cos(alpha),            d],
                [0,           0,                      0,                     1]
            ])
            T = T @ A_i
            positions.append(np.array([T[0, 3], T[1, 3], T[2, 3]]))
        return positions

    def step(self, action):
        self.current_step += 1
        current_angles = self.ros_node.current_joint_angles
        
        # Calculate new target angles relative to current position
        target_angles = current_angles + (action * self.max_joint_step)
        target_angles = np.clip(target_angles, -np.pi, np.pi)

        # Send the action to Gazebo
        msg = JointTrajectory()
        msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = target_angles.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1s execution
        msg.points = [point]
        self.ros_node.publisher.publish(msg)
        
        # Wait EXACTLY the duration of the movement (0.1s)
        time.sleep(0.1) 
        obs = self.ros_node.current_joint_angles
        
        # Physics / Reward Math
        joint_positions = self.get_all_joint_positions(obs)
        end_effector_pos = joint_positions[-1]
        
        distance_to_goal = np.linalg.norm(end_effector_pos - self.target_position)
        
        collision = False
        min_dist_to_obstacle = float('inf')
        
        for pos in joint_positions:
            dist = np.linalg.norm(pos - self.obstacle_center)
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist
            if dist < self.obstacle_radius:
                collision = True
                break

        if collision:
            reward = -100.0  
            terminated = True
        elif distance_to_goal < 0.05:
            reward = 100.0   
            terminated = True
        else:
            safety_margin = max(0, min_dist_to_obstacle - self.obstacle_radius)
            reward = -distance_to_goal + (0.1 * safety_margin)
            terminated = False

        truncated = self.current_step >= self.max_steps

        return obs.astype(np.float32), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Send the robot back to a safe "Home" position to escape collision death loops
        home_angles = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
        
        msg = JointTrajectory()
        msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = home_angles.tolist()
        
        # FIXED: Give the robot 3 full seconds to slowly move home without triggering Gazebo errors
        point.time_from_start.sec = 3  
        point.time_from_start.nanosec = 0
        msg.points = [point]
        self.ros_node.publisher.publish(msg)
        
        # FIXED: Wait 3.5 seconds to ensure the physical arm has reached home before the episode begins
        time.sleep(3.5)
        
        return self.ros_node.current_joint_angles.astype(np.float32), {}