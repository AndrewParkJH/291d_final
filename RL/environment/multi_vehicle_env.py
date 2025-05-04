from simulator.sav_simulator_rl import ShuttleSim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pandas as pd
import simpy

class MultiVehicleEnv(gym.Env):
    def __init__(self, kwargs_sim):
        self.kwargs_sim = kwargs_sim
        self.sim = ShuttleSim(**self.kwargs_sim)
        self.simpy_env = None
        self.render_mode = None
        self.decision_epoch = kwargs_sim['accumulation_time']
        self.visualization_data = {
            'vehicle_states': {},
            'request_data': [],
            'simulation_time': 0,
            'training_data': {
                'latest_reward': None,
                'latest_step': None,
                'steps_count': 0
            }
        }

        capacity = kwargs_sim['vehicle_capacity']
        self.num_vehicles = kwargs_sim['num_vehicles']
        self.action_encoder = capacity * 2 + 1

        self.action_space = spaces.MultiDiscrete([self.action_encoder ** 2] * self.num_vehicles)

        # Observation space: depends on vehicle state + stage flag
        obs_dim_per_vehicle =  5 + 8 * capacity  # To be figured out
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.num_vehicles, obs_dim_per_vehicle),
                                            dtype=np.float32)

        self.movement_interval = 10  # Update vehicle positions every 10 seconds
        self.last_movement_time = 0

        # new request: -1 of no new request, 0 if origin, 1 if destination
        # new request: time to travel to this position
        # new request: time constraint of this request (second from now to complete this)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.simpy_env = self.sim.reset_simulator()

        # initial assignment
        self.sim.update_state(agent_object='vehicle', simulation_run_time=self.decision_epoch)

        obs, info = self.sim.get_observation(agent_object='vehicle')
        return np.array(obs, dtype=np.float32), info

    def decode_action(self, action):
        i = action // self.action_encoder
        j = action % self.action_encoder
        return i, j

    def step(self, actions):
        agent_object = 'vehicle'

        # State Transition
        # Step 1: apply RL agent actions
        decoded_actions = [self.decode_action(a) for a in actions]
        invalid_flags = self.sim.apply_actions(agent_object, decoded_actions)

        # Step 2: Advance vehicle states with continuous movement
        current_time = self.simpy_env.now
        time_since_last_movement = current_time - self.last_movement_time
        
        # Update vehicle positions more frequently than decision epochs
        if time_since_last_movement >= self.movement_interval:
            # Update vehicle states
            self.sim.update_state(agent_object='vehicle', simulation_run_time=self.movement_interval)
            self.last_movement_time = current_time
            
            # Log vehicle movement for debugging
            if hasattr(self.sim, 'network') and hasattr(self.sim.network, 'vehicles'):
                for i, vehicle in enumerate(self.sim.network.vehicles):
                    if hasattr(vehicle, 'current_pos') and hasattr(vehicle, 'next_node'):
                        print(f"Vehicle {i} movement update:")
                        print(f"  Current position: {vehicle.current_pos}")
                        print(f"  Next node: {vehicle.next_node}")
                        print(f"  Traversal process alive: {vehicle.traversal_process is not None and vehicle.traversal_process.is_alive}")

        # Step 3: Get next observation
        obs, info = self.sim.get_observation(agent_object='vehicle')

        # Step 4: Compute rewards
        reward = self.sim.compute_reward(agent_object, obs)

        # Step 5: Check done
        done = self.simpy_env.now >= self.sim.end_time

        return np.array(obs), reward, done, False, info