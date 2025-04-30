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

        capacity = kwargs_sim['vehicle_capacity']
        self.num_vehicles = kwargs_sim['num_vehicles']

        # Action space: no-op (0) or each vehicle can insert at (1~max_trip_length+1)
        self.action_space = spaces.MultiDiscrete([capacity*2 + 2] * self.num_vehicles)

        # Observation space: depends on vehicle state + stage flag
        obs_dim_per_vehicle =  5 + 6 * capacity  # To be figured out
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.num_vehicles, obs_dim_per_vehicle),
                                            dtype=np.float32)

        # new request: -1 of no new request, 0 if origin, 1 if destination
        # new request: time to travel to this position
        # new request: time constraint of this request (second from now to complete this)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._initialize_simulator()
        self.sim.start_control_trigger() # start background simpy processes

        # initial assignment
        self.sim.request_accumulate()
        self.sim.assign_requests_heuristics()

        obs, info = self.sim.get_observation(agent_object='vehicle')
        return np.array(obs, dtype=np.float32), info

    def step(self, actions):
        agent_object = 'vehicle'

        # State Transition
        # Step 1: apply RL agent actions
        invalid_flags = self.sim.apply_actions(agent_object, actions)

        # Step 2: Advance vehicle states
        self.sim.update_state(agent_object) # Tell vehicles to start travelling to inserted trips

        # Step 3:Advance simulation by event step (accumulation_time)
        while not self.sim.control_event.triggered:
            self.simpy_env.step()

        self.sim.request_accumulate() # accumulate all requests that happened during this time

        # Step 4: Assign requests using heuristic (if active)
        self.sim.assign_requests_heuristics()

        # Step 5: Get next observation
        obs, info = self.sim.get_observation(agent_object='vehicle')

        # Step 6: Compute rewards
        reward = self.sim.compute_reward(agent_object)

        # Step 7: Check done
        done = self.simpy_env.now >= self.sim.end_time

        self.sim.control_event = self.simpy_env.event()  # reset control_events

        return np.array(obs), reward, done, False, info


    def _initialize_simulator(self):
        self.sim.reset()
        self.simpy_env = self.sim.env