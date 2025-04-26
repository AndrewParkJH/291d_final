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
        self.sim = None
        self.simpy_env = None

        action_dim = -1 # to be figured out later
        num_vehicles = kwargs_sim['num_vehicles']
        self.observation_space = spaces.Box()
        self.action_space = spaces.MultiDiscrete([action_dim]*num_vehicles)

    def reset(self):
        # super().reset(seed=None)
        self.sim, self.simpy_env = self._initialize_simulator(self)
        obs, info = self.sim.get_vehicle_state()

        return np.array(obs), info

    def step(self, actions):

        obs, info = self.sim.get_vehicle_state()

        for vehicle, action in zip(self.sim.network.vehicles, actions):
            vehicle.apply_policy_action(action)
        return -1

    @staticmethod
    def _initialize_simulator(self):
        simulator = ShuttleSim(**self.kwargs_sim)
        simpy_env = simulator.env

        return simulator, simpy_env