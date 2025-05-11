import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from visualizations.live.visualization_server import start_visualization_server, socketio, training_data, vehicle_states, request_data, simulation_time
from datetime import datetime

from simulator.sav_simulator_rl import ShuttleSim

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  # Insert at the beginning of the path

import logging
import datetime

log_out_path = os.path.join(project_root, 'output','logs')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_out_path, f"simulation_{timestamp}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():

    sim_kwargs = {
        'trip_date': "2019-09-17",
        'simulation_start_time': 7 * 3600,  # 7 AM
        'simulation_end_time': 10 * 3600,  # 10 AM
        'accumulation_time': 600,  # 2 minutes decision epoch
        'num_vehicles': 1,
        'vehicle_capacity': 10,
        'randomize_vehicle_position': True,
        'euclidean_radius': 800,
        'walking_speed': 1.2,
        'max_walk_time': 600
    }

    decision_epoch = sim_kwargs['accumulation_time']

    simulator = ShuttleSim(**sim_kwargs)
    simpy_env = simulator.reset_simulator()

    while simulator.env.now < sim_kwargs['simulation_end_time']:
        simulator.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)

    cluster_result = simulator.clusters_over_time

if __name__ == "__main__":
    main()