import os
import sys

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

    sim_kwargs_rl = {
        'run_mode': "rl",
        'trip_date': "2019-09-17",
        'simulation_start_time': 7 * 3600,  # 7 AM
        'simulation_end_time': 10 * 3600,  # 10 AM
        'accumulation_time': 120,  # 2 minutes decision epoch
        'num_vehicles': 1,
        'vehicle_capacity': 10,
        'randomize_vehicle_position': True,
        'euclidean_radius': 800,
        'walking_speed': 1.2,
        'max_walk_time': 600
    }

    sim_kwargs_ilp = {
        'run_mode': "ilp",
        'trip_date': "2019-09-17",
        'simulation_start_time': 7 * 3600,  # 7 AM
        'simulation_end_time': 10 * 3600,  # 10 AM
        'accumulation_time': 120,  # 2 minutes decision epoch
        'num_vehicles': 1,
        'vehicle_capacity': 10,
        'randomize_vehicle_position': True,
        'euclidean_radius': 800,
        'walking_speed': 1.2,
        'max_walk_time': 600
    }


    action_encoder = sim_kwargs_rl['vehicle_capacity'] * 2
    decision_epoch = sim_kwargs_rl['accumulation_time']

    simulator_rl = ShuttleSim(**sim_kwargs_rl)
    simpy_env_rl = simulator_rl.reset_simulator()

    simulator_ilp = ShuttleSim(**sim_kwargs_ilp)
    simpy_env_ilp = simulator_ilp.reset_simulator()


    # load model
    rl_model = None

    # while simulator_rl.env.now < sim_kwargs_rl['simulation_end_time']:
    #     obs, info = simulator_rl.get_observation(agent_object='vehicle')
    #     action = rl_model(obs)
    #     action = decode_action(action, action_encoder)
    #     simulator_rl.apply_actions(agent_object='vehicle', actions=action)
    #     simulator_rl.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)

    while simulator_ilp.env.now < sim_kwargs_ilp['simulation_end_time']:
        simulator_ilp.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)
        simulator_ilp.solve_vrp_ilp()






def decode_action(action, action_encoder):
    i = action // action_encoder
    j = action % action_encoder
    return i, j
if __name__ == "__main__":
    main()