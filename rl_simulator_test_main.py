import simpy
import os
import pandas as pd
from simulator.road_network import RoadNetwork
from simulator.sav_simulator_rl import ShuttleSim
from RL.agents.heuristics import dispatch_heuristics, vehicle_path_plan_heuristics
from RL.environment.multi_vehicle_env import MultiVehicleEnv

sim_kwargs = {
    'trip_date': "2019-09-17",
    'simulation_start_time': 7*3600,
    'simulation_end_time': 10*3600,
    'accumulation_time': 120,
    'num_vehicles': 40,
    'randomize_vehicle_position': True,
    'randomize_vehicle_passengers': False,
    'vehicle_capacity': 10
}

DEBUG = True
BASE_DIR = os.curdir
DATA_DIR = os.path.join(BASE_DIR, "data/episode_2019-09-17.csv")
GRAPH_FILE_DIR = os.path.join(BASE_DIR, "data/road_network/sf_road_network.graphml")
SIM_START_TIME = 25200 # 7am == 25200 seconds
SIM_END_TIME = 36000 # 10am == 36000 seconds
RUN_MODE = "benchmark"

def main():
    # env = simpy.Environment(initial_time=SIM_START_TIME)
    # road_network = RoadNetwork(env=env,
    #                     num_vehicles=40,
    #                     vehicle_capacity=2,
    #                     randomize_vehicles=True,
    #                     graph_path=GRAPH_FILE_DIR)
    #
    # request_df = pd.read_csv(os.path.join(DATA_DIR))

    simulator = ShuttleSim(**sim_kwargs)
    simulator.reset()
    simulator.start_simulation()



    # env.process(sim_env.step())
    # env.process(sim_env.trigger_dispatch())
    #
    # env.run(until=SIM_END_TIME)

if __name__ == "__main__":
    from data_preprocess import DataLoader
    main()

