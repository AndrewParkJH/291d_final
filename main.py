import simpy
import os
import pandas as pd
from simulator.road_network import RoadNetwork
from simulator.sav_simulator import ShuttleSim

RoadNetworkBuildq

DEBUG = True
BASE_DIR = os.curdir
DATA_DIR = os.path.join(BASE_DIR, "data/sf_mtc_od_sample.csv")
GRAPH_FILE_DIR = os.path.join(BASE_DIR, "data/road_network/sf_road_network.graphml")
SIM_START_TIME = 25200 # 7am == 25200 seconds
SIM_END_TIME = 36000 # 10am == 36000 seconds
RUN_MODE = "benchmark"

def main():
    env = simpy.Environment(initial_time=SIM_START_TIME)
    road_network = RoadNetwork(env=env,
                        num_vehicles=40,
                        vehicle_capacity=2,
                        randomize_vehicles=True,
                        graph_path=GRAPH_FILE_DIR)

    request_df = pd.read_csv(os.path.join(DATA_DIR))

    sim_env = ShuttleSim(env=env,
                         network=road_network,
                         run_mode=RUN_MODE,
                         request_df=request_df)

    env.process(sim_env.step())
    env.process(sim_env.trigger_dispatch())

    env.run(until=SIM_END_TIME)

if __name__ == "__main__":
    main()

