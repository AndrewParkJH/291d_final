import simpy
from simulator.road_network import RoadNetwork
from RL.agents.heuristics import dispatch_heuristics, vehicle_path_plan_heuristics
from RL.agents.heuristics import HeuristicDispatcher
import osmnx as ox
import os
import pandas as pd
import numpy as np
from passenger_cluster import clusterGenerator
from collections import Counter
import logging


logger = logging.getLogger(__name__)

BASE_DIR = os.curdir
GRAPH_FILE_DIR = os.path.join(BASE_DIR, "data/road_network/sf_road_network.graphml")
APPLY_CLUSTER = False

class ShuttleSim:
    def __init__(self, run_mode='rl',env=None, network=None, dispatcher=None, graph = None, request_df=None,
                 trip_date='2019-09-17', simulation_start_time=7*3600, simulation_end_time=36000,
                 accumulation_time=120, num_vehicles=40,
                 randomize_vehicle_position=True,
                 randomize_vehicle_passengers=False,
                 vehicle_capacity=10, debug=False,
                 euclidean_radius=800, walking_speed=1.2, max_walk_time=600):

        self.run_mode = run_mode
        self.env = env
        self.network = network
        self.fast_network = None
        self.dispatcher = dispatcher

        # read request_df
        data_dir = os.path.join(BASE_DIR, f"data/episode_{trip_date}.csv")
        self.graph = ox.load_graphml(GRAPH_FILE_DIR) if graph is None else graph
        self.request_df = pd.read_csv(data_dir) if request_df is None else request_df

        # cluster generator
        self.cluster_generator = clusterGenerator(self, current_request_df=None,
                                                  euclidean_radius=euclidean_radius,
                                                  walking_speed=walking_speed,
                                                  max_walk_time=max_walk_time)
        self.clusters_over_time = {}

        # simulation state data
        self.current_request            = []
        self.request_id                 = 1

        # simulation parameters
        self.accumulation_time          = accumulation_time
        self.start_time                 = simulation_start_time
        self.end_time                   = simulation_end_time
        self.simulation_duration        = simulation_end_time - simulation_start_time
        self.num_vehicles               = num_vehicles
        self.vehicle_capacity           = vehicle_capacity
        self.dispatch_trigger           = None
        self.control_event              = None
        self.debug                      = debug
        self.randomize_vehicle_position = randomize_vehicle_position
        self.randomize_vehicle_passengers = randomize_vehicle_passengers

        if self.simulation_duration <= 0:
            raise ValueError("Invalid simulation start/end time")

    def reset_simulator(self):
        self.env = self._initialize_simulation()
        return self.env

    def start_control_trigger(self, simulation_run_time):
        """
        start background simpy processes
        :return:
        """
        self.control_event = self.env.event()
        self.env.process(self.control_trigger(simulation_run_time))

    def control_trigger(self, simulation_run_time):
        """
        tracks decision epoch as accumulation time (gym.env takes this as a single timestep)
        """
        while True:
            yield self.env.timeout(simulation_run_time)
            self.control_event.succeed()

    def run_simulation(self, simulation_run_time):
        self.env.run(until=self.env.now+simulation_run_time)

        # while not self.control_event.triggered:
        #     self.env.step()

    def request_accumulate(self):
        time_now = self.env.now
        time_next = self.env.now + self.accumulation_time
        # accumulate requests here...
        request_df = self.request_df[
            (self.request_df["req_time"] >= time_now) &
            (self.request_df["req_time"] < time_next)
            ]

        if self.debug:
            print(f"\t{self.env.now}: triggered request accumulate for ({time_now, time_next}) -  current request {self.current_request}")

        return request_df.to_dict('records')

    def assign_requests_heuristics(self):
        pairs = self.dispatcher.assign_requests(self.current_request, self.network.vehicles)

        for request, vehicle in pairs :
            vehicle.add_request(request)

    def insert_requests(self, agent_object, actions):
        """
        Insert assigned requests to vehicles
        - if invalid flag is returned, the insertion is invalid
        - action consists of a tuple [1,2]
            index 1: insertion index of pickup location
            index 2: insertion index of dropoff location
        """
        invalid_flags = np.zeros(self.num_vehicles)

        if agent_object == 'vehicle':
            for idx, (vehicle, action) in enumerate(zip(self.network.vehicles, actions)):

                flag = vehicle.apply_action(action)
                invalid_flags[idx] = flag # flagging penalties for invalid action

    def apply_actions(self, agent_object, actions):
        invalid_flags = np.zeros(self.num_vehicles)

        if agent_object == 'vehicle':
            for idx, (vehicle, action) in enumerate(zip(self.network.vehicles, actions)):
                # if action < 0 or action > 27:
                #     raise ValueError("Invalid Action Selection for Vehicle Agent")

                flag = vehicle.insert_request(action)
                invalid_flags[idx] = flag # flagging penalties for invalid action

        elif agent_object == 'network':
            invalid_flags = 0

        return invalid_flags

    def update_state(self, agent_object, simulation_run_time):
        if agent_object == 'vehicle':
            # self.start_control_trigger(simulation_run_time)  # start control trigger to keep 120 s interval
            self.current_request = self.request_accumulate()
            # clustering

            if APPLY_CLUSTER:
                self.cluster_generator.update_aggregated_requests(self.current_request)
                clusters = self.cluster_generator.extract_clusters()
                clusters = [clust for clust in clusters if clust['num_req'] > 1]
                total_num_req = sum(clust['num_req'] for clust in clusters)

                # Suppose clusters is your list of dicts
                num_req_list = [clust['num_req'] for clust in clusters]
                distribution = Counter(num_req_list)

                # Convert to a list of (num_req, count)
                distribution_list = sorted(distribution.items())
                log_string = ', '.join(f'{num_req}: {count} clusters' for num_req, count in distribution_list)
                logger.info(f"[{self.env.now}-{self.env.now + simulation_run_time}], cluster_length: {len(clusters)}, size_aggregated_request: {len(self.current_request)}, total_num_pass_clustered: {total_num_req}, Cluster distribution by num_req → {log_string}")
                # print(f"[{self.env.now}] Found {len(clusters)} clusters.")
                self.clusters_over_time[self.env.now] = clusters

            self.assign_requests_heuristics()
            self.network.update_vehicle_state()
            self.env.run(until=self.env.now + simulation_run_time)

        elif agent_object == 'network':
            self.network.update_network_state()

    def compute_reward(self, agent_object, obs, invalid_flags=None):
        reward = 0.0

        if agent_object == 'vehicle':
            # Compute each vehicle's reward
            rewards = [self.compute_vehicle_reward(single_obs, vehicle) for single_obs, vehicle in zip(obs, self.network.vehicles)]

            # Use average reward across vehicles as the single scalar reward
            return float(np.mean(rewards))

        elif agent_object == 'network':
            reward = 0.0

        return reward

    def compute_vehicle_reward(self, obs, vehicle):
        """
        Compute reward for vehicles
        :param obs:
        :return:
        """
        # unpack observation
        stage                       = obs[0]
        invalid_counter             = obs[1]
        normalized_stop_count       = obs[2]
        normalized_remaining_cap    = obs[3]
        time_constraint             = obs[4]
        dist_from_trip_seq_o        = obs[5:5+2*self.vehicle_capacity]
        dist_from_trip_seq_d        = obs[5+2*self.vehicle_capacity:5+4*self.vehicle_capacity]
        group_size                  = obs[5+4*self.vehicle_capacity:5+6*self.vehicle_capacity]
        normalized_remaining_time   = obs[5+6*self.vehicle_capacity:5+8*self.vehicle_capacity]

        # if normalized_stop_count>0:
            # print('')
        # Initialize reward components
        reward = 0.0

        # 1. **Invalid action penalties** (e.g., wrong insertion or unnecessary action)
        reward += -invalid_counter
        reward += -(vehicle.deg_wrong_o_1+vehicle.deg_wrong_o_2+vehicle.deg_wrong_o_3+vehicle.deg_wrong_o_4
                    + vehicle.deg_wrong_d_1+vehicle.deg_wrong_d_2+vehicle.deg_wrong_d_3+vehicle.deg_wrong_d_4)

        # 2. **Pickup event reward** (passengers picked up and on board)
        reward += normalized_stop_count
        reward += normalized_remaining_cap

        time_penalty = [x * y for x, y in zip(group_size, normalized_remaining_time)]
        reward += sum(neg_t for neg_t in time_penalty if neg_t < 0)*20

        if time_constraint < 0:
            reward += time_constraint

        return reward

    def get_observation(self, agent_object):
        """
        TODO: get observations
        """
        obs = None
        info = None

        if agent_object=='vehicle':
            obs, info = self.network.get_vehicle_state(time_normalizer=self.simulation_duration)

        elif agent_object=='network': # for central agent learning
            obs, info  = self.network.get_network_state()

        return obs, info

    def solve_vrp_ilp(self, agent_object='vehicle'):
        if agent_object == 'vehicle':
            for idx, vehicle in enumerate(self.network.vehicles):
                invalid_flags = vehicle.optimal_trip_sequence_ilp()

        elif agent_object == 'network':
            invalid_flags = 0

        return invalid_flags

    def _initialize_simulation(self):
        simpy_env = simpy.Environment(initial_time=self.start_time) # starts at 1 accumulation for initial observation
        self.network = RoadNetwork(env=simpy_env,
                                   num_vehicles=self.num_vehicles,
                                   vehicle_capacity=self.vehicle_capacity,
                                   randomize_vehicle_position=self.randomize_vehicle_position,
                                   randomize_vehicle_passengers=self.randomize_vehicle_passengers,
                                   graph=self.graph, fast_network=self.fast_network)
        self.fast_network = self.network.fast_network
        self.dispatcher = HeuristicDispatcher(self.network)

        return simpy_env

    def _process_requests(self, request_dicts):
        processed_requests = []
        for req in request_dicts:
            origin_node = int(req['pu_osm'])  # pickup node OSM ID
            destination_node = int(req['do_osm'])  # dropoff node OSM ID
            request_time = req['req_tim']  # when the request appears
            num_passengers = req['num_passengers']  # passengers field (default 1)
            deadline = req['deadline']
            distance = req['travel_distance']
            earliest_time = req['earliest_time']
            fare = req['fare']
            pu_lon = req['pu_lon']
            do_lat = req['do_lat']
            pu_taz = req['pu_taz']
            do_taz = req['do_taz']

            # find the shortest travel time between pickup and dropoff
            t_r_star = self.network.find_shortest_travel_time(origin_node, destination_node)

            new_request = {
                "request_id": self.request_id,
                "oid": origin_node,
                "did": destination_node,
                "time": request_time,  # request time
                "deadline": deadline,
                "distance": distance,
                "fare":fare,
                "pickup_time": -1,  # pickup time (not yet picked up)
                "dropoff_time": -1,  # dropoff time (not yet dropped)
                "earliest_time": earliest_time,  # estimated earliest dropoff time
                "num_passengers": num_passengers  # number of passengers
            }

            processed_requests.append(new_request)
            self.request_id += 1

        return processed_requests

    """
    Run entire simulation in one go
    """
    def run_static_simulation(self, end_time=None):
        self._initialize_simulation()
        self.end_time = end_time if end_time is not None else self.end_time
        self.dispatch_trigger = self.env.event()

        self.env.process(self.runner())
        self.env.process(self.trigger_dispatch())

        self.env.run(until=self.end_time)

    def runner(self):
        # request accumulate
        while True:
            print(f"[{self.env.now}] Tick")
            self.request_accumulate()
            yield self.env.timeout(self.accumulation_time)

            # add request cluster
            self.dispatch_trigger.succeed()
            print(f"[{self.env.now}] Dispatch event triggered")

            self.dispatch_trigger = self.env.event()

            if self.env.now >= self.end_time:
                break

    def trigger_dispatch(self):
        while True:
            print(f"{self.env.now}: Waiting for customer accumulation")
            yield self.dispatch_trigger
            print(f"[{self.env.now}] Tick | {len(self.current_request)} requests")

            # dispatch logic here...
            # benchmark: ilp logic
            # if self.run_mode == "benchmark":
                # self.ilp_solver.solve(self.current_request)  # integer linear program

                # reinforcement learning

                # print(f"\tbenchmark logic performed")
                # benchmark logic here

            # clear accumulated requests after dispatching
            self.current_request.clear()  # add conditional (clear served requests only) later