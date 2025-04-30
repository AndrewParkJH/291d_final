import simpy
from simulator.road_network import RoadNetwork
from RL.agents.heuristics import dispatch_heuristics, vehicle_path_plan_heuristics
from RL.agents.heuristics import HeuristicDispatcher
import osmnx as ox
import os
import pandas as pd
import numpy as np

BASE_DIR = os.curdir
GRAPH_FILE_DIR = os.path.join(BASE_DIR, "data/road_network/sf_road_network.graphml")

class ShuttleSim:
    def __init__(self, env=None, network=None, dispatcher=None, graph = None, request_df=None,
                 trip_date='2019-09-17', simulation_start_time=7*3600, simulation_end_time=36000,
                 accumulation_time=120, num_vehicles=40,
                 randomize_vehicle_position=True,
                 randomize_vehicle_passengers=False,
                 vehicle_capacity=10, debug=False):

        self.env = env
        self.network = network
        self.dispatcher = dispatcher

        # read request_df
        data_dir = os.path.join(BASE_DIR, f"data/episode_{trip_date}.csv")
        self.graph = ox.load_graphml(GRAPH_FILE_DIR) if graph is None else graph
        self.request_df = pd.read_csv(data_dir) if request_df is None else request_df

        # simulation state data
        self.current_request = []

        # simulation parameters
        self.accumulation_time = accumulation_time
        self.start_time = simulation_start_time
        self.end_time = simulation_end_time
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.dispatch_trigger = None
        self.control_event = None
        self.debug = debug
        self.randomize_vehicle_position = randomize_vehicle_position
        self.randomize_vehicle_passengers = randomize_vehicle_passengers

    def reset(self):
        self._initialize_simulation()

    def start_control_trigger(self):
        self.control_event = self.env.event()
        self.env.process(self.control_trigger())

    def control_trigger(self):
        """
        tracks decision epoch as accumulation time (gym.env takes this as a single timestep)
        """
        while True:
            yield self.env.timeout(self.accumulation_time)
            self.control_event.succeed()

    def request_accumulate(self):
        time_now = self.env.now
        time_next = self.env.now + self.accumulation_time
        # accumulate requests here...
        request_df = self.request_df[
            (self.request_df["req_time"] >= time_now) &
            (self.request_df["req_time"] < time_next)
            ]

        self.current_request = request_df.to_dict('records')
        if self.debug:
            print(f"\t{self.env.now}: triggered request accumulate for ({time_now, time_next}) -  current request {self.current_request}")

    def assign_requests_heuristics(self):
        pairs = self.dispatcher.assign_requests(self.current_request, self.network.vehicles)

        for request, vehicle in pairs :
            vehicle.add_request(request)

    def apply_actions(self, agent_object, actions):
        invalid_flags = np.zeros(self.num_vehicles)

        if agent_object == 'vehicle':
            for idx, (vehicle, action) in enumerate(zip(self.network.vehicles, actions)):
                if action < 0 or action > 27:
                    raise ValueError("Invalid Action Selection for Vehicle Agent")

                flag = vehicle.apply_action(action)
                invalid_flags[idx] = flag # flagging penalties for invalid action

        elif agent_object == 'network':
            invalid_flags = 0

        return invalid_flags

    def update_state(self, agent_object):
        if agent_object == 'vehicle':
            self.network.update_vehicle_state()

        elif agent_object == 'network':
            self.network.update_network_state()

    def compute_reward(self, agent_object, invalid_flags=None):
        """
        TODO: Reward design
        """

        if agent_object == 'vehicle':
            # Compute each vehicle's reward
            vehicle_rewards = [v.compute_reward() for v in self.network.vehicles]

            # Use average reward across vehicles as the single scalar reward
            return float(np.mean(vehicle_rewards))

        elif agent_object == 'network':
            reward = 0.0

    def get_observation(self, agent_object):
        """
        TODO: get observations
        """
        if agent_object=='vehicle':
            obs = self.network.get_vehicle_state()

        elif agent_object=='network': # for central agent learning
            obs = self.network.get_network_state()

        return obs, {}

    # def step(self):
    #     print(f"[{self.env.now}] Tick")
    #     self.request_accumulate()
    #     yield self.env.timeout(self.accumulation_time)
    #     print(f"[{self.env.now}] Dispatch event triggered")
    #
    #     # 1. Assign requests using dispatch_fn (e.g., greedy for Phase 1)
    #     assigned_pairs = dispatch_heuristics(self.current_request, self.network.vehicles)
    #     for request, vehicle in assigned_pairs:
    #         vehicle.add_request(request)
    #
    #     # 2. Let each vehicle apply its policy (route planning)
    #     for vehicle in self.network.vehicles:
    #         if vehicle.new_request is not None:
    #             state = vehicle.get_state()
    #             action = vehicle_path_plan_heuristics(state)
    #             vehicle.apply_action(action)
    #
    #     # 3. Move vehicles forward
    #     for vehicle in self.network.vehicles:
    #         vehicle.advance_to_next_stop()
    #
    #     # 4. Compute reward
    #     rewards = {v.vehicle_id: v.compute_reward() for v in self.network.vehicles}
    #
    #     # 5. Clear requests for next step
    #     self.current_request.clear()
    #
    #     return rewards

    def _initialize_simulation(self):
        self.env = simpy.Environment(initial_time=self.start_time+self.accumulation_time) # starts at 1 accumulation for initial observation
        self.network = RoadNetwork(env=self.env,
                                   num_vehicles=self.num_vehicles,
                                   vehicle_capacity=self.vehicle_capacity,
                                   randomize_vehicle_position=self.randomize_vehicle_position,
                                   randomize_vehicle_passengers=self.randomize_vehicle_passengers,
                                   graph=self.graph)
        self.dispatcher = HeuristicDispatcher(self.network)

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

    def get_vehicle_state(self):
        vehicle_state = []
        info = {}
        for vehicle in self.network.vehicles:
            state = vehicle.get_state()
            vehicle_state.append(state)

        return vehicle_state, info

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