import simpy
from simulator.road_network import RoadNetwork
import os
import pandas as pd

BASE_DIR = os.curdir
DATA_DIR = os.path.join(BASE_DIR, "data/episode_2019-09-17.csv")
GRAPH_FILE_DIR = os.path.join(BASE_DIR, "data/road_network/sf_road_network.graphml")

class ShuttleSim():
    def __init__(self, trip_date, simulation_start_time=7*3600, simulation_end_time=36000,
                 accumulation_time=120, num_vehicles=40, randomize_vehicle_position=True,
                 vehicle_capacity=10, debug=False):
        self.env = None
        self.network = None
        self.request_df = None
        self._initialize_simulation()

        self.accumulation_time = accumulation_time
        self.start_time = simulation_start_time
        self.end_time = simulation_end_time
        self.num_vehicles = num_vehicles
        self.randomize_vehicle_position = randomize_vehicle_position
        self.vehicle_capacity = vehicle_capacity

        self.data_dir = os.path.join(BASE_DIR, f"data/episode_{trip_date}.csv")

        self.dispatch_trigger = None
        self.debug = debug

        self.current_request = []

    def step(self, dispatch_fn, vehicle_policy_fn):
        print(f"[{self.env.now}] Tick")
        self.request_accumulate()
        yield self.env.timeout(self.accumulation_time)
        print(f"[{self.env.now}] Dispatch event triggered")

        # 1. Assign requests using dispatch_fn (e.g., greedy for Phase 1)
        assigned_pairs = dispatch_fn(self.current_request, self.network.vehicles)
        for request, vehicle in assigned_pairs:
            vehicle.add_request(request)

        # 2. Let each vehicle apply its policy (route planning)
        for vehicle in self.network.vehicles:
            if vehicle.has_new_request():
                state = vehicle.get_state()
                action = vehicle_policy_fn(state)
                vehicle.apply_policy_action(action)

        # 3. Move vehicles forward
        for vehicle in self.network.vehicles:
            vehicle.advance_to_next_stop()

        # 4. Compute reward
        rewards = {v.vehicle_id: v.compute_reward() for v in self.network.vehicles}

        # 5. Clear requests for next step
        self.current_request.clear()

        return rewards

    def request_accumulate(self):
        time_now = self.env.now
        time_next = self.env.now + self.accumulation_time
        # accumulate requests here...
        request_df = self.request_df[
            (self.request_df["req_time"] >= time_now) &
            (self.request_df["req_time"] < time_next)
            ]

        for _, request in request_df.iterrows():
            self.current_request.append(request.to_dict())

        if self.debug:
            print(f"\t{self.env.now}: triggered request accumulate for ({time_now, time_next}) -  current request {self.current_request}")

    def _initialize_simulation(self):
        self.env = simpy.Environment(initial_time=self.start_time)
        self.network = RoadNetwork(env=self.env,
                                   num_vehicles=self.num_vehicles,
                                   vehicle_capacity=self.vehicle_capacity,
                                   randomize_vehicles=self.randomize_vehicle_position,
                                   graph_path=GRAPH_FILE_DIR)
        self.request_df = pd.read_csv(os.path.join(DATA_DIR))

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
            if self.run_mode == "benchmark":
                # self.ilp_solver.solve(self.current_request)  # integer linear program

                # reinforcement learning

                print(f"\tbenchmark logic performed")
                # benchmark logic here

            # clear accumulated requests after dispatching
            self.current_request.clear()  # add conditional (clear served requests only) later