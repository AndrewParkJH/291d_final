import networkx as nx
from itertools import combinations, permutations
from benchmark.RVGraphGenerator import generate_rv_graph, visualize_rv_graph, visualize_rv_graph_with_annotations
from benchmark.RTVGraphGenerator import generate_rtv_graph, greedy_assignment, visualize_assignment
from benchmark.OptimalAssignment import assignment_ilp

class ILP_Solver:
    def __init__(self, env, network, omega=900, max_delay=600):
        self.env = env
        self.network = network
        self.omega = omega # latest acceptable pickup waiting time
        self.max_delay = max_delay
        self.request_id = 1
        self.vehicles = []


    def solve(self, request_dict):
        print(f"{self.env.now}: Solving ILP problem for {len(request_dict)} requests")

        self.update_vehicle_state()
        vehicles = self.vehicles
        requests = self.process_requests(request_dict)

        # Validate requests (ensures that all requests are reachable by each vehicle in the network)
        reachable_requests, unreachable_requests = self.validate_request_reachability(self.network.graph, requests, vehicles)

        print(f"\t Reachable requests: {len(reachable_requests)}")
        print(f"\t Unreachable requests: {len(unreachable_requests)}")

        rv_graph = generate_rv_graph(
            graph=self.network.graph,
            vehicles=vehicles,
            requests=reachable_requests,
            current_time=self.env.now,
            max_capacity=self.network.max_vehicle_capacity,
            max_delay=self.max_delay,
            prune_edges=False,
            top_k=10,
            debug=False
        )

        rtv_graph = generate_rtv_graph(
            rv_graph=rv_graph,
            vehicles=vehicles,
            requests=requests,
            graph=self.network.graph,
            max_capacity=self.network.max_vehicle_capacity,
            max_delay=self.max_delay,
            debug=False)

        # Perform Greedy Assignment
        initial_assignment = greedy_assignment(
            rtv_graph=rtv_graph,
            vehicles=vehicles,
            debug=False)

        for vehicle_id, trip_id in initial_assignment.items():
            print(f"Vehicle {vehicle_id} is assigned to Trip {trip_id}.")

        optimized_assignment = assignment_ilp(
            rtv_graph=rtv_graph,
            vehicles=vehicles,
            requests=requests,
            greedy_assignment=initial_assignment,
            cost_penalty=1000,
            time_limit=30,
            gap=0.001)

        return 0

    def process_requests(self, request_dict):
        requests = []
        for request in request_dict:
            time = request['time_departure']
            origin_coordinates = (request['x_x'], request['y_x'])
            destination_coordinates = (request['x_y'], request['y_y'])

            origin_node = self.network.find_nearest_node_from_coodrinate(origin_coordinates)
            destination_node = self.network.find_nearest_node_from_coodrinate(destination_coordinates)

            t_r_star = self.network.find_shortest_travel_time(origin_node, destination_node)

            new_request = {
                "id": self.request_id,
                "o_r": origin_node,
                "d_r": destination_node,
                "t_r^r": time,
                "t_r^pl": time + self.omega, # latest acceptable pick up time
                "t_r^p": -1,  # pick up time
                "t_r^d": -1,  # expected drop off
                "t_r^*": 0 + t_r_star  # Earliest drop-off time
                # t_r^p and t_r^d will be computed during vehicle matching
            }

            self.request_id += 1

            requests.append(new_request)

        return requests

    def update_vehicle_state(self):
        vehicles = self.network.vehicles # list of Vehicle classes

        processed_vehicles = []

        for vehicle in vehicles:
            vehicle = {
                "id": vehicle.vehicle_id,  # Unique vehicle ID
                "q_v": vehicle.current_node,  # Randomly assigned node (position)
                "t_v": self.env.now,  # Current time (always 0 initially)
                "passengers": vehicle.current_num_pax,  # Random passenger count
                "trip_set": vehicle.trips  # Initial trip set for current passengers
            }

            processed_vehicles.append(vehicle)


        self.vehicles = processed_vehicles

    @staticmethod
    def validate_request_reachability(graph, requests, vehicles):
        """
        Validates that each request's origin is reachable by at least one vehicle.

        Parameters:
        - graph: networkx.Graph, road network with precomputed travel times
        - requests: list of dicts, each containing request attributes
        - vehicles: list of dicts, each containing vehicle attributes

        Returns:
        - reachable_requests: list of dicts, requests that are reachable by at least one vehicle
        - unreachable_requests: list of dicts, requests that are unreachable
        """
        reachable_requests = []
        unreachable_requests = []

        for request in requests:
            o_r = request["o_r"]  # Origin node of the request
            is_reachable = False
            for vehicle in vehicles:
                q_v = vehicle["q_v"]  # Current node of the vehicle
                # Check if there's a path from vehicle's position to the request origin
                if nx.has_path(graph, source=q_v, target=o_r):
                    is_reachable = True
                    break  # No need to check further vehicles if one is reachable

            if is_reachable:
                reachable_requests.append(request)
            else:
                unreachable_requests.append(request)

        return reachable_requests, unreachable_requests



