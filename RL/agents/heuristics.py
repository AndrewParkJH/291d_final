
def dispatch_heuristics(current_request, vehicles):
    assigned_pairs = []

    return assigned_pairs

def vehicle_path_plan_heuristics(vehicle_state):
    path_action = []

    return path_action


import numpy as np
from utils.time_manager import TravelTimeManager
import random
random.seed(42)

class HeuristicDispatcher:
    def __init__(self, network, sigma=300.0):
        """
        network: RoadNetwork or FastRoadNetwork with node coordinates available
        sigma: for Gaussian smoothing if needed later
        """
        self.network = network
        self.time_manager = TravelTimeManager(network)
        self.sigma = sigma
        self.node_coordinates = self._build_node_coordinates()

    def _build_node_coordinates(self):
        """Precompute node coordinates."""
        coords = {}
        for node_id, data in self.network.graph.nodes(data=True):
            coords[node_id] = (data['x'], data['y'])  # assuming OSMnx format
        return coords

    def _euclidean_distance(self, node1, node2):
        """Fast 2D Euclidean distance."""
        pos1 = self.node_coordinates.get(node1, (0.0, 0.0))
        pos2 = self.node_coordinates.get(node2, (0.0, 0.0))
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _compute_pickup_insertion_cost(self, vehicle, pickup_node):
        virtual_trip = [vehicle.current_node] + vehicle.trip_sequence
        min_insertion_cost = float('inf')

        for i in range(1, len(virtual_trip) + 1):
            prev = virtual_trip[i - 1]
            next = virtual_trip[i] if i < len(virtual_trip) else None

            travel_time_prev_pickup, _, _, _ = self.time_manager.query(prev, pickup_node)
            travel_time_pickup_next, _, _, _ = self.time_manager.query(pickup_node, next) if next else (
            0.0, 0.0, [], [])

            travel_time_prev_next, _, _, _ = self.time_manager.query(prev, next) if next else (0.0, 0.0, [], [])

            insertion_cost = (travel_time_prev_pickup + travel_time_pickup_next) - travel_time_prev_next

            if insertion_cost < min_insertion_cost:
                min_insertion_cost = insertion_cost

        return min_insertion_cost

    def assign_requests(self, requests, vehicles):
        assigned_pairs = []
        assigned_ids = set()

        for vehicle in vehicles:
            if vehicle.current_num_pax >= vehicle.max_capacity:
                continue

            if vehicle.new_request:
                continue

            # ➔ Add randomness based on trip count
            trip_count = len(vehicle.trip_sequence)
            skip_probability = min(0.1 * trip_count, 0.95)  # cap at 90%

            if random.random() < skip_probability:
                continue  # Skip assigning this vehicle

            if not vehicle.trip_sequence:
                ref_nodes = [vehicle.current_node]
            else:
                ref_nodes = [vehicle.current_node]+[sequence['node_id'] for sequence in vehicle.trip_sequence]

            pickup_scores = []
            for req in requests:
                if req['request_id'] in assigned_ids:
                    continue
                pickup_node = req['pu_osmid']
                min_dist = min(self._euclidean_distance(pickup_node, trip_node) for trip_node in ref_nodes)
                pickup_scores.append((min_dist, req))

            top_candidates = sorted(pickup_scores, key=lambda x: x[0])[:5]

            # random selection
            _, chosen_request = random.choice(top_candidates)

            # compute time
            # insertion_costs = []
            # for dist, req in top_candidates:
            #     pickup_node = req['pu_osmid']
            #     cost = self._compute_pickup_insertion_cost(vehicle, pickup_node)
            #     insertion_costs.append((cost, req))
            #
            # insertion_costs.sort(key=lambda x: x[0])
            #
            # if len(insertion_costs) >= 3:
            #     _, chosen_request = insertion_costs[2]
            # elif insertion_costs:
            #     _, chosen_request = insertion_costs[0]
            # else:
            #     continue

            assigned_pairs.append((chosen_request, vehicle))
            assigned_ids.add(chosen_request['request_id'])

        return assigned_pairs
