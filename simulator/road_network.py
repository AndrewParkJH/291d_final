import osmnx as ox
import networkx as nx
from simulator.vehicle import Vehicle

DEFAULT_GRAPH_PATH = "./data/road_network/sf_road_network.graphml"


class RoadNetwork:
    def __init__(self, env, num_vehicles=40, vehicle_capacity=10,
                 randomize_vehicle_position=True,
                 randomize_vehicle_passengers=False,
                 randomize_vehicles=True, graph=None):
        self.env = env
        self.graph = ox.load_graphml(DEFAULT_GRAPH_PATH) if graph is None else graph
        self.fast_network = FastRoadNetwork(self.graph)
        self.vehicles = self.initialize_vehicle(num_vehicles, vehicle_capacity, randomize_vehicle_position, randomize_vehicle_passengers)
        self.max_vehicle_capacity = vehicle_capacity

        self.network_state = None

    def initialize_vehicle(self, num_vehicles, vehicle_capacity, randomize_position, randomize_passengers):
        """
        initializes vehicle positions
        """
        vehicles = []

        for i in range(1, num_vehicles+1):
            vehicle = Vehicle(env=self.env, network=self, vid=i,max_capacity=vehicle_capacity,
                              randomize_position=randomize_position, randomize_passengers=randomize_passengers)
            vehicles.append(vehicle)

        return vehicles

    def update_vehicle_state(self):
        """
        Update state based on the action of the vehicle
        :return:
        """
        for vehicle in self.vehicles:
            vehicle.update_state()

    def get_vehicle_state(self):
        vehicle_states = []
        for vehicle in self.vehicles:
            state = vehicle.get_state()
            vehicle_states.append(state)
        return vehicle_states

    def get_vehicle_rewards(self):
        reward = []
        for vehicle in self.vehicles:
            reward.append(vehicle.compute_reward())

        return reward

    def get_node_coordinate(self, node_id):
        """
        get (lon, lat) position of the given node
        """
        node = self.graph.nodes[node_id] # returns dict
        try:
            return node['x'], node['y']
        except:
            raise Warning(f'queried node does not exist in the graph : {node_id}')

    def find_nearest_node_from_coordinate(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        node = ox.distance.nearest_nodes(self.graph, X=x, Y=y)
        return node

    def find_shortest_path_route(self, origin_node, destination_node):
        route = ox.shortest_path(self.graph, orig=origin_node, dest=destination_node, weight="travel_time")
        return route

    def find_shortest_travel_time(self, origin_node, destination_node):
        time = nx.shortest_path_length(self.graph, source=origin_node, target=destination_node, weight="travel_time")
        return time


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import heapq
import numpy as np
import math

class FastRoadNetwork:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.node_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.adj_matrix = self._build_adjacency()
        self.csr_adj = csr_matrix(self.adj_matrix)

    def _build_adjacency(self):
        n = len(self.nodes)
        adj_matrix = np.full((n, n), np.inf)
        for u, v, data in self.graph.edges(data=True):
            time = data.get('travel_time', 1.0)
            adj_matrix[self.node_idx[u], self.node_idx[v]] = time
        return adj_matrix

    def fast_dijkstra(self, from_node, to_node):
        source_idx = self.node_idx[from_node]
        target_idx = self.node_idx[to_node]

        # noinspection PyTupleAssignmentBalance
        dist_matrix, predecessors = dijkstra(
            csgraph=self.csr_adj,
            directed=True,
            indices=source_idx,
            return_predecessors=True
        )

        travel_time = dist_matrix[target_idx]
        if np.isinf(travel_time):
            return np.inf, np.inf, [], []

        path, segment_times, segment_distances = [], [], []
        get_edge_data = self.graph.get_edge_data

        current = target_idx
        while current != source_idx:
            prev = predecessors[current]
            path.append(self.idx_node[current])

            data = get_edge_data(self.idx_node[prev], self.idx_node[current])
            edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
            segment_times.append(edge.get('travel_time', 1.0) if edge else 1.0)
            segment_distances.append(float(edge.get('length_m', 1.0)) if edge else 1.0)

            current = prev

        path.append(self.idx_node[source_idx])
        path.reverse()
        segment_times.reverse()
        segment_distances.reverse()

        total_distance = sum(segment_distances)
        return travel_time, total_distance, path, segment_times

    def fast_custom_dijkstra(self, from_node, to_node):
        visited = set()
        queue = [(0.0, from_node, [from_node])]
        get_edge_data = self.graph.get_edge_data

        while queue:
            current_time, current_node, path = heapq.heappop(queue)

            if current_node == to_node:
                segment_times, segment_distances = [], []
                for u, v in zip(path[:-1], path[1:]):
                    data = get_edge_data(u, v)
                    edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
                    segment_times.append(edge.get('travel_time', 1.0) if edge else 1.0)
                    segment_distances.append(float(edge.get('length_m', 1.0)) if edge else 1.0)

                total_time = sum(segment_times)
                total_distance = sum(segment_distances)
                return total_time, total_distance, path, segment_times

            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor in self.graph.successors(current_node):
                if neighbor not in visited:
                    data = get_edge_data(current_node, neighbor)
                    edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
                    travel_time = edge.get('travel_time', 1.0) if edge else 1.0
                    heapq.heappush(queue, (current_time + travel_time, neighbor, path + [neighbor]))

        return np.inf, np.inf, [], []

    def fast_astar(self, from_node, to_node):
        visited = set()
        queue = [(0.0, 0.0, from_node, [from_node])]
        get_edge_data = self.graph.get_edge_data

        while queue:
            priority, current_time, current_node, path = heapq.heappop(queue)

            if current_node == to_node:
                segment_times, segment_distances = [], []
                for u, v in zip(path[:-1], path[1:]):
                    data = get_edge_data(u, v)
                    edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
                    segment_times.append(edge.get('travel_time', 1.0) if edge else 1.0)
                    segment_distances.append(float(edge.get('length_m', 1.0)) if edge else 1.0)

                total_time = sum(segment_times)
                total_distance = sum(segment_distances)
                return total_time, total_distance, path, segment_times

            if current_node in visited:
                continue
            visited.add(current_node)

            for neighbor in self.graph.successors(current_node):
                if neighbor not in visited:
                    data = get_edge_data(current_node, neighbor)
                    edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
                    travel_time = edge.get('travel_time', 1.0) if edge else 1.0

                    heuristic = self._haversine_distance(neighbor, to_node) / 13.4  # assume 30 mph ~ 13.4 m/s
                    new_priority = current_time + travel_time + heuristic
                    heapq.heappush(queue, (new_priority, current_time + travel_time, neighbor, path + [neighbor]))

        return np.inf, np.inf, [], []

    def fast_bidirectional_dijkstra(self, from_node, to_node):
        forward_visited = {from_node: (0.0, [from_node])}
        backward_visited = {to_node: (0.0, [to_node])}
        forward_queue = [(0.0, from_node)]
        backward_queue = [(0.0, to_node)]
        best_meeting_node, best_time = None, np.inf
        get_edge_data = self.graph.get_edge_data

        while forward_queue and backward_queue:
            f_time, f_node = heapq.heappop(forward_queue)
            for neighbor in self.graph.successors(f_node):
                data = get_edge_data(f_node, neighbor)
                edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
                travel_time = edge.get('travel_time', 1.0) if edge else 1.0
                new_time = f_time + travel_time

                if neighbor not in forward_visited or new_time < forward_visited[neighbor][0]:
                    forward_visited[neighbor] = (new_time, forward_visited[f_node][1] + [neighbor])
                    heapq.heappush(forward_queue, (new_time, neighbor))
                    if neighbor in backward_visited:
                        total_time = new_time + backward_visited[neighbor][0]
                        if total_time < best_time:
                            best_time = total_time
                            best_meeting_node = neighbor

            b_time, b_node = heapq.heappop(backward_queue)
            for neighbor in self.graph.predecessors(b_node):
                data = get_edge_data(neighbor, b_node)
                edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
                travel_time = edge.get('travel_time', 1.0) if edge else 1.0
                new_time = b_time + travel_time

                if neighbor not in backward_visited or new_time < backward_visited[neighbor][0]:
                    backward_visited[neighbor] = (new_time, backward_visited[b_node][1] + [neighbor])
                    heapq.heappush(backward_queue, (new_time, neighbor))
                    if neighbor in forward_visited:
                        total_time = new_time + forward_visited[neighbor][0]
                        if total_time < best_time:
                            best_time = total_time
                            best_meeting_node = neighbor

        if best_meeting_node is None:
            return np.inf, np.inf, [], []

        forward_path = forward_visited[best_meeting_node][1]
        backward_path = backward_visited[best_meeting_node][1][::-1][1:]  # exclude meeting node duplicate
        path = forward_path + backward_path

        segment_times, segment_distances = [], []
        for u, v in zip(path[:-1], path[1:]):
            data = get_edge_data(u, v)
            edge = (data.get(0) if data and 0 in data else (list(data.values())[0] if data else None))
            segment_times.append(edge.get('travel_time', 1.0) if edge else 1.0)
            segment_distances.append(float(edge.get('length_m', 1.0)) if edge else 1.0)

        total_distance = sum(segment_distances)
        return best_time, total_distance, path, segment_times

    def _haversine_distance(self, node1, node2):
        lon1, lat1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y']
        lon2, lat2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y']

        R = 6371000  # meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RoadNetworkTester:
    def __init__(self, graph=None):
        self.graph = ox.load_graphml(DEFAULT_GRAPH_PATH) if graph is None else graph
        self.fast_network = FastRoadNetwork(self.graph)

    def get_node_coordinate(self, node_id):
        """
        get (lon, lat) position of the given node
        """
        node = self.graph.nodes[node_id] # returns dict
        try:
            return node['x'], node['y']
        except:
            raise Warning(f'queried node does not exist in the graph : {node_id}')

    def find_nearest_node_from_coordinate(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        node = ox.distance.nearest_nodes(self.graph, X=x, Y=y)
        return node

    def find_shortest_path_route(self, origin_node, destination_node):
        route = ox.shortest_path(self.graph, orig=origin_node, dest=destination_node, weight="travel_time")
        return route

    def find_shortest_travel_time(self, origin_node, destination_node):
        time = nx.shortest_path_length(self.graph, source=origin_node, target=destination_node, weight="travel_time")
        return time