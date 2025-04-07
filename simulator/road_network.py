import osmnx as ox
import networkx as nx
from simulator.vehicle import Vehicle
import os

DEFAULT_GRAPH_PATH = "./data/road_network/sf_road_network.graphml"

class RoadNetwork:
    def __init__(self, env, num_vehicles=40, vehicle_capacity=10, randomize_vehicles=True, graph_path=DEFAULT_GRAPH_PATH):
        self.env = env
        self.graph = ox.load_graphml(graph_path)
        self.vehicles = self.initialize_vehicle(num_vehicles, vehicle_capacity, randomize_vehicles)
        self.max_vehicle_capacity = vehicle_capacity

        self.one = 0

    def update_congestion(self):
        # update travel time in each TAZ

        return 0

    def initialize_vehicle(self, num_vehicles, vehicle_capacity, randomize_vehicles):
        """
        initializes vehicle positions
        """
        vehicles = []

        for i in range(1, num_vehicles+1):
            vehicle = Vehicle(env=self.env, network=self, vid=i,max_capacity=vehicle_capacity, randomize=randomize_vehicles)
            vehicles.append(vehicle)

        return vehicles

    def return_vehicle_state(self):
        """
        Vehicle state = (current position, destination_position, current_num_passengers)
        """
        state = []
        for vehicle in self.vehicles:
            state.append((vehicle.current_pos, vehicle.next_pos, vehicle.current_num_pax))


    def return_state(self):
        """
        :return: state information
                vehicles in each TAZ
                requests in each TAZ
        """
        return 0

    def get_node_coordinate(self, node_id):
        """
        get (lon, lat) position of the given node
        """
        node = self.graph.nodes[node_id] # returns dict
        try:
            return node['x'], node['y']
        except:
            raise Warning(f'queried node does not exist in the graph : {node_id}')

    def find_nearest_node_from_coodrinate(self, coordinates):
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

