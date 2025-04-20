import pandas as pd
import networkx as nx
import math
from math import radians, sin, cos, sqrt, atan2


class clusterGenerator:
    def __init__(self, simulator, current_request_df):
        self.simulator = simulator
        self.request_df = current_request_df
        self.euclidean_radius = 0.5 # acceptable walking mile
        self.walking_speed = 1.4 # average walking speed 1.4 meters/second

    def update_aggregated_requests(self, aggregated_request_df):
        """
        can be called by the simulator after each aggregation time interval (e.g. 120 seconds)
        updates the request df
        """
        self.request_df = aggregated_request_df

    def euclidean_distance(self, coord1, coord2): # calculate it based on latitude and longitude
        R = 6371000  # Earth radius in meters
        lon1, lat1 = map(radians, coord1)
        lon2, lat2 = map(radians, coord2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 # Haversine formula
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c 
    

    def create_subgraph(self):

        G = nx.Graph()
        
        for idx, row in self.request_df.iterrows():
            G.add_node(idx, **row.to_dict())
        """
        pseudo-code
        
        initialize a graph
        add all requests as nodes (from self.request_df)
        
        for each request in request_df    
            draw a circle around the request coordinates
            get all requests within that circle
            
            for each request_in_circle in all_requests_in_circle:
                if no edge between request and request_in_circle:
                    compute distance from request's location to the location of each request_in_circle
                
                    if distance < walking_distance_threshold:
                        add edge between request and request_in_circle.
        Output:
            request-request subgraph with undirected edge if walking distance (and hence time) is within threshold
            ==> connected requests be combined as a single request
        """

        # tip: the functions below will give you the shortest travel distance and time
        node1 = 65317547
        node2 = 4044911147

        distance = self.simulator.network.find_shortest_path_route(node1, node2)
        travel_time_walk = distance/self.walking_speed
        travel_time_drive = self.simulator.network.find_shortest_travel_time(node1, node2)

        return 0
