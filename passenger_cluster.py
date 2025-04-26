import pandas as pd
import networkx as nx
import math
from math import radians, sin, cos, sqrt, atan2

class clusterGenerator:
    def __init__(self, simulator, current_request_df):
        self.simulator = simulator
        self.request_df = current_request_df
        self.euclidean_radius = 800  # meters (~0.5 miles)
        self.walking_speed = 1.4     # m/s
        self.max_walk_time = 600     # seconds (e.g. 10 min)

    def update_aggregated_requests(self, aggregated_request_df):
        self.request_df = aggregated_request_df

    def euclidean_distance(self, coord1, coord2):
        R = 6371000  # Earth radius in meters
        lon1, lat1 = map(radians, coord1)
        lon2, lat2 = map(radians, coord2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c 

    def create_subgraph(self):
        """
        1. Create a graph of requests as nodes.
        2. Add edges between nodes with Euclidean distance < threshold.
        3. Refine edges using road network travel time.
        4. Return the final graph.
        """
        G = nx.Graph()
        matches = []

        # Step 1: Identify all request pairs within Euclidean threshold
        for i, row1 in self.request_df.iterrows():
            coord1 = (row1['pu_lon'], row1['pu_lat'])
            for j, row2 in self.request_df.loc[i+1:].iterrows():
                coord2 = (row2['pu_lon'], row2['pu_lat'])
                dist = self.euclidean_distance(coord1, coord2)
                if dist <= self.euclidean_radius:
                    matches.append((i, j))

        # Step 2: Add request nodes and Euclidean edges
        G.add_nodes_from(self.request_df.index)


        # Step 3: Refine edges using shortest travel time in road network
        valid_edges = []
        for i, j in matches:
            try:
                # Pickup 
                pu_osmid_i = self.request_df.at[i, 'pu_osmid']
                pu_osmid_j = self.request_df.at[j, 'pu_osmid']
                pu_travel_time = self.simulator.network.find_shortest_travel_time(pu_osmid_i, pu_osmid_j)

                # Dropoff 
                do_osmid_i = self.request_df.at[i, 'do_osmid']
                do_osmid_j = self.request_df.at[j, 'do_osmid']
                do_travel_time = self.simulator.network.find_shortest_travel_time(do_osmid_i, do_osmid_j)

                # PU vs. DO
                if pu_travel_time <= self.max_walk_time and do_travel_time <= self.max_walk_time:
                    max_travel_time = max(pu_travel_time, do_travel_time)
                    G.add_edge(i, j, weight=max_travel_time)
                    valid_edges.append((i, j))

            except Exception as e:
                print(f"Skipping ({i}, {j}) due to error: {e}")

        # Step 4: Keep only valid edges (filtered by walking time)
        G = G.edge_subgraph(valid_edges).copy()

        return G

def extract_clusters(self):
    """
    output format: for each cluster:
    - request indices
    - representative PU point (lon, lat)
    - representative DO point (lon, lat)
    """
    G = self.create_subgraph()
    clusters = list(nx.connected_components(G))

    output_clusters = []

    for cluster in clusters:
        cluster_list = list(cluster)  

        pu_lons = self.request_df.loc[cluster_list, 'pu_lon'].values
        pu_lats = self.request_df.loc[cluster_list, 'pu_lat'].values
        do_lons = self.request_df.loc[cluster_list, 'do_lon'].values
        do_lats = self.request_df.loc[cluster_list, 'do_lat'].values

        # take centroid
        pu_lon_mean = pu_lons.mean()
        pu_lat_mean = pu_lats.mean()
        do_lon_mean = do_lons.mean()
        do_lat_mean = do_lats.mean()

        # Package everything
        output = {
            'requests': cluster_list,           
            'pickup_point': (pu_lon_mean, pu_lat_mean),   
            'dropoff_point': (do_lon_mean, do_lat_mean)  
        }

        output_clusters.append(output)

    return output_clusters

