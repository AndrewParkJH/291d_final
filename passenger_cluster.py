import pandas as pd
import networkx as nx
import math
from math import radians, sin, cos, sqrt, atan2
from itertools import chain

class clusterGenerator:
    def __init__(self, simulator, current_request_df, euclidean_radius=800, walking_speed=1.2, max_walk_time=600):
        self.simulator = simulator
        self.request_df = current_request_df
        self.euclidean_radius = euclidean_radius  # meters (~0.5 miles)
        self.walking_speed = walking_speed     # m/s
        self.max_walk_time = max_walk_time     # seconds (e.g. 10 min)

    def update_aggregated_requests(self, aggregated_request_df):
        self.request_df = pd.DataFrame(aggregated_request_df)

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
            coord1_pu = (row1['pu_lon'], row1['pu_lat'])
            coord1_do = (row1['do_lon'], row1['do_lat'])

            for j, row2 in self.request_df.loc[i+1:].iterrows():
                coord2_pu = (row2['pu_lon'], row2['pu_lat'])
                coord2_do = (row2['do_lon'], row2['do_lat'])

                dist_pu = self.euclidean_distance(coord1_pu, coord2_pu)
                dist_do = self.euclidean_distance(coord1_do, coord2_do)
                if max(dist_pu, dist_do) <= self.euclidean_radius:
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
                _, pu_travel_distance, _, _ = self.simulator.network.fast_network.fast_custom_dijkstra(pu_osmid_i, pu_osmid_j)
                pu_travel_time = pu_travel_distance/self.walking_speed
                # pu_travel_time = self.simulator.network.find_shortest_travel_time(pu_osmid_i, pu_osmid_j)

                # Dropoff 
                do_osmid_i = self.request_df.at[i, 'do_osmid']
                do_osmid_j = self.request_df.at[j, 'do_osmid']
                _, do_travel_distance, _, _ = self.simulator.network.fast_network.fast_custom_dijkstra(do_osmid_i, do_osmid_j)
                do_travel_time = do_travel_distance / self.walking_speed

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
        groups = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        clusters = []
        assigned_nodes = set()

        for subgraph in groups:
            for clique in nx.find_cliques(subgraph):  # only maximal cliques
                if not any(node in assigned_nodes for node in clique):
                    clusters.append(clique)
                    assigned_nodes.update(clique)

        # Add any remaining unassigned isolated nodes as singleton clusters
            for node in subgraph.nodes():
                if node not in assigned_nodes:
                    clusters.append([node])
                    assigned_nodes.add(node)

        output_clusters = []

        for cluster in clusters:
            
            if any(isinstance(i, list) for i in cluster):
                cluster_list = [item for sublist in cluster for item in sublist]
            else:
                cluster_list = list(chain.from_iterable(cluster)) if isinstance(cluster[0], list) else cluster

            pu_lons = self.request_df.loc[cluster_list, 'pu_lon'].values
            pu_lats = self.request_df.loc[cluster_list, 'pu_lat'].values
            do_lons = self.request_df.loc[cluster_list, 'do_lon'].values
            do_lats = self.request_df.loc[cluster_list, 'do_lat'].values
            pu_time = self.request_df.loc[cluster_list, 'do_lat'].values

            # take centroid
            pu_lon_mean = pu_lons.mean()
            pu_lat_mean = pu_lats.mean()
            do_lon_mean = do_lons.mean()
            do_lat_mean = do_lats.mean()

            # find nearest nodes for the pickup and drop off point
            pu_node = self.simulator.network.find_nearest_node_from_coordinate((pu_lon_mean, pu_lat_mean))
            do_node = self.simulator.network.find_nearest_node_from_coordinate((do_lon_mean, do_lat_mean))

            # find the time at which the cluster is formed (should this be the latest request or end of timestamp?)
            max_req_time = self.request_df.loc[cluster_list, 'req_time'].max()

            # find the time it will take for all requests to arrive at the pickup point
            max_walk_time = max(
                self.simulator.network.fast_network.fast_custom_dijkstra(self.request_df.at[idx, 'pu_osmid'], pu_node)[0]
                for idx in cluster_list
            )

            # Package everything
            output = {
                'requests': cluster_list,
                'num_req': len(cluster_list),
                'pickup_point': (pu_lon_mean, pu_lat_mean),
                'pickup_node': pu_node,
                'pickup_time': max_req_time + max_walk_time,
                'dropoff_point': (do_lon_mean, do_lat_mean),
                'dropoff_node': do_node
            }

            output_clusters.append(output)

        return output_clusters

