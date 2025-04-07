import pandas as pd
import geopandas as gpd
import datetime as dt
import os

import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from utils.geographic_helper import get_random_coordinate

PATH_DATA = './data'
PATH_ROAD = './data/road_network'
PATH_TAZ = './data/taz_shape'
PATH_OUTPUT = './output'

COLUMN_FILTER_COL = {
    'AppOnOrPassengerDroppedOffZip': 'driver_last_pdo_zip',
    'TripReqDriverZip': 'driver_current_zip',
    'TripReqRequesterZip': 'requester_zip',
    'TripReqDate':'req_date',
    'ReqAcceptedDate':'req_accept_date',
    'ReqAcceptedZip':'driver_zip_at_req_acceptance',
    'PassengerPickupDate': 'passenger_po_date',
    'PassengerPickupZip': 'passenger_po_zip',
    'PassengerDropoffDate': 'passenger_do_date',
    'PassengerDropoffZip': 'passenger_do_zip',
    'TotalAmountPaid': 'trip_fare',
    'Tip': 'tip',
    'AppOnOrPassengerDroppedOffOSMID': 'driver_last_pdo_osmid',
    'ReqAcceptedOSMID':'req_accept_osmid',
    'PassengerPickupOSMID':'passenger_po_osmid',
    'PassengerDropoffOSMID':'passenger_do_osmid',
}

TRIP_TIME_COL = {
    'req_date': 'req_time',
    'req_accept_date': 'accept_time',
    'passenger_po_date': 'po_time',
    'passenger_do_date': 'do_time'
}

FARE_COL = {
    'TotalAmountPaid': 'trip_fare',
    'Tip': 'tip',
}

# trip time = pax_do - pax_po (pick up -> drop off)
# total trip time = pax_do - req_date (request time -> drop off)
# filter by request date

class DataLoader:
    def __init__(self, request_file_path, osm_file_path=PATH_ROAD, randomize_position=True):

        self.request_df = self.load_request_data(request_file_path, COLUMN_FILTER_COL)
        drop_col = [list(COLUMN_FILTER_COL.items())[-2][1], list(COLUMN_FILTER_COL.items())[-1][1]]
        self.request_df = self.request_df.dropna(subset=drop_col).reset_index(drop=True)

        self.request_df, self.error_df = self.process_datatime(self.request_df, TRIP_TIME_COL)

        if len(self.error_df) > 0:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            error_output_path = os.path.join(PATH_OUTPUT, f'error_{timestamp}')
            self.error_df.to_csv(error_output_path)
            raise Warning(f'check request error data {error_output_path}')

        if randomize_position:
            self.randomize_req_coordinate_based_on_zip()

    @staticmethod
    def load_request_data(file_path, column_rename_dict):
        df = pd.read_csv(file_path, low_memory=False)
        # Filter columns: only keep columns that are in the keys of column_rename_dict
        columns_to_keep = list(column_rename_dict.keys())
        df_filtered = df[columns_to_keep]
        # Rename columns
        df_renamed = df_filtered.rename(columns=column_rename_dict)

        # cast to integers
        exclude_cols = [col for _, col in FARE_COL.items()] + [key for key, _ in TRIP_TIME_COL.items()]
        for col in df_renamed.columns:
            if col not in exclude_cols:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce').astype('Int64')
        return df_renamed

    @staticmethod
    def process_datatime(request_df, time_column_dict):
        # time_column = list(time_column_dict.keys())

        req_date_col = list(time_column_dict.items())[0][0]
        req_time_col = list(time_column_dict.items())[0][1]

        for column, time_column in time_column_dict.items():
            request_df[column] = pd.to_datetime(request_df[column], format="%Y%m%d %H:%M:%S")

            request_df[time_column] = (
                request_df[column].dt.hour * 3600 +
                request_df[column].dt.minute * 60 +
                request_df[column].dt.second
            )

        for date_column, time_column in list(time_column_dict.items())[1:]:
            day_diff = (request_df[date_column].dt.normalize() - request_df[req_date_col].dt.normalize()).dt.days

            # Add 86400 seconds (1 day) to time_column if over midnight
            adjusted_time = request_df[time_column] + day_diff.astype(int) * 86400

            # Normalize against request time
            request_df[time_column] = adjusted_time - request_df[req_time_col]

        # Check for negative time deltas
        error_mask = None
        for _, time_column in list(time_column_dict.items())[1:]:
            if error_mask is None:
                error_mask = request_df[time_column] < 0
            else:
                error_mask |= request_df[time_column] < 0

        # Create error dataframe and clean the original
        error_checker_df = request_df[error_mask].copy()
        request_df = request_df[~error_mask].copy()

        return request_df, error_checker_df

    def randomize_req_coordinate_based_on_zip(self):
        """
        Randomly generate lat/lon coordinates within pickup and dropoff zip code boundaries.
        Adds two new columns: 'passenger_po_pos' and 'passenger_do_pos'
        """
        def safe_random(zip_code):
            try:
                return get_random_coordinate(str(zip_code))
            except Exception as e:
                print(f"[ZIP ERROR] Zip {zip_code} failed: {e}")
                return None

        print("randomized requests po and do position")
        self.request_df['passenger_po_pos'] = self.request_df['passenger_po_zip'].apply(safe_random)
        self.request_df['passenger_do_pos'] = self.request_df['passenger_do_zip'].apply(safe_random)

    def filter_requests_based_on_taz(self, taz_gdf):
        """
        Filters request_df to only include rows where both pickup and dropoff coordinates fall within TAZ zones.
        """
        from shapely.geometry import Point

        # Create GeoDataFrames for PO and DO points
        po_geom = self.request_df['passenger_po_pos'].apply(lambda x: Point(x[1], x[0]) if pd.notnull(x) else None)
        do_geom = self.request_df['passenger_do_pos'].apply(lambda x: Point(x[1], x[0]) if pd.notnull(x) else None)

        po_gdf = gpd.GeoDataFrame(self.request_df.copy(), geometry=po_geom, crs='EPSG:4326')
        do_gdf = gpd.GeoDataFrame(self.request_df.copy(), geometry=do_geom, crs='EPSG:4326')

        # Spatial join with TAZ for both pickup and dropoff
        po_in_taz = gpd.sjoin(po_gdf, taz_gdf, how='inner', predicate='within')
        do_in_taz = gpd.sjoin(do_gdf, taz_gdf, how='inner', predicate='within')

        # Find common indices that exist in both joins
        common_idx = po_in_taz.index.intersection(do_in_taz.index)

        # Filter original request_df based on common indices
        self.request_df = self.request_df.loc[common_idx].reset_index(drop=True)

    def assign_osmid_to_requests(self, nodes_df):
        """
        Assigns the nearest OSMID to passenger_po_pos and passenger_do_pos using KDTree.
        Adds two new columns: passenger_po_osmid_nearest and passenger_do_osmid_nearest.
        """
        from scipy.spatial import KDTree
        import numpy as np

        # Build KDTree from node coordinates (lon, lat)
        node_coords = nodes_df[['x', 'y']].values  # (lon, lat)
        node_osmids = nodes_df['osmid'].values
        kdtree = KDTree(node_coords)

        def find_nearest_osmid(pos):
            if pd.isnull(pos):
                return None
            lon, lat = pos[1], pos[0]  # (lat, lon) → (lon, lat)
            dist, idx = kdtree.query([lon, lat])
            return node_osmids[idx]

        self.request_df['passenger_po_osmid_nearest'] = self.request_df['passenger_po_pos'].apply(find_nearest_osmid)
        self.request_df['passenger_do_osmid_nearest'] = self.request_df['passenger_do_pos'].apply(find_nearest_osmid)


class RoadNetworkBuilder:
    def __init__(self, road_path=PATH_ROAD, taz_path=PATH_TAZ):

        self.nodes_df = pd.read_csv(os.path.join(road_path, 'nodes.csv'))
        self.edges_df = pd.read_csv(os.path.join(road_path, 'edges.csv'))
        self.taz_gdf = gpd.read_file(os.path.join(taz_path,'sf_taz.geojson')).to_crs("EPSG:4326") # read taz in GeoDataFrame
        self.G = self._build_graph()

    def _build_graph(self, save=True):
        G = nx.MultiDiGraph()

        print(f"Detected node number: {len(self.nodes_df)} and edge number: {len(self.edges_df)}")

        # Optional progress print every 10,000 rows
        for i, row in self.nodes_df.iterrows():
            if i % 10000 == 0:
                print(f"\tBuilding nodes - osmid at {i}")

            G.add_node(str(row['osmid']),
                       x=str(row['x']) if pd.notna(row['x']) else '',
                       y=str(row['y']) if pd.notna(row['y']) else '',
                       taz=str(row['taz']) if pd.notna(row['taz']) else '',
                       gacres=str(row['gacres']) if pd.notna(row['gacres']) else '')

        # Progress logging for edges
        for i, row in self.edges_df.iterrows():
            if i % 10000 == 0:
                print(f"\tBuilding edges - osmid at {i}")

            # Add edges with attributes

            u = str(row['osmid_u'])
            v = str(row['osmid_v'])

            travel_time = ((row['length']/1609.34)/row['speed_mph'])*3600 # ([m/[m/mile]] / [1/mile/hour])*[second/hour] == travel time in seconds

            edge_data = {
                'travel_time': str(travel_time),
                'length_m': str(row['length']) if pd.notna(row['length']) else '',
                'lanes': str(row['lanes']) if pd.notna(row['lanes']) else '',
                'speed_mph': str(row['speed_mph']) if pd.notna(row['speed_mph']) else '',
                'u': str(row['u']) if pd.notna(row['u']) else '',
                'v': str(row['v']) if pd.notna(row['v']) else '',
                'lane_mile': str(row['lane_mile']) if pd.notna(row['lane_mile']) else '',
                'taz_u': str(row['taz_u']) if pd.notna(row['taz_u']) else '',
                'taz_v': str(row['taz_v']) if pd.notna(row['taz_v']) else '',
                'taz': str(row['taz']) if pd.notna(row['taz']) else ''
            }

            G.add_edge(u, v, **edge_data)

        G.graph['crs'] = 'EPSG:4326'

        if save:
            save_path = os.path.join(PATH_ROAD, 'sf_road_network.graphml')
            nx.write_graphml(G, save_path)
            print(f"GraphML file saved to: {save_path}")

        return G

    def visualize_network(self, output_path='./output/network.png'):
        """
        Visualize and save the road network graph.
        """
        output_path = os.path.join(PATH_OUTPUT, f'network_{dt.datetime.now():%Y%m%d_%H%M%S}.png')

        # G_plot = ox.simplify_graph(self.G)
        # G_plot = nx.MultiDiGraph(ox.simplify_graph(self.G))

        fig, ax = ox.plot_graph(
            self.G,
            show=False,
            close=False,
            node_size=5,
            edge_color="gray",
            bgcolor="white"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def visualize_requests(self, request_points):
        pos = {node: (data['x'], data['y']) for node, data in self.G.nodes(data=True)}
        plt.figure(figsize=(10, 8))
        nx.draw(self.G, pos, node_size=10, edge_color='gray', node_color='blue', alpha=0.5)

        # Plot request points
        for lat, lon in request_points:
            plt.plot(lon, lat, 'ro')  # red dots for requests

        plt.title("Road Network with Requests")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

def map_osmid_to_coordinates(request_df, nodes_df):
    import networkx as nx

    osmid_columns = [
        'passenger_po_osmid',
        'passenger_do_osmid'
    ]

    for col in osmid_columns:
        # Define the new column name
        coord_col = col.replace('_osmid', '_coordinate')

        # Map OSMID to (lat, lon) using the graph
        def get_coord(osmid):
            try:
                node_data = nodes_df[nodes_df.osmid==int(osmid)]
                coordinates = (node_data['y'].iloc[0], node_data['x'].iloc[0])
                return coordinates  # (lat, lon)
            except IndexError:
                return None  # OSM ID not in graph


        request_df[coord_col] = request_df[col].apply(get_coord)

    return request_df


def main():
    file_path = os.path.join(PATH_DATA, 'sample_data0.csv')
    dl = DataLoader(request_file_path=file_path)
    rn = RoadNetworkBuilder()
    # rn.visualize_network()

    nodes_df = pd.read_csv(os.path.join(PATH_ROAD, 'nodes.csv'))
    edges_df = pd.read_csv(os.path.join(PATH_ROAD, 'edges.csv'))

    request_df = map_osmid_to_coordinates(dl.request_df, nodes_df)

    # filter requests
    dl.filter_requests_based_on_taz(rn.taz_gdf)

    # assign osmid
    dl.assign_osmid_to_requests(rn.nodes_df)

if __name__ == "data_preprocess":
    # main()
    rn = RoadNetworkBuilder()
    print("data_preprocess")

rn = RoadNetworkBuilder()

def filter_date(self, request_df, time_column_dict):
    return 0
