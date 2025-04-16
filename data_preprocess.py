import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
from tqdm import tqdm
import os
import json
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from utils.geographic_helper import build_zipcode_bounds, load_zipcode_bounds, get_random_coordinate_cached

PATH_DATA = './data'
PATH_ROAD = './data/road_network'
PATH_TAZ = './data/taz_shape'
PATH_OUTPUT = './output'
PATH_EPISODE = './data/request_episodes'

COLUMN_FILTER_COL = {
    'AppOnOrPassengerDroppedOffZip': 'driver_last_pdo_zip',
    'TripReqDriverZip': 'driver_current_zip',
    'TripReqRequesterZip': 'requester_zip',
    'TripReqDate':'req_date',
    'ReqAcceptedDate':'req_accept_date',
    'ReqAcceptedZip':'driver_zip_at_req_acceptance',
    'PassengerPickupDate': 'pu_date',
    'PassengerPickupZip': 'pu_zip',
    'PassengerDropoffDate': 'do_date',
    'PassengerDropoffZip': 'do_zip',
    'TotalAmountPaid': 'trip_fare',
    'Tip': 'tip',
    'AppOnOrPassengerDroppedOffOSMID': 'driver_last_pdo_osmid',
    'ReqAcceptedOSMID':'req_accept_osmid',
    'PassengerPickupOSMID':'pu_osmid',
    'PassengerDropoffOSMID':'do_osmid',
}

TRIP_TIME_COL = {
    'req_date': 'req_time',
    'req_accept_date': 'accept_time_art',
    'pu_date': 'pu_time_art',
    'do_date': 'do_time_art'
}

FARE_COL = {
    'TotalAmountPaid': 'trip_fare',
    'Tip': 'tip',
}

OUT_COLUMNS = ['pu_osmid', 'do_osmid', 'req_time', 'pu_taz', 'do_taz','pu_lon', 'pu_lat', 'do_lon', 'do_lat',
               'accept_time_art', 'pu_time_art', 'do_time_art', 'trip_fare', 'tip']

# trip time = pax_do - pax_po (pick up -> drop off)
# total trip time = pax_do - req_date (request time -> drop off)
# filter by request date

class DataLoader:
    def __init__(self, request_file_path):

        self.request_df = self.load_request_data(request_file_path, COLUMN_FILTER_COL)
        self.pax_osmid_col = ['pu_osmid', 'do_osmid'] # [list(COLUMN_FILTER_COL.items())[-2][1], list(COLUMN_FILTER_COL.items())[-1][1]] #['passenger_pu_osmid', 'passenger_do_osmid']
        self.pax_zip_col = ['pu_zip', 'do_zip']
        self.pax_taz_col = ['pu_taz', 'do_taz']
        # self.request_df = self.request_df.dropna(subset=drop_col).reset_index(drop=True)

        self.request_df, self.error_df = self.process_datetime(self.request_df, TRIP_TIME_COL)

        if len(self.error_df) > 0:
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            error_output_path = os.path.join(PATH_OUTPUT, f'error_{timestamp}')
            self.error_df.to_csv(error_output_path)
            raise Warning(f'check request error data {error_output_path}')

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
    def process_datetime(request_df, time_column_dict):
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

    def populate_missing_osmid(self, graph, zip_bounds):
        for osmid_col, zip_col in zip(self.pax_osmid_col, self.pax_zip_col):
            missing_mask = self.request_df[osmid_col].isna()

            if missing_mask.any():
                print(f"\nPopulating missing values for {osmid_col} using ZIP column {zip_col}...")

                for idx in tqdm(self.request_df[missing_mask].index, desc=f"Assigning {osmid_col}"):
                    zipcode = self.request_df.at[idx, zip_col]

                    try:
                        lat, lon = get_random_coordinate_cached(zipcode, zip_bounds)
                        nearest_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                        self.request_df.at[idx, osmid_col] = nearest_node

                    except Exception as e:
                        print(f"\t[Warning] Could not assign OSMID for index {idx} (zip: {zipcode}): {e}")

    def populate_missing_osmid_vectorized(self, graph, zip_bounds):

        for osmid_col, zip_col in zip(self.pax_osmid_col, self.pax_zip_col):
            missing_idx = self.request_df[self.request_df[osmid_col].isna()].index

            if len(missing_idx) > 0:
                print(f"Populating missing values for {osmid_col} using zip column {zip_col}...")

                def assign_node(zipcode):
                    try:
                        lat, lon = get_random_coordinate_cached(zipcode, zip_bounds)
                        return ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                    except:
                        return pd.NA

                self.request_df.loc[missing_idx, osmid_col] = (
                    self.request_df.loc[missing_idx, zip_col]
                    .map(assign_node)
                    .astype("Int64")
                )

    def populate_missing_osmid_original(self, graph, zip_bounds):
        for osmid_col, zip_col in zip(self.pax_osmid_col, self.pax_zip_col):
            missing_mask = self.request_df[osmid_col].isna()

            if missing_mask.any():
                print(f"Populating missing values for {osmid_col} using zip column {zip_col}...")

                for idx in self.request_df[missing_mask].index:
                    if idx%100 == 0:
                        print(f"{osmid_col} at index: {idx}")
                    zipcode = self.request_df.at[idx, zip_col]

                    try:
                        lat,lon = get_random_coordinate_cached(zipcode, zip_bounds)
                        nearest_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                        self.request_df.at[idx, osmid_col] = nearest_node

                    except Exception as e:
                        print(f"\t[Warning] Could not assign OSMID for index {idx} (zip: {zipcode}): {e}")

    def populate_unassigned_taz(self, graph, zip_bounds):
        for taz_col, osmid_col, zip_col in zip(self.pax_taz_col, self.pax_osmid_col, self.pax_zip_col):
            missing_mask = self.request_df[taz_col].isna()

            if missing_mask.any():
                print(f"Populating missing values for {taz_col} using ZIP column {zip_col}...")

                for idx in tqdm(self.request_df[missing_mask].index, desc=f"Assigning {taz_col}"):
                    zipcode = self.request_df.at[idx, zip_col]

                    try:
                        lat, lon = get_random_coordinate_cached(zipcode, zip_bounds)
                        nearest_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)

                        self.request_df.at[idx, osmid_col] = nearest_node
                        self.request_df.at[idx, taz_col] = graph.nodes[nearest_node].get('taz', None)

                    except Exception as e:
                        print(f"\t[Warning] Could not assign TAZ for index {idx} (zip: {zipcode}): {e}")

    def populate_unassigned_taz_vectorized(self, graph, zip_bounds):
        """
        Vectorized TAZ assignment using random coordinates from ZIP and nearest OSM node lookup.
        Follows the same logic as `populate_missing_osmid_vectorized`, with added TAZ mapping.
        """
        node_to_taz = nx.get_node_attributes(graph, 'taz')

        for taz_col, osmid_col, zip_col in zip(self.pax_taz_col, self.pax_osmid_col, self.pax_zip_col):
            missing_idx = self.request_df[self.request_df[taz_col].isna()].index

            if len(missing_idx) > 0:
                print(f"Populating missing values for {taz_col} using zip column {zip_col}...")

                def assign_node_and_taz(zipcode):
                    try:
                        lat, lon = get_random_coordinate_cached(zipcode, zip_bounds)
                        node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                        taz = node_to_taz.get(node, pd.NA)
                        return pd.Series([node, taz])
                    except:
                        return pd.Series([pd.NA, pd.NA])

                node_taz_df = self.request_df.loc[missing_idx, zip_col].map(assign_node_and_taz).to_list()
                node_taz_df = pd.DataFrame(node_taz_df, index=missing_idx, columns=[osmid_col, taz_col]).astype(
                    "Int64")

                self.request_df.loc[missing_idx, osmid_col] = node_taz_df[osmid_col]
                self.request_df.loc[missing_idx, taz_col] = node_taz_df[taz_col]

    def populate_unassigned_taz_original(self, graph, zip_bounds):

        for taz_col, osmid_col, zip_col in zip(self.pax_taz_col, self.pax_osmid_col, self.pax_zip_col):
            missing_mask = self.request_df[taz_col].isna()

            if missing_mask.any():
                print(f"Populating missing values for {taz_col} using zip column {zip_col}...")

                for idx in self.request_df[missing_mask].index:
                    if idx%100 == 0:
                        print(f"{taz_col} at index: {idx}")
                    zipcode = self.request_df.at[idx, zip_col]

                    try:
                        lat,lon = get_random_coordinate_cached(zipcode, zip_bounds)
                        nearest_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                        self.request_df.at[idx, osmid_col] = nearest_node
                        self.request_df.at[idx, taz_col] = graph.nodes[nearest_node]['taz']

                    except Exception as e:
                        print(f"\t[Warning] Could not assign TAZ for index {idx} (zip: {zipcode}): {e}")

    def create_zip_bounds(self):
        zipcodes = np.unique(self.request_df[self.pax_zip_col].values.ravel())
        zip_bounds = build_zipcode_bounds(zipcodes)

        return zip_bounds

    def filter_requests_based_on_taz(self, taz_gdf):
        """
        Filters request_df to only include rows where both pickup and dropoff coordinates fall within TAZ zones.
        """
        from shapely.geometry import Point

        # Create GeoDataFrames for PO and DO points
        pu_geom = self.request_df['passenger_pu_pos'].apply(lambda x: Point(x[1], x[0]) if pd.notnull(x) else None)
        do_geom = self.request_df['passenger_do_pos'].apply(lambda x: Point(x[1], x[0]) if pd.notnull(x) else None)

        pu_gdf = gpd.GeoDataFrame(self.request_df.copy(), geometry=pu_geom, crs='EPSG:4326')
        do_gdf = gpd.GeoDataFrame(self.request_df.copy(), geometry=do_geom, crs='EPSG:4326')

        # Spatial join with TAZ for both pickup and dropoff
        pu_in_taz = gpd.sjoin(pu_gdf, taz_gdf, how='inner', predicate='within')
        do_in_taz = gpd.sjoin(do_gdf, taz_gdf, how='inner', predicate='within')

        # Find common indices that exist in both joins
        common_idx = pu_in_taz.index.intersection(do_in_taz.index)

        # Filter original request_df based on common indices
        self.request_df = self.request_df.loc[common_idx].reset_index(drop=True)

    def map_request_to_taz(self, taz_file_path):

        with open(taz_file_path, "r") as f:
            taz_map = json.load(f)

        # reverse look up
        node_to_taz = {}
        for taz_id, node_list in taz_map.items():
            for node in node_list:
                node_to_taz[int(node)] = int(taz_id)

        self.request_df['pu_taz'] = self.request_df['pu_osmid'].map(node_to_taz)
        self.request_df['do_taz'] = self.request_df['do_osmid'].map(node_to_taz)

    def map_request_to_coordinates(self, graph):

        pu_osmid_col = self.pax_osmid_col[0]
        do_osmid_col = self.pax_osmid_col[1]

        # Extract node attributes into lookup dicts
        node_x = nx.get_node_attributes(graph, 'x')
        node_y = nx.get_node_attributes(graph, 'y')

        # Map OSMIDs to coordinates
        self.request_df['pu_lon'] = self.request_df[pu_osmid_col].map(node_x)
        self.request_df['pu_lat'] = self.request_df[pu_osmid_col].map(node_y)

        self.request_df['do_lon'] = self.request_df[do_osmid_col].map(node_x)
        self.request_df['do_lat'] = self.request_df[do_osmid_col].map(node_y)

    def filter_by_time(self, dates=[2019,9,-1], save_weekdays=[1,2,3,4,5], start_time=7*3600, end_time=10*3600):
        """
        Filters request_df based on:
        - Flexible date spec: [year, month(s), day(s)]
        - Only includes weekdays in `save_weekdays`
        - Only keeps rows where req_time is within start_time to end_time
        """
        from collections.abc import Iterable

        def ensure_list(x):
            return list(x) if isinstance(x, Iterable) and not isinstance(x, str) else [x]

        df = self.request_df.copy()

        # Ensure datetime conversion only once
        dt_series = pd.to_datetime(df['req_date'])
        df['req_date_only'] = dt_series.dt.date
        df['weekday'] = dt_series.dt.weekday
        df['year'] = dt_series.dt.year
        df['month'] = dt_series.dt.month
        df['day'] = dt_series.dt.day

        # Apply filters - Build mask from date rules
        date_mask = pd.Series(False, index=df.index)

        years, months, days = dates
        years = ensure_list(years)
        months = ensure_list(months)
        days = ensure_list(days)

        for y in years:
            for m in months:
                for d in days:
                    if d == -1:
                        match = (df['year'] == y) & (df['month'] == m)
                    else:
                        match = (df['year'] == y) & (df['month'] == m) & (df['day'] == d)
                    date_mask |= match

        df = df[
            date_mask &
            (df['weekday'].isin(save_weekdays)) &
            (df['req_time'] >= start_time) &
            (df['req_time'] < end_time)
            ]

        self.request_df = df.sort_values(by="req_date").reset_index(drop=True)
        print(f"filtered by time to dataframe of size {len(self.request_df)}")

    def save_as_episodes(self, save_path=PATH_EPISODE, out_columns=OUT_COLUMNS):
        """
        Saves filtered episodes from request_df into daily CSV files.
        Each file contains requests within a specific time range, grouped by req_date.
        """
        grouped = self.request_df.groupby("req_date_only")

        for date, group in grouped:
            # Filter by request time range
            episode_df = group[out_columns]

            # Save
            filename = f"episode_{date}.csv"
            episode_df.to_csv(os.path.join(save_path, filename), index=False)
            print(f"[✓] Saved: {filename} ({len(episode_df)} rows)")


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
            if (i % 10000) == 0:
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

DEFAULT_GRAPH_PATH = "./data/road_network/sf_road_network.graphml"

def main():
    file_path = os.path.join(PATH_DATA, 'cb_match_osm_data_part1_2.csv') #'sample_data0.csv'
    dl = DataLoader(request_file_path=file_path)
    # rn = RoadNetworkBuilder()
    graph = ox.load_graphml(DEFAULT_GRAPH_PATH)
    zip_bounds = load_zipcode_bounds('./utils/zip_bounds.json')

    # dl.filter_by_time(dates=[2019,9,-1], save_weekdays=[1, 2, 3, 4, 5], start_time=7 * 3600, end_time=10 * 3600)
    dl.filter_by_time(dates=[2019,9,17], save_weekdays=[1, 2, 3, 4, 5], start_time=7 * 3600, end_time=10 * 3600)
    # dl.filter_by_time(dates=[-1, -1, -1], save_weekdays=[1, 2, 3, 4, 5], start_time=7 * 3600, end_time=10 * 3600)
    dl.populate_missing_osmid(graph, zip_bounds)

    dl.map_request_to_taz(taz_file_path="./data/taz_shape/taz_nodes.json")

    dl.populate_unassigned_taz(graph, zip_bounds)

    dl.map_request_to_coordinates(graph)

    dl.save_as_episodes(save_path=PATH_EPISODE,
                        out_columns=OUT_COLUMNS)


    # # filter requests
    # dl.filter_requests_based_on_taz(rn.taz_gdf)
    #
    # # assign osmid
    # dl.assign_osmid_to_requests(rn.nodes_df)

if __name__ == "__main__":
    main()

def filter_date(self, request_df, time_column_dict):
    return 0
